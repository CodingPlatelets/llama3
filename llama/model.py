# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import os
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import fairscale.nn.model_parallel.initialize as fs_init
import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from torch import nn


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000

    max_batch_size: int = 32
    max_seq_len: int = 2048


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

def repeat_weight_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    first_dim, second_dim, third_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:,None,:,:]
        .expand(first_dim, n_rep, second_dim, third_dim)
        .reshape(first_dim * n_rep, second_dim, third_dim)
    )

class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        model_parallel_size = fs_init.get_model_parallel_world_size()
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wk = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wv = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wo = RowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
        )

        self.cache_k = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        ).cuda()
        self.cache_v = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        ).cuda()

        # cache x value for each layer
        self.cache_x = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                args.dim,
            )
        ).cuda()

    def forward(
        self,
        layer_id,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        bsz, seqlen, _ = x.shape
        self.cache_x[:bsz, start_pos : start_pos + seqlen] = x
        
        '''if seqlen == 1:
            x_uint16 = x.cpu().view(torch.uint16)

            # 打印每个 bfloat16 数值的二进制表示
            for i, value in enumerate(x_uint16.numpy().flatten()):
                if i == 0:
                    bin_repr = f"{int(value):016b}"  # 转换为 16-bit 二进制字符串
                    print(f"Index {i}: {bin_repr} (Hex: {int(value):04X})")
            exit()
        if seqlen == 1:
            try:
                os.makedirs("debug_output", exist_ok=True)
                with open('debug_output/x_value.bin', 'wb') as f:
                    x_flat = x.reshape(-1)
                    x.cpu().to(torch.float32).numpy().tofile(f)
                exit()
            except Exception as e:
                print(f"保存x数据时出错: {e}")
                exit()'''
        
        # 获取两个权重矩阵的值
        '''try:
            os.makedirs("debug_output", exist_ok=True)
            with open('debug_output/wq_weight.bin', 'wb') as f:
                self.wq.weight.data.cpu().to(torch.float32).numpy().tofile(f)
            with open('debug_output/wk_weight.bin', 'wb') as f:
                wk_weight = self.wk.weight.data.view(8,128,4096)
                wk_weight = repeat_weight_kv(wk_weight, 4)
                wk_weight = wk_weight.view(4096,4096)
                wk_weight.cpu().to(torch.float32).numpy().tofile(f)
            with open('debug_output/wv_weight.bin', 'wb') as f:
                wv_weight = self.wv.weight.data.view(8,128,4096)
                wv_weight = repeat_weight_kv(wv_weight, 4)
                wv_weight = wv_weight.view(4096,4096)
                wv_weight.cpu().to(torch.float32).numpy().tofile(f)
            exit()
        except Exception as e:
            print(f"保存权重数据时出错: {e}")
            exit()'''
        
        # new pattern
        if seqlen == 1:
            print(f"x value: {x[:,:,-1]}")
            xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
            xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)

            #########################################################
            xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
            xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
            self.cache_k = self.cache_k.to(xq)
            self.cache_v = self.cache_v.to(xq)

            self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
            self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

            keys = self.cache_k[:bsz, : start_pos + seqlen]
            values = self.cache_v[:bsz, : start_pos + seqlen]
            values = repeat_kv(values, self.n_rep)  
            #########################################################

            # 复制wk权重
            wk_data = self.wk.weight.data.view(self.n_kv_heads,self.head_dim, self.n_local_heads * self.head_dim)
            wk_data = repeat_weight_kv(wk_data, self.n_rep)
            wk_data = wk_data.view(4096,4096)
            wk_data = wk_data.transpose(0,1)
            wk_data = wk_data.view(4096, 32, 128)
            wk_data = wk_data.transpose(0,1)
            temp1 = torch.matmul(xq.transpose(1,2), wk_data.transpose(1,2))
            # temp1 shape: (1, 32, 1, 4096)

            x_pre = self.cache_x[:bsz, : start_pos + seqlen]
            x_pre = x_pre.view(bsz, 20, 4096)
            new_scores = torch.matmul(temp1, x_pre.transpose(1,2)) / math.sqrt(self.head_dim)
            new_scores = F.softmax(new_scores.float(), dim=-1).type_as(xq)
            
            
            new_output = torch.matmul(new_scores, values.transpose(1,2))  # (bs, n_local_heads, seqlen, head_dim)
            new_output = new_output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
            
            import numpy as np
            np.set_printoptions(threshold=np.inf)
            new_output_np = new_output.cpu().to(torch.float32).detach().numpy()
            with open('debug_output/new_sv_result.txt', 'w') as f:
                f.write(f"Scores shape: {new_output.shape}\n")
                f.write(str(new_output_np))
            exit()
            output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
            os.makedirs("debug_output", exist_ok=True)
            import numpy as np
            scores_np = new_scores.cpu().to(torch.float32).detach().numpy()
                # 如果需要以文本形式保存
            with open('debug_output/new_qk_result.txt', 'w') as text_file:
                text_file.write(f"Scores shape: {new_scores.shape}\n")
                text_file.write(str(scores_np))
            exit()

        
        # 继续正常的前向传播
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        #xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)

        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(
            keys, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        values = repeat_kv(
            values, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        values = values.transpose(
            1, 2
        )  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        # socres shape: (bs, n_local_heads, seqlen, cache_len + seqlen)

        #测试两种方式结果是否一样
        # original pattern 
        '''if seqlen == 1:
            print(f"x value: {x[:,:,-1]}")
            os.makedirs("debug_output", exist_ok=True)
            import numpy as np
            scores_np = scores.cpu().to(torch.float32).detach().numpy()
                # 如果需要以文本形式保存
            with open('debug_output/original_qk_result.txt', 'w') as text_file:
                text_file.write(f"Scores shape: {scores.shape}\n")
                text_file.write(str(scores_np))
            exit()'''
        
        
            

        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )
        self.w2 = RowParallelLinear(
            hidden_dim, dim, bias=False, input_is_parallel=True, init_method=lambda x: x
        )
        self.w3 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        layer_id,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        h = x + self.attention(layer_id, self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = VocabParallelEmbedding(
            params.vocab_size, params.dim, init_method=lambda x: x
        )

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = ColumnParallelLinear(
            params.dim, params.vocab_size, bias=False, init_method=lambda x: x
        )

        self.freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads,
            params.max_seq_len * 2,
            params.rope_theta,
        )

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)

            mask = torch.triu(mask, diagonal=1)

            # When performing key-value caching, we compute the attention scores
            # only for the new sequence. Thus, the matrix of scores is of size
            # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
            # j > cache_len + i, since row i corresponds to token cache_len + i.
            mask = torch.hstack(
                [torch.zeros((seqlen, start_pos), device=tokens.device), mask]
            ).type_as(h)
        
        layer_id = 0

        for layer in self.layers:
            h = layer(layer_id, h, start_pos, freqs_cis, mask)
            layer_id += 1
        h = self.norm(h)
        output = self.output(h).float()
        return output