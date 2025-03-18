import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import re

# 从文本文件中解析张量数据
def parse_tensor_from_text(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    
    # 提取形状信息
    shape_match = re.search(r'Scores shape: torch.Size\(\[(.*?)\]\)', content)
    if shape_match:
        shape_str = shape_match.group(1)
        shape = [int(dim) for dim in shape_str.split(',')]
    else:
        raise ValueError(f"无法从文件中提取张量形状: {file_path}")
    
    # 提取数值数据
    # 使用正则表达式匹配所有数值
    numbers = re.findall(r'[-+]?\d*\.\d+e[-+]?\d+|[-+]?\d+\.\d+|\d+', content)
    numbers = [float(num) for num in numbers if num != '']
    
    # 跳过形状信息中的数字
    numbers = numbers[len(shape):]
    
    # 重塑为正确的形状
    tensor = torch.tensor(numbers).reshape(shape)
    return tensor

# 文件路径
file_path1 = "/home/zb/models/llama3/debug_output/new_sf_result.txt"
file_path2 = "/home/zb/models/llama3/debug_output/original_sf_result.txt"

# 解析张量数据
try:
    tensor1 = parse_tensor_from_text(file_path1)
    tensor2 = parse_tensor_from_text(file_path2)
    
    # 确认张量形状
    print(f"新SF结果形状: {tensor1.shape}")
    print(f"原始SF结果形状: {tensor2.shape}")
    
    # 计算差值
    diff = tensor1 - tensor2  # 直接相减
    diff = diff.squeeze(0).squeeze(1)  # 移除第一维和第三维，结果形状为[32,20]
    
    
    # 计算统计信息
    abs_diff = torch.abs(diff)
    max_diff = torch.max(abs_diff).item()
    mean_diff = torch.mean(abs_diff).item()
    print(f"最大绝对差值: {max_diff:.6f}")
    print(f"平均绝对差值: {mean_diff:.6f}")
    
    # 创建绝对差值热力图
    plt.figure(figsize=(24, 20))
    sns.heatmap(abs_diff.numpy(), cmap="Reds", annot=True, fmt=".5f")
    plt.title("新旧QK结果绝对差值热力图")
    plt.xlabel("元素索引 (最后一维)")
    plt.ylabel("注意力头索引 (第二维)")
    plt.tight_layout()
    plt.savefig("sf_abs_difference_heatmap.png")
    plt.show()

    # 计算相对差异（绝对差值与tensor1绝对值的比例）
    abs_tensor1 = torch.abs(tensor1.squeeze(0).squeeze(1))  # 移除第一维和第三维，与diff形状一致
    
    # 避免除以零
    epsilon = 1e-10
    relative_diff = abs_diff / (abs_tensor1 + epsilon)
    
    # 创建相对差异热力图（百分比表示）
    plt.figure(figsize=(24, 20))
    # 将相对差异转换为百分比（0-100%）
    relative_diff_percent = relative_diff * 100
    # 设置热力图范围为0%-100%
    sns.heatmap(relative_diff_percent.numpy(), cmap="YlOrRd", annot=True, fmt=".2f", vmin=0, vmax=50)
    plt.title("绝对差值占新QK结果绝对值的百分比热力图")
    plt.xlabel("元素索引 (最后一维)")
    plt.ylabel("注意力头索引 (第二维)")
    plt.tight_layout()
    plt.savefig("sf_relative_difference_heatmap.png")
    plt.show()
    
except Exception as e:
    print(f"处理文件时出错: {e}")