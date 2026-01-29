#!/usr/bin/env python3
"""
Python基础 - NumPy数组基础
学习NumPy数组的创建、属性和基本操作
"""

import numpy as np

print("="*60)
print("第五课：NumPy数组基础")
print("="*60)

# ============================================================
# 1. 为什么要用NumPy
# ============================================================
print("\n【1. 为什么要用NumPy】")
print("NumPy是Python科学计算的基础库")

print("\nNumPy的优势:")
print("  1. 速度快：底层用C语言实现")
print("  2. 内存效率高：数组存储更紧凑")
print("  3. 向量化运算：避免Python循环")
print("  4. 功能丰富：数学、统计、线性代数")

# 速度对比示例
print("\n速度对比:")
# Python列表
python_list = list(range(1000000))
# NumPy数组
numpy_array = np.arange(1000000)

print(f"  Python列表大小: ~{python_list.__sizeof__() // 1024} KB")
print(f"  NumPy数组大小: ~{numpy_array.nbytes // 1024} KB")
print("  NumPy数组运算速度通常快10-100倍")

# ============================================================
# 2. 创建NumPy数组
# ============================================================
print("\n【2. 创建NumPy数组】")

# 从Python列表创建
print("\n从列表创建:")
list1d = [1, 2, 3, 4, 5]
arr1d = np.array(list1d)
print(f"  列表: {list1d}")
print(f"  数组: {arr1d}")
print(f"  类型: {type(arr1d)}")

# 创建二维数组
list2d = [[1, 2, 3], [4, 5, 6]]
arr2d = np.array(list2d)
print(f"\n二维数组:\n{arr2d}")

# 使用arange创建
print("\n使用arange创建:")
arr_range = np.arange(10)  # 0到9
print(f"  np.arange(10): {arr_range}")

arr_range2 = np.arange(5, 15)  # 5到14
print(f"  np.arange(5, 15): {arr_range2}")

arr_range3 = np.arange(0, 1, 0.1)  # 步长0.1
print(f"  np.arange(0, 1, 0.1): {arr_range3}")

# 使用linspace创建
print("\n使用linspace创建:")
arr_linspace = np.linspace(0, 1, 11)  # 0到1，11个点
print(f"  np.linspace(0, 1, 11): {arr_linspace}")

# 时间轴（EMG常用）
time = np.linspace(0, 5, 5000)  # 5秒，采样率1000Hz
print(f"  时间轴（0-5秒，5000点）: 前10个值: {time[:10]}")

# 创建特殊数组
print("\n创建特殊数组:")
zeros = np.zeros(5)
print(f"  zeros(5): {zeros}")

ones = np.ones(5)
print(f"  ones(5): {ones}")

zeros_2d = np.zeros((3, 4))
print(f"  zeros((3, 4)):\n{zeros_2d}")

full_array = np.full(5, 7)
print(f"  full(5, 7): {full_array}")

# 创建单位矩阵
eye = np.eye(3)
print(f"  eye(3) 单位矩阵:\n{eye}")

# 创建随机数组
print("\n创建随机数组:")
random_uniform = np.random.rand(5)  # 0-1均匀分布
print(f"  rand(5): {random_uniform}")

random_normal = np.random.randn(5)  # 标准正态分布
print(f"  randn(5): {random_normal}")

random_int = np.random.randint(0, 10, 5)  # 0-9随机整数
print(f"  randint(0, 10, 5): {random_int}")

# ============================================================
# 3. 数组属性
# ============================================================
print("\n【3. 数组属性】")

# 创建示例数组
emg_signal = np.random.randn(1000, 4)  # 1000个样本，4个通道

print(f"\nEMG信号数组:")
print(f"  形状 (shape): {emg_signal.shape}")
print(f"  维度 (ndim): {emg_signal.ndim}")
print(f"  大小 (size): {emg_signal.size}")
print(f"  数据类型 (dtype): {emg_signal.dtype}")
print(f"  每个元素字节数 (itemsize): {emg_signal.itemsize}")
print(f"  总字节数 (nbytes): {emg_signal.nbytes}")

# 数据类型
print("\n常用数据类型:")
int_array = np.array([1, 2, 3], dtype=np.int32)
float_array = np.array([1, 2, 3], dtype=np.float64)
print(f"  int32: {int_array.dtype}")
print(f"  float64: {float_array.dtype}")

# ============================================================
# 4. 数组索引和切片
# ============================================================
print("\n【4. 数组索引和切片】")

# 一维数组索引
arr = np.array([10, 20, 30, 40, 50])
print(f"\n一维数组: {arr}")
print(f"  arr[0] = {arr[0]}")
print(f"  arr[-1] = {arr[-1]}")
print(f"  arr[1:4] = {arr[1:4]}")
print(f"  arr[::2] = {arr[::2]}")

# 二维数组索引
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"\n二维数组:\n{matrix}")
print(f"  matrix[0, 0] = {matrix[0, 0]}")
print(f"  matrix[1, 2] = {matrix[1, 2]}")
print(f"  matrix[0, :] 第0行 = {matrix[0, :]}")
print(f"  matrix[:, 1] 第1列 = {matrix[:, 1]}")
print(f"  matrix[0:2, 1:3] 子矩阵:\n{matrix[0:2, 1:3]}")

# 布尔索引
print("\n布尔索引:")
data = np.array([0.1, 0.5, 0.8, 0.3, 0.9])
print(f"  data = {data}")
mask = data > 0.5
print(f"  mask (data > 0.5) = {mask}")
print(f"  data[data > 0.5] = {data[data > 0.5]}")

# ============================================================
# 5. 数组运算
# ============================================================
print("\n【5. 数组运算】")

# 算术运算
a = np.array([1, 2, 3, 4])
b = np.array([10, 20, 30, 40])

print(f"\na = {a}")
print(f"b = {b}")
print(f"  a + b = {a + b}")
print(f"  a - b = {a - b}")
print(f"  a * b = {a * b}")
print(f"  a / b = {a / b}")
print(f"  a ** 2 = {a ** 2}")

# 标量运算
print(f"\n标量运算:")
print(f"  a + 10 = {a + 10}")
print(f"  a * 2 = {a * 2}")

# 数学函数
x = np.array([0, np.pi/2, np.pi])
print(f"\nx = {x}")
print(f"  sin(x) = {np.sin(x)}")
print(f"  cos(x) = {np.cos(x)}")
print(f"  exp(x) = {np.exp(x)}")
print(f"  sqrt(a) = {np.sqrt(a)}")
print(f"  abs([-1, -2, 3]) = {np.abs(np.array([-1, -2, 3]))}")

# ============================================================
# 6. 聚合函数
# ============================================================
print("\n【6. 聚合函数】")

data = np.array([0.5, 0.8, 1.2, 0.3, 0.9])
print(f"\ndata = {data}")
print(f"  sum: {np.sum(data)}")
print(f"  mean: {np.mean(data)}")
print(f"  std: {np.std(data)}")
print(f"  var: {np.var(data)}")
print(f"  min: {np.min(data)}")
print(f"  max: {np.max(data)}")
print(f"  argmin: {np.argmin(data)}")  # 最小值索引
print(f"  argmax: {np.argmax(data)}")  # 最大值索引

# 多维数组的聚合
matrix = np.array([[1, 2, 3], [4, 5, 6]])
print(f"\nmatrix:\n{matrix}")
print(f"  全部求和: {np.sum(matrix)}")
print(f"  按行求和 (axis=1): {np.sum(matrix, axis=1)}")
print(f"  按列求和 (axis=0): {np.sum(matrix, axis=0)}")

# ============================================================
# 7. 数组形状操作
# ============================================================
print("\n【7. 数组形状操作】")

# reshape
arr = np.arange(12)
print(f"\n原始数组: {arr}")
print(f"  reshape(3, 4):\n{arr.reshape(3, 4)}")
print(f"  reshape(2, 6):\n{arr.reshape(2, 6)}")

# flatten和ravel
matrix = np.array([[1, 2, 3], [4, 5, 6]])
print(f"\nmatrix:\n{matrix}")
print(f"  flatten(): {matrix.flatten()}")

# 转置
print(f"  转置 T:\n{matrix.T}")

# ============================================================
# 实践练习
# ============================================================
print("\n" + "="*60)
print("实践练习")
print("="*60)

print("\n练习1: 创建模拟EMG信号")
# 5秒，采样率1000Hz
fs = 1000
duration = 5
n_samples = fs * duration
time = np.linspace(0, duration, n_samples)
# 模拟信号：低频噪声 + 高频成分
signal = 0.05 * np.sin(2*np.pi*50*time) + 0.1 * np.random.randn(n_samples)
print(f"EMG信号: {n_samples}个样本")
print(f"  前10个值: {signal[:10]}")
print(f"  平均值: {np.mean(signal):.4f}")
print(f"  标准差: {np.std(signal):.4f}")

print("\n练习2: 计算RMS（均方根）")
rms = np.sqrt(np.mean(signal ** 2))
print(f"RMS: {rms:.4f}")

print("\n练习3: 找出超过阈值的样本")
threshold = 0.2
above_threshold = np.sum(np.abs(signal) > threshold)
print(f"阈值: {threshold}")
print(f"超过阈值的样本数: {above_threshold}")
print(f"比例: {above_threshold / n_samples * 100:.2f}%")

print("\n练习4: 多通道EMG数据")
n_channels = 4
multi_channel = np.random.randn(1000, n_channels)
print(f"多通道数据形状: {multi_channel.shape}")
print(f"  每个通道的均值: {np.mean(multi_channel, axis=0)}")
print(f"  每个通道的标准差: {np.std(multi_channel, axis=0)}")

print("\n练习5: 归一化到[0, 1]")
data = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
normalized = (data - np.min(data)) / (np.max(data) - np.min(data))
print(f"原始数据: {data}")
print(f"归一化: {normalized}")

# ============================================================
# 课后作业
# ============================================================
print("\n" + "="*60)
print("课后作业")
print("="*60)

print("\n请完成以下作业:")
print("1. 创建一个1000点的随机数组（标准正态分布）")
print("   计算其平均值、标准差、最大值、最小值")
print("2. 创建一个5x5的随机矩阵")
print("   计算每行的和、每列的和")
print("3. 创建数组[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]")
print("   筛选出所有大于5的元素")
print("4. 模拟EMG信号（1秒，采样率1000Hz）")
print("   包含50Hz正弦波和随机噪声")
print("   计算MAV和RMS")
print("5. 创建一个10x4的随机矩阵")
print("   将其归一化到[0, 1]范围")

# ============================================================
# 总结
# ============================================================
print("\n" + "="*60)
print("本课总结")
print("="*60)

print("\n核心要点:")
print("1. NumPy数组比Python列表快、省内存")
print("2. 创建数组: array, arange, linspace, zeros, ones, random")
print("3. 数组属性: shape, ndim, size, dtype")
print("4. 索引和切片: arr[i], arr[i:j], arr[mask]")
print("5. 运算: +, -, *, /, **, sin, cos, exp, sqrt")
print("6. 聚合: sum, mean, std, min, max")
print("7. 形状: reshape, flatten, T")

print("\nEMG应用:")
print("- 使用linspace创建时间轴")
print("- 使用random生成模拟信号")
print("- 使用mean, std计算统计特征")
print("- 使用布尔索引筛选数据")
print("- 使用axis参数处理多通道数据")

print("\n下一课: 06_numpy_indexing.py - NumPy索引和切片详解")
print("="*60)
