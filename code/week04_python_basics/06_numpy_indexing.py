#!/usr/bin/env python3
"""
Python基础 - NumPy索引和切片
深入学习NumPy数组的索引、切片和高级索引技术
"""

import numpy as np

print("="*60)
print("第六课：NumPy索引和切片")
print("="*60)

# ============================================================
# 1. 一维数组索引
# ============================================================
print("\n【1. 一维数组索引】")

arr = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
print(f"\n数组: {arr}")

# 基本索引
print("\n基本索引:")
print(f"  arr[0] = {arr[0]}")  # 第一个元素
print(f"  arr[5] = {arr[5]}")  # 第六个元素
print(f"  arr[-1] = {arr[-1]}")  # 最后一个元素
print(f"  arr[-2] = {arr[-2]}")  # 倒数第二个

# 修改元素
arr[0] = 999
print(f"\n修改 arr[0] = 999: {arr}")
arr[0] = 10  # 改回来

# ============================================================
# 2. 一维数组切片
# ============================================================
print("\n【2. 一维数组切片】")
print("切片语法: arr[start:stop:step]")

print(f"\n数组: {arr}")
print(f"  arr[2:5] = {arr[2:5]}")  # 索引2到4
print(f"  arr[:5] = {arr[:5]}")  # 前5个
print(f"  arr[5:] = {arr[5:]}")  # 从索引5到结尾
print(f"  arr[::2] = {arr[::2]}")  # 每隔一个
print(f"  arr[1::2] = {arr[1::2]}")  # 从索引1开始每隔一个
print(f"  arr[::-1] = {arr[::-1]}")  # 反转
print(f"  arr[7:2:-1] = {arr[7:2:-1]}")  # 逆向切片

# 切片是视图，不是副本
print("\n注意：切片是视图")
view = arr[2:5]
print(f"  view = arr[2:5] = {view}")
view[0] = 999
print(f"  修改 view[0] = 999")
print(f"  原数组 arr = {arr}")
print(f"  原数组也被修改了！")
arr[2] = 30  # 改回来

# 创建副本
copy_arr = arr[2:5].copy()
print(f"\n使用copy()创建副本:")
print(f"  copy_arr = arr[2:5].copy() = {copy_arr}")
copy_arr[0] = 888
print(f"  修改 copy_arr[0] = 888")
print(f"  原数组 arr = {arr}")
print(f"  原数组没有变化")

# ============================================================
# 3. 二维数组索引
# ============================================================
print("\n【3. 二维数组索引】")

# 创建示例矩阵
matrix = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
])
print(f"\nmatrix:\n{matrix}")
print(f"  形状: {matrix.shape}")

# 基本索引
print("\n基本索引:")
print(f"  matrix[0, 0] = {matrix[0, 0]}")  # 第一行第一列
print(f"  matrix[1, 2] = {matrix[1, 2]}")  # 第二行第三列
print(f"  matrix[-1, -1] = {matrix[-1, -1]}")  # 最后一行最后一列

# 两种索引方式
print("\n两种索引方式:")
print(f"  matrix[1, 2] = {matrix[1, 2]}")  # 推荐
print(f"  matrix[1][2] = {matrix[1][2]}")  # 也可以，但效率低

# ============================================================
# 4. 二维数组切片
# ============================================================
print("\n【4. 二维数组切片】")

print(f"\nmatrix:\n{matrix}")

# 单行
print("\n单行:")
print(f"  matrix[0, :] 第0行 = {matrix[0, :]}")
print(f"  matrix[1, :] 第1行 = {matrix[1, :]}")

# 单列
print("\n单列:")
print(f"  matrix[:, 0] 第0列 = {matrix[:, 0]}")
print(f"  matrix[:, 2] 第2列 = {matrix[:, 2]}")

# 子矩阵
print("\n子矩阵:")
print(f"  matrix[0:2, 1:3]:\n{matrix[0:2, 1:3]}")
print(f"  matrix[:2, :3]:\n{matrix[:2, :3]}")
print(f"  matrix[1:, 2:]:\n{matrix[1:, 2:]}")

# 步长
print("\n使用步长:")
print(f"  matrix[::2, ::2] 隔行隔列:\n{matrix[::2, ::2]}")

# ============================================================
# 5. 布尔索引
# ============================================================
print("\n【5. 布尔索引】")
print("使用布尔数组筛选元素")

# 一维布尔索引
data = np.array([0.1, 0.5, 0.8, 0.3, 0.9, 0.2, 0.7])
print(f"\ndata = {data}")

# 创建布尔掩码
mask = data > 0.5
print(f"  mask (data > 0.5) = {mask}")

# 使用掩码筛选
filtered = data[mask]
print(f"  data[mask] = {filtered}")

# 一步完成
print(f"  data[data > 0.5] = {data[data > 0.5]}")
print(f"  data[data <= 0.3] = {data[data <= 0.3]}")

# 复合条件
print("\n复合条件:")
print(f"  (data > 0.3) & (data < 0.8):")
print(f"    {data[(data > 0.3) & (data < 0.8)]}")

# 修改筛选的元素
data_copy = data.copy()
data_copy[data_copy > 0.5] = 0.5  # 截断到0.5
print(f"\n将大于0.5的值截断到0.5:")
print(f"  原始: {data}")
print(f"  截断后: {data_copy}")

# 二维布尔索引
print("\n二维布尔索引:")
matrix2 = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
print(f"matrix:\n{matrix2}")
print(f"  matrix[matrix > 5]:\n  {matrix2[matrix2 > 5]}")

# ============================================================
# 6. 花式索引
# ============================================================
print("\n【6. 花式索引】")
print("使用整数数组索引")

arr = np.array([10, 20, 30, 40, 50])
print(f"\narr = {arr}")

# 使用整数列表索引
indices = [0, 2, 4]
print(f"  indices = {indices}")
print(f"  arr[indices] = {arr[indices]}")

# 不按顺序
indices2 = [4, 1, 3, 1]
print(f"  indices2 = {indices2}")
print(f"  arr[indices2] = {arr[indices2]}")

# 二维花式索引
matrix3 = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
print(f"\nmatrix:\n{matrix3}")

# 选择特定行
rows = [0, 2]
print(f"  matrix[rows, :] 选择第0和第2行:\n{matrix3[rows, :]}")

# 选择特定元素
rows = [0, 1, 2]
cols = [0, 1, 2]
print(f"  选择对角线元素: {matrix3[rows, cols]}")

# ============================================================
# 7. 实际应用：EMG数据处理
# ============================================================
print("\n【7. 实际应用：EMG数据处理】")

# 模拟EMG数据：1000个样本，4个通道
np.random.seed(42)
emg_data = np.random.randn(1000, 4) * 0.5 + 0.1

print("\nEMG数据处理示例:")
print(f"  数据形状: {emg_data.shape}")
print(f"  (1000个样本, 4个通道)")

# 提取特定通道
print("\n提取特定通道:")
channel_0 = emg_data[:, 0]
print(f"  通道0数据: {channel_0[:10]}")  # 前10个样本

# 提取时间段
print("\n提取时间段:")
segment = emg_data[100:200, :]  # 样本100-199
print(f"  提取样本100-199: 形状 {segment.shape}")

# 提取多个通道
print("\n提取多个通道:")
channels_02 = emg_data[:, [0, 2]]  # 通道0和2
print(f"  提取通道0和2: 形状 {channels_02.shape}")

# 去除异常值
print("\n去除异常值:")
threshold = 2.0
mask = np.abs(emg_data) < threshold
filtered_count = np.sum(~mask)
print(f"  阈值: {threshold}")
print(f"  异常值数量: {filtered_count}")

# 替换异常值为0
emg_filtered = emg_data.copy()
emg_filtered[~mask] = 0
print(f"  异常值已替换为0")

# 筛选激活区段
print("\n筛选激活区段:")
# 计算每个样本的RMS（跨通道）
rms = np.sqrt(np.mean(emg_data ** 2, axis=1))
activation_threshold = 0.5
active_samples = rms > activation_threshold
active_count = np.sum(active_samples)
print(f"  RMS阈值: {activation_threshold}")
print(f"  激活样本数: {active_count} / {len(rms)}")
print(f"  激活比例: {active_count / len(rms) * 100:.2f}%")

# ============================================================
# 实践练习
# ============================================================
print("\n" + "="*60)
print("实践练习")
print("="*60)

print("\n练习1: 数组切片")
arr = np.arange(20)
print(f"arr = {arr}")
print(f"  提取索引5-15: {arr[5:16]}")
print(f"  提取所有偶数索引: {arr[::2]}")
print(f"  反转数组: {arr[::-1]}")

print("\n练习2: 二维数组操作")
matrix = np.arange(1, 26).reshape(5, 5)
print(f"5x5矩阵:\n{matrix}")
print(f"  提取第3行: {matrix[2, :]}")
print(f"  提取第4列: {matrix[:, 3]}")
print(f"  提取中心3x3子矩阵:\n{matrix[1:4, 1:4]}")

print("\n练习3: 布尔索引")
data = np.array([1, -2, 3, -4, 5, -6, 7, -8])
print(f"data = {data}")
positive = data[data > 0]
negative = data[data < 0]
print(f"  正数: {positive}")
print(f"  负数: {negative}")
print(f"  绝对值大于3: {data[np.abs(data) > 3]}")

print("\n练习4: 花式索引")
arr = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
indices = [0, 3, 5, 7]
print(f"arr = {arr}")
print(f"  选择索引{indices}: {arr[indices]}")

print("\n练习5: EMG多通道处理")
# 模拟10秒的4通道EMG数据
fs = 100  # 采样率100Hz
duration = 10
n_samples = fs * duration
n_channels = 4
emg = np.random.randn(n_samples, n_channels) * 0.3

print(f"EMG数据: {emg.shape}")
# 提取前5秒
first_5s = emg[:fs*5, :]
print(f"  前5秒: {first_5s.shape}")
# 提取通道1和3
ch_13 = emg[:, [1, 3]]
print(f"  通道1和3: {ch_13.shape}")
# 计算每个通道的平均绝对值
mav = np.mean(np.abs(emg), axis=0)
print(f"  各通道MAV: {mav}")

# ============================================================
# 课后作业
# ============================================================
print("\n" + "="*60)
print("课后作业")
print("="*60)

print("\n请完成以下作业:")
print("1. 创建数组np.arange(0, 100)")
print("   a. 提取所有能被5整除的元素")
print("   b. 提取10-50之间的元素")
print("   c. 提取索引为奇数的元素")
print("2. 创建10x10的随机矩阵")
print("   a. 提取4个角的元素")
print("   b. 提取边界元素")
print("   c. 提取对角线元素")
print("3. 给定数据数组，包含一些异常值（-999）")
print("   使用布尔索引将所有-999替换为该列的平均值")
print("4. 创建模拟的多通道EMG数据（1000样本，8通道）")
print("   a. 提取前500个样本")
print("   b. 提取通道0,2,4,6")
print("   c. 计算每个通道的RMS")
print("   d. 找出RMS最大的通道")
print("5. 实现滑动窗口:")
print("   给定信号和窗口大小，提取所有窗口")

# ============================================================
# 总结
# ============================================================
print("\n" + "="*60)
print("本课总结")
print("="*60)

print("\n核心要点:")
print("1. 基本索引: arr[i], arr[i, j]")
print("2. 切片: arr[start:stop:step]")
print("3. 布尔索引: arr[arr > 5]")
print("4. 花式索引: arr[[0, 2, 4]]")
print("5. 切片是视图，使用copy()创建副本")
print("6. axis=0是行，axis=1是列")

print("\nEMG应用:")
print("- 提取特定时间段: emg[100:200, :]")
print("- 提取特定通道: emg[:, 0]")
print("- 去除异常值: emg[abs(emg) < threshold]")
print("- 筛选激活区段: emg[rms > threshold]")

print("\n下一课: 07_numpy_math.py - NumPy数学运算和统计")
print("="*60)
