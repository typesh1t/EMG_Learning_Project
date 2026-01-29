#!/usr/bin/env python3
"""
Python基础 - NumPy数学运算和统计
学习NumPy的数学函数、统计函数和线性代数操作
"""

import numpy as np

print("="*60)
print("第七课：NumPy数学运算和统计")
print("="*60)

# ============================================================
# 1. 基本数学运算
# ============================================================
print("\n【1. 基本数学运算】")

a = np.array([1, 2, 3, 4])
b = np.array([10, 20, 30, 40])

print(f"\na = {a}")
print(f"b = {b}")

# 四则运算
print("\n四则运算:")
print(f"  a + b = {a + b}")
print(f"  a - b = {a - b}")
print(f"  a * b = {a * b}")  # 逐元素相乘
print(f"  a / b = {a / b}")
print(f"  a ** 2 = {a ** 2}")
print(f"  a ** 0.5 = {a ** 0.5}")  # 平方根

# 数学函数
print("\n常用数学函数:")
x = np.array([0, 1, 4, 9, 16])
print(f"  x = {x}")
print(f"  sqrt(x) = {np.sqrt(x)}")
print(f"  square(x) = {np.square(x)}")
print(f"  abs([-1,-2,3]) = {np.abs(np.array([-1, -2, 3]))}")
print(f"  sign([-1,0,1]) = {np.sign(np.array([-1, 0, 1]))}")

# 指数和对数
print("\n指数和对数:")
x = np.array([0, 1, 2])
print(f"  x = {x}")
print(f"  exp(x) = {np.exp(x)}")
print(f"  exp2(x) = {np.exp2(x)}")  # 2^x
print(f"  log(x+1) = {np.log(x + 1)}")  # 自然对数
print(f"  log10(x+1) = {np.log10(x + 1)}")  # 常用对数

# 三角函数
print("\n三角函数:")
angles = np.array([0, np.pi/4, np.pi/2, np.pi])
print(f"  angles = {angles}")
print(f"  sin(angles) = {np.sin(angles)}")
print(f"  cos(angles) = {np.cos(angles)}")
print(f"  tan(angles[:2]) = {np.tan(angles[:2])}")

# ============================================================
# 2. 聚合统计函数
# ============================================================
print("\n【2. 聚合统计函数】")

# 基本统计
data = np.array([0.5, 0.8, 1.2, 0.3, 0.9, 1.5, 0.7])
print(f"\ndata = {data}")
print(f"  sum: {np.sum(data):.2f}")
print(f"  mean: {np.mean(data):.2f}")
print(f"  median: {np.median(data):.2f}")
print(f"  std: {np.std(data):.2f}")
print(f"  var: {np.var(data):.2f}")
print(f"  min: {np.min(data):.2f}")
print(f"  max: {np.max(data):.2f}")
print(f"  range: {np.ptp(data):.2f}")  # peak to peak

# 百分位数
print("\n百分位数:")
print(f"  25%分位: {np.percentile(data, 25):.2f}")
print(f"  50%分位: {np.percentile(data, 50):.2f}")
print(f"  75%分位: {np.percentile(data, 75):.2f}")

# 多维数组统计
print("\n多维数组统计:")
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
print(f"matrix:\n{matrix}")
print(f"  全部求和: {np.sum(matrix)}")
print(f"  按行求和 (axis=1): {np.sum(matrix, axis=1)}")
print(f"  按列求和 (axis=0): {np.sum(matrix, axis=0)}")
print(f"  按行平均 (axis=1): {np.mean(matrix, axis=1)}")
print(f"  按列平均 (axis=0): {np.mean(matrix, axis=0)}")

# ============================================================
# 3. 累积统计
# ============================================================
print("\n【3. 累积统计】")

arr = np.array([1, 2, 3, 4, 5])
print(f"\narr = {arr}")
print(f"  cumsum 累积和: {np.cumsum(arr)}")
print(f"  cumprod 累积乘积: {np.cumprod(arr)}")

# 差分
print("\n差分:")
signal = np.array([10, 12, 15, 13, 18])
print(f"  signal = {signal}")
print(f"  diff 一阶差分: {np.diff(signal)}")
print(f"  diff n=2 二阶差分: {np.diff(signal, n=2)}")

# ============================================================
# 4. 比较和逻辑运算
# ============================================================
print("\n【4. 比较和逻辑运算】")

a = np.array([1, 2, 3, 4, 5])
b = np.array([5, 4, 3, 2, 1])

print(f"\na = {a}")
print(f"b = {b}")

# 比较
print("\n比较运算:")
print(f"  a > 3: {a > 3}")
print(f"  a == b: {a == b}")
print(f"  a < b: {a < b}")

# 逻辑运算
print("\n逻辑运算:")
print(f"  (a > 2) & (a < 5): {(a > 2) & (a < 5)}")
print(f"  (a < 2) | (a > 4): {(a < 2) | (a > 4)}")
print(f"  ~(a > 3): {~(a > 3)}")

# any和all
data = np.array([0.1, 0.5, 0.8, 0.3])
print(f"\ndata = {data}")
print(f"  any(data > 0.5): {np.any(data > 0.5)}")  # 至少一个True
print(f"  all(data > 0): {np.all(data > 0)}")     # 全部True

# ============================================================
# 5. 排序和搜索
# ============================================================
print("\n【5. 排序和搜索】")

# 排序
arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])
print(f"\narr = {arr}")
print(f"  sort: {np.sort(arr)}")
print(f"  argsort 排序索引: {np.argsort(arr)}")

# 原地排序
arr_copy = arr.copy()
arr_copy.sort()
print(f"  原地排序后: {arr_copy}")

# 查找
print("\n查找:")
print(f"  argmin 最小值索引: {np.argmin(arr)}")
print(f"  argmax 最大值索引: {np.argmax(arr)}")
print(f"  where(arr > 5): {np.where(arr > 5)}")
print(f"  nonzero: {np.nonzero(arr > 5)}")

# ============================================================
# 6. EMG信号处理常用计算
# ============================================================
print("\n【6. EMG信号处理常用计算】")

# 生成模拟EMG信号
np.random.seed(42)
fs = 1000  # 采样率
duration = 1  # 1秒
n_samples = fs * duration
time = np.linspace(0, duration, n_samples)

# 模拟信号
signal = 0.3 * np.random.randn(n_samples)
signal += 0.05 * np.sin(2 * np.pi * 50 * time)  # 50Hz干扰

print(f"\n模拟EMG信号: {n_samples}个样本")

# MAV - 平均绝对值
mav = np.mean(np.abs(signal))
print(f"\n1. MAV (平均绝对值): {mav:.4f}")

# RMS - 均方根
rms = np.sqrt(np.mean(signal ** 2))
print(f"2. RMS (均方根): {rms:.4f}")

# VAR - 方差
var = np.var(signal)
print(f"3. VAR (方差): {var:.4f}")

# WL - 波形长度
wl = np.sum(np.abs(np.diff(signal)))
print(f"4. WL (波形长度): {wl:.2f}")

# ZC - 过零率
threshold = 0.01
zc = 0
for i in range(len(signal) - 1):
    if (signal[i] > threshold and signal[i+1] < -threshold) or \
       (signal[i] < -threshold and signal[i+1] > threshold):
        zc += 1
print(f"5. ZC (过零次数): {zc}")

# 归一化
normalized = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
print(f"\n归一化到[0,1]:")
print(f"  原始范围: [{np.min(signal):.2f}, {np.max(signal):.2f}]")
print(f"  归一化后: [{np.min(normalized):.2f}, {np.max(normalized):.2f}]")

# 标准化（零均值，单位方差）
standardized = (signal - np.mean(signal)) / np.std(signal)
print(f"\n标准化（Z-score）:")
print(f"  均值: {np.mean(standardized):.2e}")
print(f"  标准差: {np.std(standardized):.2f}")

# ============================================================
# 7. 多通道处理
# ============================================================
print("\n【7. 多通道EMG处理】")

# 模拟4通道EMG数据
n_channels = 4
multi_channel = np.random.randn(1000, n_channels) * 0.5

print(f"\n多通道数据: {multi_channel.shape}")

# 每个通道的统计
print("\n各通道统计:")
for ch in range(n_channels):
    print(f"  通道{ch}: MAV={np.mean(np.abs(multi_channel[:, ch])):.4f}, "
          f"RMS={np.sqrt(np.mean(multi_channel[:, ch]**2)):.4f}")

# 使用axis参数一次计算所有通道
mav_all = np.mean(np.abs(multi_channel), axis=0)
rms_all = np.sqrt(np.mean(multi_channel ** 2, axis=0))
print(f"\n使用axis参数:")
print(f"  所有通道MAV: {mav_all}")
print(f"  所有通道RMS: {rms_all}")

# 找出最大激活的通道
max_channel = np.argmax(rms_all)
print(f"  最大RMS的通道: {max_channel}")

# ============================================================
# 8. 滑动窗口计算
# ============================================================
print("\n【8. 滑动窗口计算】")

# 简单滑动窗口MAV
signal = np.random.randn(1000) * 0.5
window_size = 100
step = 50

print(f"\n信号长度: {len(signal)}")
print(f"窗口大小: {window_size}")
print(f"步长: {step}")

mav_windows = []
for i in range(0, len(signal) - window_size, step):
    window = signal[i:i+window_size]
    mav = np.mean(np.abs(window))
    mav_windows.append(mav)

print(f"窗口数: {len(mav_windows)}")
print(f"每个窗口的MAV（前5个）: {mav_windows[:5]}")

# ============================================================
# 实践练习
# ============================================================
print("\n" + "="*60)
print("实践练习")
print("="*60)

print("\n练习1: 信号统计")
signal = np.random.randn(1000) * 0.5 + 0.2
print(f"信号统计:")
print(f"  长度: {len(signal)}")
print(f"  均值: {np.mean(signal):.4f}")
print(f"  标准差: {np.std(signal):.4f}")
print(f"  最小值: {np.min(signal):.4f}")
print(f"  最大值: {np.max(signal):.4f}")
print(f"  中位数: {np.median(signal):.4f}")

print("\n练习2: 计算EMG特征")
emg = np.random.randn(500) * 0.3
mav = np.mean(np.abs(emg))
rms = np.sqrt(np.mean(emg ** 2))
var = np.var(emg)
print(f"EMG特征:")
print(f"  MAV: {mav:.4f}")
print(f"  RMS: {rms:.4f}")
print(f"  VAR: {var:.4f}")

print("\n练习3: 异常值检测")
data = np.array([1, 2, 3, 100, 4, 5, 6, -50, 7])
mean = np.mean(data)
std = np.std(data)
threshold = 2 * std
outliers = np.abs(data - mean) > threshold
print(f"数据: {data}")
print(f"异常值掩码: {outliers}")
print(f"异常值: {data[outliers]}")

print("\n练习4: 多通道比较")
ch_data = np.random.randn(100, 3) * np.array([0.5, 0.8, 0.3])
rms_values = np.sqrt(np.mean(ch_data ** 2, axis=0))
print(f"各通道RMS: {rms_values}")
print(f"最强通道: {np.argmax(rms_values)}")
print(f"最弱通道: {np.argmin(rms_values)}")

print("\n练习5: 信号能量")
signal = np.sin(2 * np.pi * 10 * np.linspace(0, 1, 1000))
energy = np.sum(signal ** 2)
power = energy / len(signal)
print(f"信号能量: {energy:.2f}")
print(f"平均功率: {power:.4f}")

# ============================================================
# 课后作业
# ============================================================
print("\n" + "="*60)
print("课后作业")
print("="*60)

print("\n请完成以下作业:")
print("1. 生成1000个随机数（正态分布），计算:")
print("   a. 均值、标准差、方差")
print("   b. 25%, 50%, 75%分位数")
print("   c. 超过1个标准差的样本数量")
print("2. 生成模拟EMG信号，计算10种时域特征")
print("   MAV, RMS, VAR, WL, ZC, SSC, IEMG, WAMP, LOG, DASDV")
print("3. 创建4通道EMG数据（1000样本）")
print("   a. 计算每个通道的RMS")
print("   b. 找出最强和最弱的通道")
print("   c. 归一化到[0, 1]")
print("4. 实现滑动窗口RMS计算")
print("   窗口大小200，步长100")
print("5. 去除异常值:")
print("   创建包含异常值的数据")
print("   使用3σ准则识别并去除")

# ============================================================
# 总结
# ============================================================
print("\n" + "="*60)
print("本课总结")
print("="*60)

print("\n核心要点:")
print("1. 数学运算: +, -, *, /, **, sqrt, exp, log, sin, cos")
print("2. 统计函数: mean, std, var, min, max, median, percentile")
print("3. 聚合函数: sum, cumsum, diff")
print("4. 比较运算: >, <, ==, &, |, ~")
print("5. 排序搜索: sort, argsort, argmin, argmax, where")
print("6. axis参数: 0表示列(竖直), 1表示行(水平)")

print("\nEMG应用:")
print("- 计算时域特征: MAV, RMS, VAR, WL, ZC")
print("- 信号归一化和标准化")
print("- 多通道统计和比较")
print("- 滑动窗口特征提取")
print("- 异常值检测和去除")

print("\n下一课: 08_exercises.py - 综合练习")
print("="*60)
