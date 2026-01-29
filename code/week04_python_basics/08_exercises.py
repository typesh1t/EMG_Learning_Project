#!/usr/bin/env python3
"""
Python基础 - 综合练习
综合运用所学的Python和NumPy知识
"""

import numpy as np

print("="*60)
print("第八课：综合练习")
print("="*60)

print("\n本课包含10个练习题，从易到难")
print("每道题都有参考答案，建议先自己尝试，再查看答案")
print("="*60)

# ============================================================
# 练习1: 基本统计
# ============================================================
print("\n【练习1: 基本统计】")
print("题目: 计算列表的最大值、最小值、平均值（不使用内置函数）")

# 数据
data = [3.2, 1.5, 4.8, 2.1, 5.0, 0.9, 3.7]

print(f"数据: {data}")
print("\n--- 请在此处编写你的代码 ---")

# 参考答案
print("\n参考答案:")
# 最大值
max_val = data[0]
for val in data:
    if val > max_val:
        max_val = val

# 最小值
min_val = data[0]
for val in data:
    if val < min_val:
        min_val = val

# 平均值
total = 0
for val in data:
    total += val
avg_val = total / len(data)

print(f"  最大值: {max_val}")
print(f"  最小值: {min_val}")
print(f"  平均值: {avg_val:.2f}")

# ============================================================
# 练习2: 列表推导式
# ============================================================
print("\n" + "="*60)
print("【练习2: 列表推导式】")
print("题目: 使用列表推导式")
print("  a. 创建1-100中所有能被3整除的数")
print("  b. 将['MAV', 'RMS', 'VAR']转换为小写")
print("  c. 从[1,2,3,4,5]创建平方值列表")

print("\n--- 请在此处编写你的代码 ---")

# 参考答案
print("\n参考答案:")
divisible_by_3 = [x for x in range(1, 101) if x % 3 == 0]
print(f"  a. 能被3整除: {divisible_by_3[:10]}... (共{len(divisible_by_3)}个)")

features = ['MAV', 'RMS', 'VAR']
lowercase = [f.lower() for f in features]
print(f"  b. 转小写: {lowercase}")

numbers = [1, 2, 3, 4, 5]
squares = [x ** 2 for x in numbers]
print(f"  c. 平方: {squares}")

# ============================================================
# 练习3: 字典操作
# ============================================================
print("\n" + "="*60)
print("【练习3: 字典操作】")
print("题目: 创建字典存储EMG特征值，并计算平均值")

features_dict = {
    'MAV': 0.52,
    'RMS': 0.68,
    'VAR': 0.31,
    'WL': 12.5,
    'ZC': 85
}

print(f"\n特征字典: {features_dict}")
print("\n任务: 计算所有特征值的平均值")
print("--- 请在此处编写你的代码 ---")

# 参考答案
print("\n参考答案:")
avg_value = sum(features_dict.values()) / len(features_dict)
print(f"  平均值: {avg_value:.2f}")

# ============================================================
# 练习4: NumPy数组创建
# ============================================================
print("\n" + "="*60)
print("【练习4: NumPy数组创建】")
print("题目: 创建以下数组")
print("  a. 0到10的整数数组")
print("  b. 0到1之间的11个均匀分布的点")
print("  c. 5x5的全1矩阵")
print("  d. 3x4的随机数组（0-1均匀分布）")

print("\n--- 请在此处编写你的代码 ---")

# 参考答案
print("\n参考答案:")
a = np.arange(11)
print(f"  a. {a}")

b = np.linspace(0, 1, 11)
print(f"  b. {b}")

c = np.ones((5, 5))
print(f"  c. 5x5全1矩阵:\n{c}")

d = np.random.rand(3, 4)
print(f"  d. 3x4随机数组:\n{d}")

# ============================================================
# 练习5: 数组索引
# ============================================================
print("\n" + "="*60)
print("【练习5: 数组索引】")
print("题目: 给定数组，完成以下操作")

arr = np.arange(0, 100)
print(f"\n数组: {arr[:20]}... (0-99)")
print("\n任务:")
print("  a. 提取所有能被5整除的元素")
print("  b. 提取10-50之间的元素")
print("  c. 提取索引为偶数的元素（前10个）")

print("\n--- 请在此处编写你的代码 ---")

# 参考答案
print("\n参考答案:")
divisible_5 = arr[arr % 5 == 0]
print(f"  a. 能被5整除: {divisible_5}")

between_10_50 = arr[(arr >= 10) & (arr <= 50)]
print(f"  b. 10-50之间: {between_10_50}")

even_indices = arr[::2][:10]
print(f"  c. 偶数索引（前10个）: {even_indices}")

# ============================================================
# 练习6: 计算EMG特征
# ============================================================
print("\n" + "="*60)
print("【练习6: 计算EMG特征】")
print("题目: 生成模拟EMG信号并计算4个基本特征")

np.random.seed(42)
emg_signal = np.random.randn(1000) * 0.3

print(f"\nEMG信号: {len(emg_signal)}个样本")
print("\n任务: 计算MAV, RMS, VAR, WL")
print("--- 请在此处编写你的代码 ---")

# 参考答案
print("\n参考答案:")
mav = np.mean(np.abs(emg_signal))
rms = np.sqrt(np.mean(emg_signal ** 2))
var = np.var(emg_signal)
wl = np.sum(np.abs(np.diff(emg_signal)))

print(f"  MAV: {mav:.4f}")
print(f"  RMS: {rms:.4f}")
print(f"  VAR: {var:.4f}")
print(f"  WL: {wl:.2f}")

# ============================================================
# 练习7: 多通道处理
# ============================================================
print("\n" + "="*60)
print("【练习7: 多通道EMG处理】")
print("题目: 处理4通道EMG数据")

np.random.seed(42)
# 模拟4个通道，强度不同
multi_ch = np.random.randn(1000, 4) * np.array([0.3, 0.5, 0.4, 0.6])

print(f"\n多通道数据: {multi_ch.shape}")
print("\n任务:")
print("  a. 计算每个通道的RMS")
print("  b. 找出最强和最弱的通道")
print("  c. 归一化每个通道到[0, 1]")

print("\n--- 请在此处编写你的代码 ---")

# 参考答案
print("\n参考答案:")
rms_channels = np.sqrt(np.mean(multi_ch ** 2, axis=0))
print(f"  a. 各通道RMS: {rms_channels}")

strongest = np.argmax(rms_channels)
weakest = np.argmin(rms_channels)
print(f"  b. 最强通道: {strongest}, 最弱通道: {weakest}")

normalized = np.zeros_like(multi_ch)
for ch in range(multi_ch.shape[1]):
    ch_data = multi_ch[:, ch]
    normalized[:, ch] = (ch_data - ch_data.min()) / (ch_data.max() - ch_data.min())
print(f"  c. 归一化完成，范围检查:")
print(f"     通道0: [{normalized[:, 0].min():.2f}, {normalized[:, 0].max():.2f}]")

# ============================================================
# 练习8: 异常值处理
# ============================================================
print("\n" + "="*60)
print("【练习8: 异常值检测与处理】")
print("题目: 使用3σ准则识别并处理异常值")

np.random.seed(42)
data_with_outliers = np.concatenate([
    np.random.randn(95) * 0.5,
    np.array([10, -10, 15, -15, 20])  # 异常值
])

print(f"\n数据: {len(data_with_outliers)}个样本")
print("\n任务:")
print("  a. 使用3σ准则识别异常值")
print("  b. 统计异常值数量和比例")
print("  c. 将异常值替换为中位数")

print("\n--- 请在此处编写你的代码 ---")

# 参考答案
print("\n参考答案:")
mean = np.mean(data_with_outliers)
std = np.std(data_with_outliers)
threshold = 3 * std

outliers_mask = np.abs(data_with_outliers - mean) > threshold
n_outliers = np.sum(outliers_mask)
print(f"  a. 3σ阈值: {threshold:.2f}")
print(f"  b. 异常值数量: {n_outliers}, 比例: {n_outliers/len(data_with_outliers)*100:.2f}%")

data_cleaned = data_with_outliers.copy()
median = np.median(data_with_outliers)
data_cleaned[outliers_mask] = median
print(f"  c. 用中位数{median:.2f}替换了{n_outliers}个异常值")

# ============================================================
# 练习9: 滑动窗口
# ============================================================
print("\n" + "="*60)
print("【练习9: 滑动窗口特征提取】")
print("题目: 实现滑动窗口RMS计算")

np.random.seed(42)
long_signal = np.random.randn(1000) * 0.5

window_size = 200
step = 100

print(f"\n信号长度: {len(long_signal)}")
print(f"窗口大小: {window_size}")
print(f"步长: {step}")
print("\n任务: 计算每个窗口的RMS")

print("\n--- 请在此处编写你的代码 ---")

# 参考答案
print("\n参考答案:")
rms_windows = []
window_indices = []

for i in range(0, len(long_signal) - window_size + 1, step):
    window = long_signal[i:i+window_size]
    rms = np.sqrt(np.mean(window ** 2))
    rms_windows.append(rms)
    window_indices.append(i)

print(f"  窗口数量: {len(rms_windows)}")
print(f"  前5个窗口的RMS: {rms_windows[:5]}")
print(f"  最大RMS: {max(rms_windows):.4f} (窗口{rms_windows.index(max(rms_windows))})")

# ============================================================
# 练习10: 综合应用
# ============================================================
print("\n" + "="*60)
print("【练习10: 综合应用 - EMG手势分类数据准备】")
print("题目: 模拟3种手势的EMG数据，提取特征")

np.random.seed(42)

gestures = ['rest', 'fist', 'open']
n_trials = 5
n_samples = 500

print(f"\n生成数据:")
print(f"  手势: {gestures}")
print(f"  每种手势试验数: {n_trials}")
print(f"  每次试验样本数: {n_samples}")

print("\n任务:")
print("  a. 为每种手势生成不同强度的信号")
print("  b. 计算每次试验的MAV和RMS")
print("  c. 创建特征矩阵和标签数组")

print("\n--- 请在此处编写你的代码 ---")

# 参考答案
print("\n参考答案:")

# 手势信号强度
gesture_intensity = {'rest': 0.1, 'fist': 0.8, 'open': 0.5}

# 存储所有特征和标签
all_features = []
all_labels = []

for gesture in gestures:
    intensity = gesture_intensity[gesture]

    for trial in range(n_trials):
        # 生成信号
        signal = np.random.randn(n_samples) * intensity

        # 提取特征
        mav = np.mean(np.abs(signal))
        rms = np.sqrt(np.mean(signal ** 2))

        # 保存
        all_features.append([mav, rms])
        all_labels.append(gesture)

# 转换为NumPy数组
features_matrix = np.array(all_features)
labels_array = np.array(all_labels)

print(f"  特征矩阵: {features_matrix.shape}")
print(f"  标签数组: {labels_array.shape}")
print(f"\n各手势平均特征:")
for gesture in gestures:
    mask = labels_array == gesture
    gesture_features = features_matrix[mask]
    avg_mav = np.mean(gesture_features[:, 0])
    avg_rms = np.mean(gesture_features[:, 1])
    print(f"    {gesture}: MAV={avg_mav:.3f}, RMS={avg_rms:.3f}")

# ============================================================
# 总结
# ============================================================
print("\n" + "="*60)
print("练习总结")
print("="*60)

print("\n你已完成10个练习，涵盖:")
print("1. 基本统计计算")
print("2. 列表推导式")
print("3. 字典操作")
print("4. NumPy数组创建")
print("5. 数组索引和切片")
print("6. EMG特征计算")
print("7. 多通道数据处理")
print("8. 异常值检测")
print("9. 滑动窗口")
print("10. 综合数据准备")

print("\n这些技能是EMG信号处理的基础")
print("接下来可以学习更高级的内容：")
print("  - 信号滤波（Week 6）")
print("  - 特征提取（Week 7）")
print("  - 模式识别（Week 8）")

print("\n下一课: 09_emg_data_exercise.py - EMG数据处理练习")
print("="*60)
