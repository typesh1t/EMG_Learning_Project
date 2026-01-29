#!/usr/bin/env python3
"""
Python基础 - EMG数据处理综合练习
使用真实的EMG数据进行完整的处理流程
"""

import numpy as np

print("="*60)
print("第九课：EMG数据处理综合练习")
print("="*60)

print("\n本课模拟完整的EMG数据处理流程：")
print("1. 数据生成/加载")
print("2. 数据探索")
print("3. 预处理")
print("4. 特征提取")
print("5. 数据可视化（文本形式）")
print("="*60)

# ============================================================
# 步骤1: 生成模拟EMG数据
# ============================================================
print("\n" + "="*60)
print("步骤1: 生成模拟EMG数据")
print("="*60)

np.random.seed(42)

# 参数设置
fs = 1000  # 采样率 Hz
duration = 5  # 时长 秒
n_samples = fs * duration
n_channels = 4  # 4个通道

print(f"\n参数:")
print(f"  采样率: {fs} Hz")
print(f"  时长: {duration} 秒")
print(f"  样本数: {n_samples}")
print(f"  通道数: {n_channels}")

# 生成时间轴
time = np.linspace(0, duration, n_samples)

# 生成多通道EMG信号
# 模拟场景：1-2秒静息，2-4秒握拳（激活），4-5秒静息
emg_data = np.zeros((n_samples, n_channels))

for ch in range(n_channels):
    # 基础噪声
    noise = np.random.randn(n_samples) * 0.05

    # 工频干扰 (50Hz)
    powerline = 0.02 * np.sin(2 * np.pi * 50 * time)

    # 肌肉激活（2-4秒）
    activation = np.zeros(n_samples)
    start_idx = int(2 * fs)
    end_idx = int(4 * fs)

    # 不同通道不同强度
    intensity = 0.3 + 0.2 * ch

    # 在激活区间添加EMG信号
    for freq in range(50, 200, 15):
        amplitude = intensity * np.random.uniform(0.1, 0.3)
        activation[start_idx:end_idx] += amplitude * np.sin(
            2 * np.pi * freq * time[start_idx:end_idx] + np.random.uniform(0, 2*np.pi)
        )

    # 添加随机突发
    activation[start_idx:end_idx] += np.random.randn(end_idx - start_idx) * intensity * 0.3

    # 合成信号
    emg_data[:, ch] = noise + powerline + activation

print(f"\n✓ 生成EMG数据: {emg_data.shape}")

# ============================================================
# 步骤2: 数据探索
# ============================================================
print("\n" + "="*60)
print("步骤2: 数据探索")
print("="*60)

print("\n基本信息:")
print(f"  数据形状: {emg_data.shape}")
print(f"  数据类型: {emg_data.dtype}")
print(f"  内存大小: {emg_data.nbytes / 1024:.2f} KB")

print("\n全局统计:")
print(f"  均值: {np.mean(emg_data):.4f}")
print(f"  标准差: {np.std(emg_data):.4f}")
print(f"  最小值: {np.min(emg_data):.4f}")
print(f"  最大值: {np.max(emg_data):.4f}")

print("\n各通道统计:")
for ch in range(n_channels):
    ch_data = emg_data[:, ch]
    print(f"  通道{ch}: 均值={np.mean(ch_data):.4f}, "
          f"标准差={np.std(ch_data):.4f}, "
          f"范围=[{np.min(ch_data):.2f}, {np.max(ch_data):.2f}]")

# 检查异常值
threshold = 3 * np.std(emg_data)
outliers = np.abs(emg_data) > threshold
n_outliers = np.sum(outliers)
print(f"\n异常值检测 (3σ准则):")
print(f"  阈值: ±{threshold:.2f}")
print(f"  异常值数量: {n_outliers} / {emg_data.size} ({n_outliers/emg_data.size*100:.2f}%)")

# ============================================================
# 步骤3: 数据预处理
# ============================================================
print("\n" + "="*60)
print("步骤3: 数据预处理")
print("="*60)

# 3.1 去除直流偏置
print("\n3.1 去除直流偏置")
emg_processed = emg_data - np.mean(emg_data, axis=0)
print(f"  去偏置后均值: {np.mean(emg_processed, axis=0)}")

# 3.2 归一化
print("\n3.2 归一化到[-1, 1]")
for ch in range(n_channels):
    ch_data = emg_processed[:, ch]
    max_abs = np.max(np.abs(ch_data))
    if max_abs > 0:
        emg_processed[:, ch] = ch_data / max_abs

print(f"  归一化后范围: [{np.min(emg_processed):.2f}, {np.max(emg_processed):.2f}]")

# 3.3 分段：静息 vs 激活
print("\n3.3 信号分段")
rest_segment1 = emg_processed[:int(2*fs), :]    # 0-2秒
active_segment = emg_processed[int(2*fs):int(4*fs), :]  # 2-4秒
rest_segment2 = emg_processed[int(4*fs):, :]    # 4-5秒

print(f"  静息段1: {rest_segment1.shape}")
print(f"  激活段: {active_segment.shape}")
print(f"  静息段2: {rest_segment2.shape}")

# ============================================================
# 步骤4: 特征提取
# ============================================================
print("\n" + "="*60)
print("步骤4: 特征提取")
print("="*60)

def extract_features(segment):
    """提取EMG特征"""
    features = {}

    # 逐通道计算
    for ch in range(segment.shape[1]):
        ch_data = segment[:, ch]

        # MAV - 平均绝对值
        mav = np.mean(np.abs(ch_data))

        # RMS - 均方根
        rms = np.sqrt(np.mean(ch_data ** 2))

        # VAR - 方差
        var = np.var(ch_data)

        # WL - 波形长度
        wl = np.sum(np.abs(np.diff(ch_data)))

        # ZC - 过零次数
        threshold = 0.01
        zc = 0
        for i in range(len(ch_data) - 1):
            if (ch_data[i] > threshold and ch_data[i+1] < -threshold) or \
               (ch_data[i] < -threshold and ch_data[i+1] > threshold):
                zc += 1

        features[f'CH{ch}'] = {
            'MAV': mav,
            'RMS': rms,
            'VAR': var,
            'WL': wl,
            'ZC': zc
        }

    return features

# 提取各段特征
print("\n4.1 静息段1特征:")
rest1_features = extract_features(rest_segment1)
for ch, feat in rest1_features.items():
    print(f"  {ch}: MAV={feat['MAV']:.4f}, RMS={feat['RMS']:.4f}, "
          f"VAR={feat['VAR']:.4f}, WL={feat['WL']:.2f}, ZC={feat['ZC']}")

print("\n4.2 激活段特征:")
active_features = extract_features(active_segment)
for ch, feat in active_features.items():
    print(f"  {ch}: MAV={feat['MAV']:.4f}, RMS={feat['RMS']:.4f}, "
          f"VAR={feat['VAR']:.4f}, WL={feat['WL']:.2f}, ZC={feat['ZC']}")

print("\n4.3 静息段2特征:")
rest2_features = extract_features(rest_segment2)
for ch, feat in rest2_features.items():
    print(f"  {ch}: MAV={feat['MAV']:.4f}, RMS={feat['RMS']:.4f}, "
          f"VAR={feat['VAR']:.4f}, WL={feat['WL']:.2f}, ZC={feat['ZC']}")

# 特征对比
print("\n4.4 激活段 vs 静息段特征增长:")
for ch in range(n_channels):
    ch_name = f'CH{ch}'
    rest_rms = rest1_features[ch_name]['RMS']
    active_rms = active_features[ch_name]['RMS']
    increase = (active_rms - rest_rms) / rest_rms * 100 if rest_rms > 0 else 0
    print(f"  {ch_name}: RMS从{rest_rms:.4f}增加到{active_rms:.4f} (+{increase:.1f}%)")

# ============================================================
# 步骤5: 滑动窗口特征
# ============================================================
print("\n" + "="*60)
print("步骤5: 滑动窗口特征提取")
print("="*60)

window_size = 200  # 200ms
step = 100  # 100ms
print(f"\n窗口大小: {window_size}ms ({window_size}个样本)")
print(f"步长: {step}ms ({step}个样本)")

# 计算第一个通道的滑动窗口RMS
channel_0 = emg_processed[:, 0]
rms_windows = []
window_times = []

for i in range(0, len(channel_0) - window_size, step):
    window = channel_0[i:i+window_size]
    rms = np.sqrt(np.mean(window ** 2))
    rms_windows.append(rms)
    window_times.append(time[i + window_size//2])  # 窗口中心时间

rms_windows = np.array(rms_windows)
window_times = np.array(window_times)

print(f"\n窗口数量: {len(rms_windows)}")

# 简单的文本"可视化"
print("\n通道0的RMS时间序列 (文本图):")
print("RMS值:")
max_rms = np.max(rms_windows)
for i, (t, rms) in enumerate(zip(window_times, rms_windows)):
    # 归一化到50个字符宽度
    bar_length = int((rms / max_rms) * 50) if max_rms > 0 else 0
    bar = '█' * bar_length
    print(f"  {t:.1f}s |{bar} {rms:.4f}")

# 识别激活区域
activation_threshold = np.mean(rms_windows) + np.std(rms_windows)
active_windows = rms_windows > activation_threshold
print(f"\n激活检测:")
print(f"  阈值: {activation_threshold:.4f}")
print(f"  激活窗口数: {np.sum(active_windows)} / {len(rms_windows)}")

# 找出激活时段
active_indices = np.where(active_windows)[0]
if len(active_indices) > 0:
    start_time = window_times[active_indices[0]]
    end_time = window_times[active_indices[-1]]
    print(f"  激活时段: {start_time:.2f}s - {end_time:.2f}s")
    print(f"  （实际激活: 2.0s - 4.0s）")

# ============================================================
# 步骤6: 数据质量评估
# ============================================================
print("\n" + "="*60)
print("步骤6: 数据质量评估")
print("="*60)

# 信噪比估计（简化版）
# 使用静息段作为噪声估计
noise_power = np.mean(rest_segment1 ** 2)
signal_power = np.mean(active_segment ** 2)
snr = signal_power / noise_power if noise_power > 0 else 0
snr_db = 10 * np.log10(snr)

print(f"\n信噪比估计:")
print(f"  噪声功率: {noise_power:.6f}")
print(f"  信号功率: {signal_power:.6f}")
print(f"  SNR: {snr:.2f} ({snr_db:.2f} dB)")

# 数据完整性
print(f"\n数据完整性:")
print(f"  缺失值: {np.sum(np.isnan(emg_data))}")
print(f"  无穷值: {np.sum(np.isinf(emg_data))}")
print(f"  数据完整: {'是' if not np.any(np.isnan(emg_data)) and not np.any(np.isinf(emg_data)) else '否'}")

# ============================================================
# 步骤7: 综合报告
# ============================================================
print("\n" + "="*60)
print("步骤7: 综合报告")
print("="*60)

print("\n实验总结:")
print(f"  采集时长: {duration}秒")
print(f"  通道数: {n_channels}")
print(f"  采样率: {fs} Hz")
print(f"  总样本数: {n_samples}")

print("\n信号质量:")
print(f"  SNR: {snr_db:.2f} dB")
print(f"  异常值比例: {n_outliers/emg_data.size*100:.2f}%")

print("\n激活检测:")
print(f"  检测到激活: 是")
print(f"  激活时段: ~2.0-4.0秒")
print(f"  RMS增长: {increase:.1f}%")

print("\n特征统计:")
print(f"  静息段平均RMS: {np.mean([feat['RMS'] for feat in rest1_features.values()]):.4f}")
print(f"  激活段平均RMS: {np.mean([feat['RMS'] for feat in active_features.values()]):.4f}")

# ============================================================
# 课后练习
# ============================================================
print("\n" + "="*60)
print("课后练习")
print("="*60)

print("\n基于本课内容，请完成以下练习：")
print("\n1. 修改参数生成不同场景的数据:")
print("   - 改变激活时间段")
print("   - 增加或减少通道数")
print("   - 调整信号强度")

print("\n2. 实现更多特征:")
print("   - SSC (斜率符号变化)")
print("   - IEMG (积分EMG)")
print("   - WAMP (威尔逊幅度)")

print("\n3. 尝试不同的预处理方法:")
print("   - 标准化（Z-score）")
print("   - 截断异常值")
print("   - 不同的归一化方法")

print("\n4. 比较不同窗口大小:")
print("   - 100ms, 200ms, 500ms")
print("   - 观察对特征的影响")

print("\n5. 生成多试验数据:")
print("   - 3种手势（rest, fist, open）")
print("   - 每种5次试验")
print("   - 提取特征并保存")

# ============================================================
# 总结
# ============================================================
print("\n" + "="*60)
print("总结")
print("="*60)

print("\n恭喜！你已完成Python基础和NumPy的所有学习内容")

print("\n本课涵盖了完整的EMG数据处理流程：")
print("  1. 数据生成/加载")
print("  2. 数据探索和统计")
print("  3. 预处理（去偏置、归一化）")
print("  4. 特征提取（MAV, RMS, VAR, WL, ZC）")
print("  5. 滑动窗口分析")
print("  6. 信号质量评估")
print("  7. 综合报告")

print("\n你现在已经具备：")
print("  ✓ Python编程基础")
print("  ✓ NumPy数组操作")
print("  ✓ EMG信号处理基本技能")
print("  ✓ 特征提取能力")

print("\n下一步学习方向：")
print("  → Week 5: 使用matplotlib可视化")
print("  → Week 6: 高级滤波技术")
print("  → Week 7: 更多特征提取方法")
print("  → Week 8: 机器学习分类")

print("\n继续加油！")
print("="*60)
