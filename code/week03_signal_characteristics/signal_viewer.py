#!/usr/bin/env python3
"""
EMG信号时域和频域查看器
帮助理解EMG信号的基本特征
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import sys
from pathlib import Path

# 配置字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def load_signal(filepath):
    """加载EMG信号"""
    data = pd.read_csv(filepath)
    print(f"已加载数据: {filepath}")
    print(f"数据形状: {data.shape}")
    print(f"列名: {data.columns.tolist()}")
    return data

def analyze_time_domain(signal, fs=1000):
    """
    分析时域特征

    参数:
        signal: 信号数组
        fs: 采样率

    返回:
        dict: 时域特征字典
    """
    features = {}

    # 基本统计量
    features['mean'] = np.mean(signal)
    features['std'] = np.std(signal)
    features['max'] = np.max(signal)
    features['min'] = np.min(signal)
    features['peak_to_peak'] = features['max'] - features['min']

    # RMS (均方根)
    features['rms'] = np.sqrt(np.mean(signal ** 2))

    # MAV (平均绝对值)
    features['mav'] = np.mean(np.abs(signal))

    # 过零率 (Zero Crossing Rate)
    threshold = 0.01
    zc = 0
    for i in range(len(signal) - 1):
        if ((signal[i] > threshold and signal[i+1] < -threshold) or
            (signal[i] < -threshold and signal[i+1] > threshold)):
            zc += 1
    features['zero_crossing_rate'] = zc / (len(signal) / fs)

    return features

def analyze_frequency_domain(signal, fs=1000):
    """
    分析频域特征

    参数:
        signal: 信号数组
        fs: 采样率

    返回:
        dict: 频域特征和频谱数据
    """
    N = len(signal)

    # FFT
    yf = fft(signal)
    xf = fftfreq(N, 1/fs)[:N//2]
    power = 2.0/N * np.abs(yf[:N//2])
    power_squared = power ** 2

    features = {}
    features['frequencies'] = xf
    features['power'] = power

    # 总功率
    features['total_power'] = np.sum(power_squared)

    # 峰值频率
    peak_idx = np.argmax(power)
    features['peak_frequency'] = xf[peak_idx]

    # 平均频率 (Mean Frequency)
    if features['total_power'] > 0:
        features['mean_frequency'] = np.sum(xf * power_squared) / np.sum(power_squared)
    else:
        features['mean_frequency'] = 0

    # 中值频率 (Median Frequency)
    cumsum = np.cumsum(power_squared)
    if cumsum[-1] > 0:
        mdf_idx = np.where(cumsum >= cumsum[-1] / 2)[0]
        features['median_frequency'] = xf[mdf_idx[0]] if len(mdf_idx) > 0 else 0
    else:
        features['median_frequency'] = 0

    return features

def plot_signal_characteristics(data, fs=1000, channel='channel_0'):
    """
    绘制信号的时域和频域特征

    参数:
        data: DataFrame包含EMG数据
        fs: 采样率
        channel: 要分析的通道名
    """
    signal = data[channel].values
    time = data['time'].values if 'time' in data.columns else np.arange(len(signal)) / fs

    # 分析特征
    time_features = analyze_time_domain(signal, fs)
    freq_features = analyze_frequency_domain(signal, fs)

    # 创建图表
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f'EMG信号特征分析 - {channel}', fontsize=16, fontweight='bold')

    # 1. 完整信号时域图
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(time, signal, linewidth=0.5, color='blue')
    ax1.set_xlabel('时间 (秒)')
    ax1.set_ylabel('幅度 (mV)')
    ax1.set_title('完整信号（时域）')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color='black', linewidth=0.5)

    # 2. 局部放大（前1秒）
    ax2 = plt.subplot(3, 2, 2)
    zoom_samples = min(int(fs * 1), len(signal))
    ax2.plot(time[:zoom_samples], signal[:zoom_samples], linewidth=0.8, color='blue')
    ax2.set_xlabel('时间 (秒)')
    ax2.set_ylabel('幅度 (mV)')
    ax2.set_title('前1秒放大')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0, color='black', linewidth=0.5)

    # 3. 频谱
    ax3 = plt.subplot(3, 2, 3)
    ax3.plot(freq_features['frequencies'], freq_features['power'],
            linewidth=1, color='green')
    ax3.set_xlabel('频率 (Hz)')
    ax3.set_ylabel('功率')
    ax3.set_title('功率谱')
    ax3.set_xlim(0, 300)
    ax3.grid(True, alpha=0.3)

    # 标注峰值频率
    ax3.axvline(freq_features['peak_frequency'],
               color='red', linestyle='--', alpha=0.7)
    ax3.text(freq_features['peak_frequency'] + 10,
            np.max(freq_features['power']) * 0.9,
            f"峰值: {freq_features['peak_frequency']:.1f}Hz",
            fontsize=9)

    # 4. 对数频谱
    ax4 = plt.subplot(3, 2, 4)
    ax4.semilogy(freq_features['frequencies'], freq_features['power'],
                linewidth=1, color='green')
    ax4.set_xlabel('频率 (Hz)')
    ax4.set_ylabel('功率 (对数)')
    ax4.set_title('功率谱（对数坐标）')
    ax4.set_xlim(0, 500)
    ax4.grid(True, alpha=0.3)

    # 5. 信号直方图
    ax5 = plt.subplot(3, 2, 5)
    ax5.hist(signal, bins=50, color='purple', alpha=0.7, edgecolor='black')
    ax5.set_xlabel('幅度 (mV)')
    ax5.set_ylabel('频数')
    ax5.set_title('幅度分布直方图')
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.axvline(0, color='red', linestyle='--', linewidth=1)

    # 6. 统计信息表格
    ax6 = plt.subplot(3, 2, 6)
    ax6.axis('off')
    ax6.set_title('统计特征', fontsize=12, fontweight='bold')

    # 时域特征
    time_text = "时域特征:\n"
    time_text += f"  均值: {time_features['mean']:.6f} mV\n"
    time_text += f"  标准差: {time_features['std']:.6f} mV\n"
    time_text += f"  RMS: {time_features['rms']:.6f} mV\n"
    time_text += f"  MAV: {time_features['mav']:.6f} mV\n"
    time_text += f"  最大值: {time_features['max']:.6f} mV\n"
    time_text += f"  最小值: {time_features['min']:.6f} mV\n"
    time_text += f"  峰峰值: {time_features['peak_to_peak']:.6f} mV\n"
    time_text += f"  过零率: {time_features['zero_crossing_rate']:.2f} Hz\n"

    # 频域特征
    freq_text = "\n频域特征:\n"
    freq_text += f"  总功率: {freq_features['total_power']:.6f}\n"
    freq_text += f"  峰值频率: {freq_features['peak_frequency']:.2f} Hz\n"
    freq_text += f"  平均频率: {freq_features['mean_frequency']:.2f} Hz\n"
    freq_text += f"  中值频率: {freq_features['median_frequency']:.2f} Hz\n"

    ax6.text(0.1, 0.95, time_text + freq_text,
            transform=ax6.transAxes,
            fontsize=10,
            verticalalignment='top',
            fontfamily='monospace')

    plt.tight_layout()

    # 保存
    output_file = f'signal_analysis_{channel}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n图表已保存到: {output_file}")

    plt.show()

    return time_features, freq_features

def compare_rest_vs_contraction(data, fs=1000, channel='channel_0'):
    """
    对比静息期和收缩期的信号特征
    """
    signal = data[channel].values
    time = data['time'].values if 'time' in data.columns else np.arange(len(signal)) / fs

    # 假设前20%是静息，中间40%是收缩
    rest_end = int(len(signal) * 0.2)
    contract_start = int(len(signal) * 0.3)
    contract_end = int(len(signal) * 0.7)

    rest_signal = signal[:rest_end]
    contract_signal = signal[contract_start:contract_end]

    # 分析两个阶段
    rest_features = analyze_time_domain(rest_signal, fs)
    contract_features = analyze_time_domain(contract_signal, fs)

    print("\n" + "="*60)
    print("静息期 vs 收缩期对比")
    print("="*60)

    print("\n静息期特征:")
    print(f"  RMS: {rest_features['rms']:.6f} mV")
    print(f"  MAV: {rest_features['mav']:.6f} mV")
    print(f"  峰峰值: {rest_features['peak_to_peak']:.6f} mV")

    print("\n收缩期特征:")
    print(f"  RMS: {contract_features['rms']:.6f} mV")
    print(f"  MAV: {contract_features['mav']:.6f} mV")
    print(f"  峰峰值: {contract_features['peak_to_peak']:.6f} mV")

    print("\n比值:")
    if rest_features['rms'] > 0:
        print(f"  RMS比值: {contract_features['rms'] / rest_features['rms']:.2f}x")
    if rest_features['mav'] > 0:
        print(f"  MAV比值: {contract_features['mav'] / rest_features['mav']:.2f}x")

    print("="*60 + "\n")

if __name__ == '__main__':
    print("\n" + "="*60)
    print("EMG信号特征查看器")
    print("="*60)

    # 检查命令行参数
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        # 使用默认样本数据
        project_root = Path(__file__).parent.parent.parent
        default_path = project_root / 'data' / 'sample' / 'subject_01' / 'fist' / 'trial_001.csv'

        if default_path.exists():
            filepath = str(default_path)
            print(f"\n使用默认样本数据: {filepath}")
        else:
            print("\n错误: 未找到样本数据")
            print("请先运行: python tools/generate_sample_data.py")
            print("或指定文件: python signal_viewer.py <filepath>")
            sys.exit(1)

    # 加载和分析
    data = load_signal(filepath)

    # 获取第一个通道
    channels = [col for col in data.columns if col.startswith('channel_')]
    if not channels:
        print("错误: 未找到EMG通道数据")
        sys.exit(1)

    channel = channels[0]
    print(f"\n分析通道: {channel}")

    # 绘制特征
    time_feat, freq_feat = plot_signal_characteristics(data, fs=1000, channel=channel)

    # 对比静息和收缩
    compare_rest_vs_contraction(data, fs=1000, channel=channel)

    print("\n分析完成!")
