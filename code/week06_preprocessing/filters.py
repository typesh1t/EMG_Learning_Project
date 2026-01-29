#!/usr/bin/env python3
"""
EMG信号滤波器模块
包含各种滤波器的实现
"""

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
import matplotlib.pyplot as plt

class EMGFilters:
    """EMG信号滤波器类"""

    def __init__(self, fs=1000):
        """
        初始化滤波器

        参数:
            fs: 采样率（Hz）
        """
        self.fs = fs

    def bandpass_filter(self, signal, lowcut=20, highcut=500, order=4):
        """
        带通滤波器

        参数:
            signal: 输入信号
            lowcut: 高通截止频率（Hz）
            highcut: 低通截止频率（Hz）
            order: 滤波器阶数

        返回:
            filtered_signal: 滤波后的信号
        """
        # 计算归一化频率（奈奎斯特频率的分数）
        nyquist = 0.5 * self.fs
        low = lowcut / nyquist
        high = highcut / nyquist

        # 设计Butterworth带通滤波器
        b, a = butter(order, [low, high], btype='band')

        # 应用零相位滤波（filtfilt避免相位失真）
        filtered_signal = filtfilt(b, a, signal)

        return filtered_signal

    def lowpass_filter(self, signal, cutoff=500, order=4):
        """
        低通滤波器

        参数:
            signal: 输入信号
            cutoff: 截止频率（Hz）
            order: 滤波器阶数

        返回:
            filtered_signal: 滤波后的信号
        """
        nyquist = 0.5 * self.fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low')
        filtered_signal = filtfilt(b, a, signal)
        return filtered_signal

    def highpass_filter(self, signal, cutoff=20, order=4):
        """
        高通滤波器

        参数:
            signal: 输入信号
            cutoff: 截止频率（Hz）
            order: 滤波器阶数

        返回:
            filtered_signal: 滤波后的信号
        """
        nyquist = 0.5 * self.fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='high')
        filtered_signal = filtfilt(b, a, signal)
        return filtered_signal

    def notch_filter(self, signal, freq=50, Q=30):
        """
        陷波滤波器（去除特定频率干扰）

        参数:
            signal: 输入信号
            freq: 要去除的频率（Hz）
            Q: 品质因子，越大带宽越窄

        返回:
            filtered_signal: 滤波后的信号
        """
        # 设计陷波滤波器
        b, a = iirnotch(freq, Q, self.fs)

        # 应用滤波
        filtered_signal = filtfilt(b, a, signal)

        return filtered_signal

    def preprocess_emg(self, signal, remove_powerline=True,
                      powerline_freq=50):
        """
        EMG信号完整预处理流程

        参数:
            signal: 输入信号
            remove_powerline: 是否去除工频干扰
            powerline_freq: 工频频率（50或60Hz）

        返回:
            processed_signal: 处理后的信号
        """
        # 1. 带通滤波（20-500Hz）
        signal_bp = self.bandpass_filter(signal, lowcut=20, highcut=500)

        # 2. 去除工频干扰（可选）
        if remove_powerline:
            signal_notch = self.notch_filter(signal_bp, freq=powerline_freq)
        else:
            signal_notch = signal_bp

        return signal_notch


def calculate_snr(signal_clean, signal_noisy):
    """
    计算信噪比（SNR）

    参数:
        signal_clean: 干净信号或滤波后信号
        signal_noisy: 含噪信号或原始信号

    返回:
        snr_db: 信噪比（分贝）
    """
    # 信号功率
    signal_power = np.mean(signal_clean ** 2)

    # 噪声功率（通过差值估计）
    noise = signal_noisy - signal_clean
    noise_power = np.mean(noise ** 2)

    # 计算SNR
    if noise_power > 0:
        snr = signal_power / noise_power
        snr_db = 10 * np.log10(snr)
    else:
        snr_db = np.inf

    return snr_db


def calculate_snr_simple(signal):
    """
    简化的SNR计算（假设信号功率>>噪声）

    参数:
        signal: 输入信号

    返回:
        snr_db: 信噪比（分贝）
    """
    signal_power = np.mean(signal ** 2)
    noise_power = np.var(signal)

    if noise_power > 0:
        snr_db = 10 * np.log10(signal_power / noise_power)
    else:
        snr_db = np.inf

    return snr_db


def normalize_signal(signal, method='zscore'):
    """
    信号归一化

    参数:
        signal: 输入信号
        method: 'zscore' 或 'minmax'

    返回:
        normalized: 归一化后的信号
    """
    if method == 'zscore':
        # Z-score标准化：(x - mean) / std
        mean = np.mean(signal)
        std = np.std(signal)
        if std > 0:
            normalized = (signal - mean) / std
        else:
            normalized = signal - mean

    elif method == 'minmax':
        # Min-Max归一化：(x - min) / (max - min)
        min_val = np.min(signal)
        max_val = np.max(signal)
        if max_val > min_val:
            normalized = (signal - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(signal)

    else:
        raise ValueError("method must be 'zscore' or 'minmax'")

    return normalized


# 示例使用
if __name__ == "__main__":
    print("\n" + "="*60)
    print("EMG滤波器模块测试".center(60))
    print("="*60 + "\n")

    # 生成测试信号
    fs = 1000
    t = np.linspace(0, 2, 2000)  # 2秒

    # EMG-like信号
    emg_signal = 0.5 * np.random.randn(len(t))  # 基础随机信号

    # 添加50Hz工频干扰
    powerline_interference = 0.3 * np.sin(2 * np.pi * 50 * t)
    signal_with_noise = emg_signal + powerline_interference

    # 创建滤波器
    filters = EMGFilters(fs=fs)

    # 应用滤波
    print("正在应用滤波器...")
    filtered_signal = filters.preprocess_emg(signal_with_noise,
                                            remove_powerline=True,
                                            powerline_freq=50)

    # 计算SNR
    snr_before = calculate_snr(emg_signal, signal_with_noise)
    snr_after = calculate_snr(emg_signal, filtered_signal)

    print(f"\n滤波前 SNR: {snr_before:.2f} dB")
    print(f"滤波后 SNR: {snr_after:.2f} dB")
    print(f"改善: {snr_after - snr_before:.2f} dB\n")

    print("✓ 滤波器模块测试完成")
    print("\n使用方法：")
    print("  from filters import EMGFilters")
    print("  filters = EMGFilters(fs=1000)")
    print("  filtered = filters.preprocess_emg(signal)\n")
