#!/usr/bin/env python3
"""
EMG特征提取模块
包含时域和频域特征提取函数
"""

import numpy as np
from scipy.fft import fft, fftfreq

class EMGFeatures:
    """EMG特征提取类"""

    @staticmethod
    def extract_time_features(signal):
        """
        提取时域特征

        参数:
            signal: 一维numpy数组

        返回:
            features: 字典，包含各种时域特征
        """
        N = len(signal)

        # 1. MAV - Mean Absolute Value (平均绝对值)
        mav = np.mean(np.abs(signal))

        # 2. RMS - Root Mean Square (均方根)
        rms = np.sqrt(np.mean(signal ** 2))

        # 3. VAR - Variance (方差)
        var = np.var(signal)

        # 4. WL - Waveform Length (波形长度)
        wl = np.sum(np.abs(np.diff(signal)))

        # 5. ZC - Zero Crossing (过零率)
        threshold = 0.01  # 阈值避免噪声影响
        zc = 0
        for i in range(len(signal) - 1):
            if ((signal[i] > threshold and signal[i+1] < -threshold) or
                (signal[i] < -threshold and signal[i+1] > threshold)):
                zc += 1

        # 6. SSC - Slope Sign Change (斜率符号变化)
        ssc = 0
        for i in range(1, len(signal) - 1):
            if ((signal[i] - signal[i-1]) * (signal[i] - signal[i+1]) > threshold):
                ssc += 1

        # 7. IEMG - Integrated EMG (积分EMG)
        iemg = np.sum(np.abs(signal))

        # 8. DASDV - Difference Absolute Standard Deviation Value
        dasdv = np.sqrt(np.mean(np.diff(signal) ** 2))

        # 9. Peak Value (峰值)
        peak = np.max(np.abs(signal))

        # 10. Mean Value (均值，应接近0)
        mean = np.mean(signal)

        features = {
            'MAV': mav,
            'RMS': rms,
            'VAR': var,
            'WL': wl,
            'ZC': zc,
            'SSC': ssc,
            'IEMG': iemg,
            'DASDV': dasdv,
            'PEAK': peak,
            'MEAN': mean
        }

        return features

    @staticmethod
    def extract_freq_features(signal, fs=1000, return_spectrum=False):
        """
        提取频域特征

        参数:
            signal: 一维numpy数组
            fs: 采样率（Hz）

        返回:
            features: 字典，包含频域特征
        """
        N = len(signal)

        # 计算FFT
        yf = fft(signal)
        xf = fftfreq(N, 1/fs)[:N//2]

        # 功率谱（只取正频率部分）
        power = (2.0 / N) * np.abs(yf[:N//2])
        power_squared = power ** 2

        # 1. Total Power (总功率)
        total_power = np.sum(power_squared)

        # 2. Mean Frequency (MNF - 平均频率)
        if total_power > 0:
            mnf = np.sum(xf * power_squared) / np.sum(power_squared)
        else:
            mnf = 0

        # 3. Median Frequency (MDF - 中值频率)
        cumsum = np.cumsum(power_squared)
        if cumsum[-1] > 0:
            mdf_idx = np.where(cumsum >= cumsum[-1] / 2)[0]
            mdf = xf[mdf_idx[0]] if len(mdf_idx) > 0 else 0
        else:
            mdf = 0

        # 4. Peak Frequency (峰值频率)
        peak_freq = xf[np.argmax(power)] if len(power) > 0 else 0

        # 5. Spectral Moments (频谱矩)
        if total_power > 0:
            sm1 = np.sum((xf ** 1) * power_squared) / total_power
            sm2 = np.sum((xf ** 2) * power_squared) / total_power
            sm3 = np.sum((xf ** 3) * power_squared) / total_power
        else:
            sm1 = sm2 = sm3 = 0

        # 6. Frequency Ratio (频率比)
        # 低频能量 vs 高频能量
        low_freq_idx = xf < 100
        high_freq_idx = (xf >= 100) & (xf < 300)

        low_freq_power = np.sum(power_squared[low_freq_idx])
        high_freq_power = np.sum(power_squared[high_freq_idx])

        if low_freq_power > 0:
            freq_ratio = high_freq_power / low_freq_power
        else:
            freq_ratio = 0

        # 默认只返回“数值特征”，避免把频谱数组也拼进特征向量导致维度错误。
        features = {
            'MNF': mnf,
            'MDF': mdf,
            'Peak_Freq': peak_freq,
            'Total_Power': total_power,
            'SM1': sm1,
            'SM2': sm2,
            'SM3': sm3,
            'Freq_Ratio': freq_ratio
        }

        if return_spectrum:
            features['power_spectrum'] = power
            features['frequencies'] = xf

        return features

    @staticmethod
    def sliding_window_features(signal, window_size=200, step=100, fs=1000,
                                include_time=True, include_freq=True):
        """
        使用滑动窗口提取特征

        参数:
            signal: 输入信号
            window_size: 窗口大小（样本数）
            step: 步长（样本数）
            fs: 采样率

        返回:
            feature_matrix: 特征矩阵 (n_windows, n_features)
            feature_names: 特征名称列表
            window_times: 每个窗口的中心时间
        """
        if not include_time and not include_freq:
            raise ValueError("At least one of include_time/include_freq must be True")

        num_windows = (len(signal) - window_size) // step + 1

        all_features = []
        window_times = []

        for i in range(num_windows):
            start = i * step
            end = start + window_size
            window = signal[start:end]

            # 窗口中心时间
            center_time = (start + end) / 2 / fs
            window_times.append(center_time)

            # 提取时域特征
            time_feat = EMGFeatures.extract_time_features(window) if include_time else None

            # 提取频域特征
            freq_feat = EMGFeatures.extract_freq_features(window, fs) if include_freq else None

            # 合并特征（保持固定顺序，确保特征矩阵列一致）
            features = {}

            if include_time:
                for name in ['MAV', 'RMS', 'VAR', 'WL', 'ZC', 'SSC',
                             'IEMG', 'DASDV', 'PEAK', 'MEAN']:
                    features[name] = time_feat[name]

            if include_freq:
                for name in ['MNF', 'MDF', 'Peak_Freq', 'Total_Power',
                             'SM1', 'SM2', 'SM3', 'Freq_Ratio']:
                    features[name] = freq_feat[name]

            all_features.append(features)

        # 转换为矩阵
        feature_names = list(all_features[0].keys())
        feature_matrix = np.array([[f[name] for name in feature_names]
                                   for f in all_features])

        return feature_matrix, feature_names, np.array(window_times)


# 示例使用
if __name__ == "__main__":
    print("\n" + "="*60)
    print("EMG特征提取模块测试".center(60))
    print("="*60 + "\n")

    # 生成测试信号
    fs = 1000
    t = np.linspace(0, 2, 2000)  # 2秒
    signal = np.random.randn(len(t)) * 0.5

    # 模拟肌肉收缩（1-1.5秒）
    contract_start = 1000
    contract_end = 1500
    signal[contract_start:contract_end] *= 3

    print("提取单个窗口的特征...")
    window = signal[1000:1200]  # 200个样本

    # 时域特征
    time_features = EMGFeatures.extract_time_features(window)
    print("\n时域特征:")
    for name, value in time_features.items():
        if name not in ['power_spectrum', 'frequencies']:
            print(f"  {name:12s}: {value:.6f}")

    # 频域特征
    freq_features = EMGFeatures.extract_freq_features(window, fs)
    print("\n频域特征:")
    for name, value in freq_features.items():
        if name not in ['power_spectrum', 'frequencies']:
            print(f"  {name:12s}: {value:.6f}")

    # 滑动窗口特征
    print("\n\n使用滑动窗口提取特征...")
    features, names, times = EMGFeatures.sliding_window_features(
        signal, window_size=200, step=100, fs=fs
    )

    print(f"提取了 {len(features)} 个窗口的特征")
    print(f"每个窗口有 {len(names)} 个特征")
    print(f"特征名称: {names}")

    print("\n✓ 特征提取模块测试完成")
    print("\n使用方法：")
    print("  from features import EMGFeatures")
    print("  features = EMGFeatures.extract_time_features(signal)")
    print("  features_matrix, names, times = EMGFeatures.sliding_window_features(signal)\n")
