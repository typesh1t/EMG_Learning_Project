#!/usr/bin/env python3
"""
EMG信号噪声识别和分析
识别常见的噪声类型：工频干扰、运动伪影、基线漂移等
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def generate_clean_emg(duration=5, fs=1000):
    """生成干净的EMG信号（用于演示）"""
    t = np.linspace(0, duration, int(duration * fs))
    # 基础EMG信号（多频率随机成分）
    signal = np.zeros(len(t))
    for freq in range(50, 200, 10):
        amplitude = np.random.uniform(0.1, 0.3)
        phase = np.random.uniform(0, 2*np.pi)
        signal += amplitude * np.sin(2 * np.pi * freq * t + phase)
    signal += np.random.normal(0, 0.1, len(t))
    return t, signal

def add_powerline_interference(signal, t, freq=50, amplitude=0.5):
    """
    添加工频干扰

    特征:
    - 固定频率（50Hz或60Hz）
    - 正弦波形
    - 在频谱上出现明显尖峰
    """
    interference = amplitude * np.sin(2 * np.pi * freq * t)
    # 添加谐波
    interference += 0.3 * amplitude * np.sin(2 * np.pi * 2 * freq * t)
    return signal + interference

def add_motion_artifact(signal, t, fs=1000):
    """
    添加运动伪影

    特征:
    - 低频大幅度波动（< 20Hz）
    - 不规则突变
    - 基线漂移
    """
    # 低频漂移
    drift_freq = 0.5  # 0.5Hz
    drift = 2.0 * np.sin(2 * np.pi * drift_freq * t)

    # 随机突变
    num_artifacts = 3
    for _ in range(num_artifacts):
        pos = np.random.randint(0, len(signal))
        width = int(fs * 0.2)  # 200ms
        if pos + width < len(signal):
            artifact = np.random.uniform(1, 3) * np.random.randn(width)
            signal[pos:pos+width] += artifact

    return signal + drift

def add_electrode_noise(signal, amplitude=0.2):
    """
    添加电极接触噪声

    特征:
    - 高频随机噪声
    - 使信号变得"毛躁"
    - 增加基线不稳定性
    """
    noise = np.random.normal(0, amplitude, len(signal))
    return signal + noise

def detect_powerline_interference(signal, fs=1000, freq_range=(45, 65)):
    """
    检测工频干扰

    方法: 分析频谱，查找50Hz或60Hz附近的尖峰
    """
    N = len(signal)
    yf = fft(signal)
    xf = fftfreq(N, 1/fs)[:N//2]
    power = 2.0/N * np.abs(yf[:N//2])

    # 查找指定范围内的峰值
    mask = (xf >= freq_range[0]) & (xf <= freq_range[1])
    if np.any(mask):
        peak_idx = np.argmax(power[mask])
        peak_freq = xf[mask][peak_idx]
        peak_power = power[mask][peak_idx]

        # 计算该频率的相对功率
        total_power = np.sum(power ** 2)
        relative_power = (peak_power ** 2) / total_power * 100

        return {
            'detected': True,
            'frequency': peak_freq,
            'power': peak_power,
            'relative_power': relative_power
        }

    return {'detected': False}

def detect_motion_artifact(signal, fs=1000, threshold=2.0):
    """
    检测运动伪影

    方法: 检查低频能量和突变
    """
    # 低通滤波获取低频成分
    b, a = butter(4, 20, btype='low', fs=fs)
    low_freq = filtfilt(b, a, signal)

    # 计算低频成分的幅度
    low_freq_amplitude = np.std(low_freq)

    # 检测突变（差分的大值）
    diff = np.abs(np.diff(signal))
    突变点数 = np.sum(diff > threshold)

    has_artifact = low_freq_amplitude > 0.5 or 突变点数 > 10

    return {
        'detected': has_artifact,
        'low_freq_amplitude': low_freq_amplitude,
        'abrupt_changes': 突变点数
    }

def analyze_noise_types(signal, fs=1000):
    """
    综合分析信号中的噪声类型
    """
    results = {}

    # 1. 工频干扰检测
    powerline = detect_powerline_interference(signal, fs)
    results['powerline_interference'] = powerline

    # 2. 运动伪影检测
    motion = detect_motion_artifact(signal, fs)
    results['motion_artifact'] = motion

    # 3. 整体噪声水平
    # 高频噪声估计（高通滤波后的标准差）
    b, a = butter(4, 200, btype='high', fs=fs)
    high_freq = filtfilt(b, a, signal)
    results['high_freq_noise_level'] = np.std(high_freq)

    # 4. 信号质量评估
    signal_power = np.mean(signal ** 2)
    noise_est = results['high_freq_noise_level'] ** 2

    if noise_est > 0:
        snr = 10 * np.log10(signal_power / noise_est)
    else:
        snr = np.inf

    results['snr_db'] = snr

    return results

def plot_noise_comparison():
    """
    可视化不同噪声类型的对比
    """
    fs = 1000
    t, clean_signal = generate_clean_emg(duration=3, fs=fs)

    # 生成不同噪声版本
    signals = {
        '干净信号': clean_signal,
        '工频干扰': add_powerline_interference(clean_signal.copy(), t, freq=50, amplitude=0.8),
        '运动伪影': add_motion_artifact(clean_signal.copy(), t, fs=fs),
        '电极噪声': add_electrode_noise(clean_signal.copy(), amplitude=0.3)
    }

    # 创建图表
    fig, axes = plt.subplots(len(signals), 2, figsize=(16, 3*len(signals)))
    fig.suptitle('EMG信号噪声类型对比', fontsize=16, fontweight='bold')

    for idx, (name, signal) in enumerate(signals.items()):
        # 时域
        ax_time = axes[idx, 0]
        ax_time.plot(t, signal, linewidth=0.5)
        ax_time.set_ylabel('幅度 (mV)')
        ax_time.set_title(f'{name} - 时域')
        ax_time.grid(True, alpha=0.3)
        ax_time.set_xlim(0, 1)  # 只显示前1秒

        # 频域
        ax_freq = axes[idx, 1]
        N = len(signal)
        yf = fft(signal)
        xf = fftfreq(N, 1/fs)[:N//2]
        power = 2.0/N * np.abs(yf[:N//2])

        ax_freq.semilogy(xf, power, linewidth=1)
        ax_freq.set_ylabel('功率')
        ax_freq.set_title(f'{name} - 频域')
        ax_freq.set_xlim(0, 300)
        ax_freq.grid(True, alpha=0.3)

        # 标注特征
        if '工频' in name:
            ax_freq.axvline(50, color='red', linestyle='--', alpha=0.7, label='50Hz')
            ax_freq.legend()
        elif '运动' in name:
            ax_time.axhspan(-3, 3, alpha=0.2, color='red', label='低频漂移')

        if idx == len(signals) - 1:
            ax_time.set_xlabel('时间 (秒)')
            ax_freq.set_xlabel('频率 (Hz)')

    plt.tight_layout()
    plt.savefig('noise_types_comparison.png', dpi=150, bbox_inches='tight')
    print("\n图表已保存到: noise_types_comparison.png")
    plt.show()

def demonstrate_noise_detection():
    """
    演示噪声检测功能
    """
    print("\n" + "="*60)
    print("噪声检测演示")
    print("="*60)

    fs = 1000
    t, clean_signal = generate_clean_emg(duration=3, fs=fs)

    # 测试不同信号
    test_signals = {
        '干净信号': clean_signal,
        '含工频干扰': add_powerline_interference(clean_signal.copy(), t, freq=50),
        '含运动伪影': add_motion_artifact(clean_signal.copy(), t, fs=fs),
        '多种噪声': add_electrode_noise(
            add_motion_artifact(
                add_powerline_interference(clean_signal.copy(), t, freq=50),
                t, fs=fs
            )
        )
    }

    for name, signal in test_signals.items():
        print(f"\n【{name}】")
        results = analyze_noise_types(signal, fs=fs)

        # 工频干扰
        if results['powerline_interference']['detected']:
            pli = results['powerline_interference']
            print(f"  工频干扰: 检测到")
            print(f"    频率: {pli['frequency']:.2f} Hz")
            print(f"    相对功率: {pli['relative_power']:.2f}%")
        else:
            print(f"  工频干扰: 未检测到")

        # 运动伪影
        if results['motion_artifact']['detected']:
            ma = results['motion_artifact']
            print(f"  运动伪影: 检测到")
            print(f"    低频幅度: {ma['low_freq_amplitude']:.3f}")
            print(f"    突变点数: {ma['abrupt_changes']}")
        else:
            print(f"  运动伪影: 未检测到")

        # 整体质量
        print(f"  高频噪声水平: {results['high_freq_noise_level']:.3f}")
        print(f"  信噪比: {results['snr_db']:.2f} dB")

    print("\n" + "="*60)

if __name__ == '__main__':
    print("\n" + "="*60)
    print("EMG噪声识别和分析工具")
    print("="*60)

    print("\n1. 生成噪声对比图...")
    plot_noise_comparison()

    print("\n2. 演示噪声检测...")
    demonstrate_noise_detection()

    print("\n" + "="*60)
    print("总结:")
    print("="*60)
    print("\n常见噪声类型及其特征:")
    print("\n1. 工频干扰 (Power Line Interference)")
    print("   - 特征: 50Hz或60Hz正弦波")
    print("   - 识别: 频谱中明显的50/60Hz尖峰")
    print("   - 解决: 陷波滤波器")

    print("\n2. 运动伪影 (Motion Artifacts)")
    print("   - 特征: 低频大幅度波动")
    print("   - 识别: 基线漂移、突然跳变")
    print("   - 解决: 高通滤波器(20Hz)、固定电极")

    print("\n3. 电极接触噪声 (Electrode Noise)")
    print("   - 特征: 高频随机噪声")
    print("   - 识别: 信号变得毛躁")
    print("   - 解决: 清洁皮肤、使用导电膏")

    print("\n4. 基线漂移 (Baseline Drift)")
    print("   - 特征: 缓慢的低频趋势")
    print("   - 识别: 信号整体上升或下降")
    print("   - 解决: 高通滤波、去趋势处理")

    print("\n信号质量评估标准:")
    print("  SNR > 20 dB: 优秀")
    print("  SNR 10-20 dB: 良好")
    print("  SNR < 10 dB: 较差，需要改善")

    print("\n" + "="*60)
    print("\n完成！请查看生成的图表了解不同噪声类型。")
