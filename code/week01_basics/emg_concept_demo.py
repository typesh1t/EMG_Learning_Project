#!/usr/bin/env python3
"""
EMG概念演示脚本
展示EMG信号的基本特征和形态
"""

import sys
from pathlib import Path
# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatter

# 兼容性处理：在某些环境（尤其是 Jupyter/IPython）里，标准库 `code` 可能先被导入，
# 导致 `code.utils.*` 不是包。这里把项目的 `code/` 目录挂到 `code.__path__`。
import code as _code
if not hasattr(_code, '__path__'):
    _code.__path__ = [str(project_root / 'code')]

# 配置中文字体
try:
    from code.utils.chinese_font_config import setup_chinese_font
    setup_chinese_font()
except:
    # 如果导入失败，使用简单配置
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False


def _is_interactive_backend():
    """判断当前 matplotlib 后端是否支持交互式显示。"""
    backend = str(plt.get_backend()).lower()
    return backend not in {'agg', 'pdf', 'ps', 'svg', 'cairo', 'template'}

def generate_emg_like_signal(duration=5, fs=1000, rest_duration=1, contraction_duration=2):
    """
    生成类EMG信号用于演示

    参数:
        duration: 总时长（秒）
        fs: 采样率（Hz）
        rest_duration: 静息时长（秒）
        contraction_duration: 收缩时长（秒）
    """
    t = np.linspace(0, duration, int(duration * fs))
    signal = np.zeros_like(t)

    # 静息阶段：低幅度噪声
    rest_noise = np.random.normal(0, 0.02, len(t))
    signal += rest_noise

    # 肌肉收缩阶段
    contraction_start = int(rest_duration * fs)
    contraction_end = int((rest_duration + contraction_duration) * fs)

    # 收缩时信号幅度增大：用多频率成分 + 噪声来模拟（更接近真实 EMG 的“宽带随机”特性）
    n = contraction_end - contraction_start
    t_seg = t[contraction_start:contraction_end]
    contraction = np.zeros(n)

    # 常见 EMG 有效频段大约在 20–450Hz，这里用 70–145Hz 的多频率叠加做示意
    for freq in range(70, 150, 15):
        amplitude = np.random.uniform(0.15, 0.35)
        phase = np.random.uniform(0, 2 * np.pi)
        contraction += amplitude * np.sin(2 * np.pi * freq * t_seg + phase)

    # 叠加随机噪声（模拟运动单元随机放电）
    contraction += np.random.normal(0, 0.15, n)

    # 包络：收缩期通常有“渐强-渐弱”，这里用 Hann 窗做示意
    envelope = np.hanning(n)
    contraction *= (0.3 + 0.7 * envelope)

    signal[contraction_start:contraction_end] += contraction

    return t, signal


def visualize_emg_concept(duration=5, fs=1000, rest_duration=1, contraction_duration=2,
                          show=None, save_path='code/week01_basics/emg_concept_demo.png'):
    """可视化EMG信号的基本概念"""

    # 生成模拟信号
    t, signal = generate_emg_like_signal(
        duration=duration,
        fs=fs,
        rest_duration=rest_duration,
        contraction_duration=contraction_duration,
    )

    # 计算一个简单的“能量包络”（整流 + 滑动平均），用于直观显示肌肉激活强度变化
    window = max(1, int(0.05 * fs))  # 50ms
    rectified = np.abs(signal)
    envelope = np.convolve(rectified, np.ones(window) / window, mode='same')

    # 计算静息/收缩段的 RMS，用于教学对比
    rest_end = int(rest_duration * fs)
    contraction_start = rest_end
    contraction_end = int((rest_duration + contraction_duration) * fs)
    rms_rest = float(np.sqrt(np.mean(signal[:rest_end] ** 2)))
    rms_contraction = float(np.sqrt(np.mean(signal[contraction_start:contraction_end] ** 2)))
    print(f"\n信号强度对比（RMS）:")
    print(f"  静息期 RMS: {rms_rest:.4f} mV")
    print(f"  收缩期 RMS: {rms_contraction:.4f} mV")

    # 创建图表
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    fig.suptitle('EMG肌电信号概念演示', fontsize=16, fontweight='bold')

    # 1. 完整信号
    axes[0].plot(t, signal, linewidth=0.5, color='blue')
    axes[0].plot(t, envelope, linewidth=1.0, color='black', alpha=0.8, label='包络(整流+平滑)')
    axes[0].set_ylabel('幅度 (mV)', fontsize=12)
    axes[0].set_title('完整EMG信号：静息 → 收缩 → 放松', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].axvspan(0, rest_duration, alpha=0.15, color='green', label='静息期')
    axes[0].axvspan(rest_duration, rest_duration + contraction_duration,
                    alpha=0.15, color='red', label='收缩期')
    axes[0].axvspan(rest_duration + contraction_duration, duration,
                    alpha=0.15, color='green', label='放松期')
    axes[0].legend(loc='upper right')
    axes[0].set_ylim(-2, 2)

    # 2. 静息期放大
    rest_samples = int(0.5 * fs)  # 前0.5秒
    axes[1].plot(t[:rest_samples], signal[:rest_samples], linewidth=0.8, color='green')
    axes[1].set_ylabel('幅度 (mV)', fontsize=12)
    axes[1].set_title('静息期放大：低幅度基线噪声', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(-0.1, 0.1)

    # 3. 收缩期放大
    axes[2].plot(t[contraction_start:contraction_end],
                signal[contraction_start:contraction_end],
                linewidth=0.8, color='red')
    axes[2].set_ylabel('幅度 (mV)', fontsize=12)
    axes[2].set_xlabel('时间 (秒)', fontsize=12)
    axes[2].set_title('收缩期放大：高幅度随机信号', fontsize=12)
    axes[2].grid(True, alpha=0.3)

    # 4. 收缩期频谱（用对数坐标更容易看清尖峰/能量分布）
    seg = signal[contraction_start:contraction_end]
    freqs = np.fft.rfftfreq(len(seg), d=1.0 / fs)
    power = (np.abs(np.fft.rfft(seg)) ** 2) / len(seg)

    axes[3].semilogy(freqs, power, linewidth=1.0, color='purple')
    axes[3].yaxis.set_major_formatter(LogFormatter(base=10, labelOnlyBase=False))
    axes[3].set_xlim(0, 300)
    axes[3].set_xlabel('频率 (Hz)', fontsize=12)
    axes[3].set_ylabel('功率', fontsize=12)
    axes[3].set_title('收缩期频谱（0–300Hz）', fontsize=12)
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"✓ 图片已保存到: {save_path}")

    if show is None:
        show = _is_interactive_backend()
    if show:
        plt.show()
    else:
        plt.close(fig)


def explain_emg_characteristics():
    """打印EMG信号的特征说明"""

    print("\n" + "="*60)
    print("EMG肌电信号的主要特征".center(60))
    print("="*60)

    characteristics = [
        ("随机性", "EMG信号看起来像噪声，没有明显的周期性模式"),
        ("突发性", "肌肉收缩时，信号幅度会突然增大"),
        ("低幅度", "信号非常微弱，通常在50微伏到5毫伏之间"),
        ("零均值", "信号在零点上下波动，平均值接近0"),
        ("非平稳", "信号特性随时间和肌肉状态变化")
    ]

    for i, (name, description) in enumerate(characteristics, 1):
        print(f"\n{i}. {name}")
        print(f"   {description}")

    print("\n" + "="*60)
    print("\n")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("EMG肌电信号概念演示程序".center(60))
    print("="*60)

    # 显示特征说明
    explain_emg_characteristics()

    # 显示可视化
    print("正在生成EMG信号可视化图表...")
    visualize_emg_concept()

    print("\n演示完成！")
    print("你已经了解了EMG信号的基本形态和特征。")
    print("\n下一步：学习EMG采集设备的组成（第2周）\n")
