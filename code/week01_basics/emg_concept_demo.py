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

# 配置中文字体
try:
    from code.utils.chinese_font_config import setup_chinese_font
    setup_chinese_font()
except:
    # 如果导入失败，使用简单配置
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False

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

    # 收缩时信号幅度增大
    contraction_signal = np.random.normal(0, 0.5, contraction_end - contraction_start)
    signal[contraction_start:contraction_end] += contraction_signal

    return t, signal


def visualize_emg_concept():
    """可视化EMG信号的基本概念"""

    # 生成模拟信号
    t, signal = generate_emg_like_signal(duration=5, fs=1000)

    # 创建图表
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle('EMG肌电信号概念演示', fontsize=16, fontweight='bold')

    # 1. 完整信号
    axes[0].plot(t, signal, linewidth=0.5, color='blue')
    axes[0].set_ylabel('幅度 (mV)', fontsize=12)
    axes[0].set_title('完整EMG信号：静息 → 收缩 → 放松', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].axvspan(0, 1, alpha=0.2, color='green', label='静息期')
    axes[0].axvspan(1, 3, alpha=0.2, color='red', label='收缩期')
    axes[0].axvspan(3, 5, alpha=0.2, color='green', label='放松期')
    axes[0].legend(loc='upper right')
    axes[0].set_ylim(-2, 2)

    # 2. 静息期放大
    rest_samples = int(0.5 * 1000)  # 前0.5秒
    axes[1].plot(t[:rest_samples], signal[:rest_samples], linewidth=0.8, color='green')
    axes[1].set_ylabel('幅度 (mV)', fontsize=12)
    axes[1].set_title('静息期放大：低幅度基线噪声', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(-0.1, 0.1)

    # 3. 收缩期放大
    contraction_start = int(1.5 * 1000)
    contraction_end = int(2 * 1000)
    axes[2].plot(t[contraction_start:contraction_end],
                signal[contraction_start:contraction_end],
                linewidth=0.8, color='red')
    axes[2].set_ylabel('幅度 (mV)', fontsize=12)
    axes[2].set_xlabel('时间 (秒)', fontsize=12)
    axes[2].set_title('收缩期放大：高幅度随机信号', fontsize=12)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('code/week01_basics/emg_concept_demo.png', dpi=150)
    print("✓ 图片已保存到: code/week01_basics/emg_concept_demo.png")
    plt.show()


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
