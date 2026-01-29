#!/usr/bin/env python3
"""
采样率对比演示
展示不同采样率对信号重建的影响
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
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False

def sampling_rate_comparison():
    """对比不同采样率的效果"""

    # 生成原始连续信号（模拟真实EMG）
    fs_original = 10000  # 原始信号用高采样率模拟连续信号
    t_original = np.linspace(0, 1, fs_original)  # 1秒

    # EMG-like信号：主要频率50Hz和150Hz的组合
    signal_original = (np.sin(2 * np.pi * 50 * t_original) +
                      0.5 * np.sin(2 * np.pi * 150 * t_original))

    # 不同采样率
    sampling_rates = [100, 500, 1000, 2000]

    # 创建子图
    fig, axes = plt.subplots(len(sampling_rates), 2, figsize=(16, 12))
    fig.suptitle('采样率对信号重建的影响', fontsize=16, fontweight='bold')

    for i, fs in enumerate(sampling_rates):
        # 采样
        num_samples = int(fs)
        t_sampled = np.linspace(0, 1, num_samples)
        signal_sampled = (np.sin(2 * np.pi * 50 * t_sampled) +
                         0.5 * np.sin(2 * np.pi * 150 * t_sampled))

        # 左侧：时域对比
        ax_time = axes[i, 0]
        ax_time.plot(t_original, signal_original, 'b-', linewidth=0.5,
                    alpha=0.5, label='原始连续信号')
        ax_time.plot(t_sampled, signal_sampled, 'ro-', markersize=3,
                    linewidth=1, label=f'采样({fs}Hz)')
        ax_time.set_ylabel('幅度')
        ax_time.set_title(f'采样率: {fs} Hz')
        ax_time.grid(True, alpha=0.3)
        ax_time.legend(loc='upper right')
        ax_time.set_xlim(0, 0.1)  # 只显示前100ms

        if i == len(sampling_rates) - 1:
            ax_time.set_xlabel('时间 (秒)')

        # 右侧：频域分析
        ax_freq = axes[i, 1]

        # FFT分析
        from scipy.fft import fft, fftfreq
        N = len(signal_sampled)
        yf = fft(signal_sampled)
        xf = fftfreq(N, 1/fs)[:N//2]
        power = 2.0/N * np.abs(yf[:N//2])

        ax_freq.plot(xf, power, 'g-', linewidth=1.5)
        ax_freq.set_ylabel('功率')
        ax_freq.set_title(f'频谱 (Nyquist频率: {fs/2} Hz)')
        ax_freq.set_xlim(0, 300)
        ax_freq.grid(True, alpha=0.3)

        # 标注主要频率
        ax_freq.axvline(50, color='r', linestyle='--', alpha=0.5, label='50Hz')
        ax_freq.axvline(150, color='orange', linestyle='--', alpha=0.5, label='150Hz')
        ax_freq.axvline(fs/2, color='black', linestyle=':', alpha=0.5, label='Nyquist')
        ax_freq.legend(loc='upper right')

        if i == len(sampling_rates) - 1:
            ax_freq.set_xlabel('频率 (Hz)')

        # 添加评价
        if fs < 300:  # 严重欠采样
            assessment = "❌ 严重欠采样！信号失真"
            color = 'red'
        elif fs < 1000:  # 勉强够用
            assessment = "⚠️  采样率偏低，可能丢失高频成分"
            color = 'orange'
        else:  # 合适
            assessment = "✓ 采样率充足，信号重建良好"
            color = 'green'

        ax_time.text(0.05, 0.95, assessment,
                    transform=ax_time.transAxes,
                    fontsize=10,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))

    plt.tight_layout()
    plt.savefig('code/week02_device/sampling_rate_comparison.png', dpi=150)
    print("✓ 图片已保存到: code/week02_device/sampling_rate_comparison.png")
    plt.show()


def nyquist_theorem_explanation():
    """解释奈奎斯特采样定理"""

    print("\n" + "="*70)
    print("奈奎斯特采样定理 (Nyquist Sampling Theorem)".center(70))
    print("="*70)

    print("""
采样率必须至少是信号最高频率的2倍，才能完整还原信号。

公式：fs ≥ 2 × fmax

其中：
  fs = 采样率
  fmax = 信号中最高频率成分

对于EMG信号：
  - 主要频率范围：20-500 Hz
  - 最高频率：500 Hz
  - 理论最低采样率：2 × 500 = 1000 Hz
  - 实际推荐：1000-2000 Hz（留有余量）

如果采样率不足会发生什么？
  - 欠采样（Aliasing）：高频成分被错误地解释为低频
  - 信号失真：无法正确还原原始信号
  - 信息丢失：信号细节被丢失

示例：
  如果你的EMG信号包含200Hz的频率成分，但采样率只有300Hz：
  - 理论上刚好满足奈奎斯特定理（2×200=400 < 300，不满足）
  - 实际上会产生混叠，200Hz可能被误认为是100Hz
  - 正确的做法：采样率应≥400Hz，实际建议≥1000Hz
    """)

    print("="*70 + "\n")


def practical_recommendations():
    """实用建议"""

    print("\n" + "="*70)
    print("采样率选择的实用建议".center(70))
    print("="*70)

    recommendations = [
        {
            "应用场景": "入门学习、简单手势识别",
            "推荐采样率": "1000 Hz",
            "说明": "足够捕捉EMG主要特征，数据量适中"
        },
        {
            "应用场景": "科研、高精度分析",
            "推荐采样率": "2000 Hz",
            "说明": "更好的信号还原，适合详细分析"
        },
        {
            "应用场景": "实时系统、嵌入式设备",
            "推荐采样率": "1000 Hz",
            "说明": "平衡性能和处理负担"
        },
        {
            "应用场景": "疲劳检测、低频分析",
            "推荐采样率": "500-1000 Hz",
            "说明": "主要关注低频成分，可以略低"
        }
    ]

    print()
    for rec in recommendations:
        print(f"【{rec['应用场景']}】")
        print(f"  推荐采样率: {rec['推荐采样率']}")
        print(f"  说明: {rec['说明']}\n")

    print("\n常见误区：")
    print("  ❌ 误区1：采样率越高越好")
    print("     → 过高的采样率增加数据量和处理负担，没有必要\n")

    print("  ❌ 误区2：500Hz够用了，因为EMG最高到500Hz")
    print("     → 根据奈奎斯特定理，需要至少1000Hz采样率\n")

    print("  ❌ 误区3：硬件说支持2000Hz就一定能达到")
    print("     → 需要考虑通信带宽、处理器性能等实际限制\n")

    print("="*70 + "\n")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("采样率对比演示程序".center(70))
    print("="*70)

    # 解释奈奎斯特定理
    nyquist_theorem_explanation()

    # 实用建议
    practical_recommendations()

    # 可视化对比
    print("\n正在生成采样率对比图...")
    print("你将看到不同采样率如何影响信号重建质量。\n")
    sampling_rate_comparison()

    print("\n演示完成！")
    print("现在你应该理解为什么EMG采集需要1000Hz或更高的采样率。")
    print("\n下一步：运行 multichannel_demo.py 了解多通道采集\n")
