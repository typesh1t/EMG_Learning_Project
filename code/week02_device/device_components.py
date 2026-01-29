#!/usr/bin/env python3
"""
EMG设备组件交互式演示
展示EMG采集系统中各个组件的作用
"""

import sys
from pathlib import Path
# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# 配置中文字体
try:
    from code.utils.chinese_font_config import setup_chinese_font
    setup_chinese_font()
except:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False

def simulate_emg_pipeline():
    """模拟EMG信号采集流水线"""

    # 创建图表
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('EMG采集系统组件演示 - 交互式', fontsize=16, fontweight='bold')

    # 1. 原始肌肉信号
    ax1 = plt.subplot(3, 2, 1)
    ax1.set_title('1. 原始肌肉电信号（极微弱）')
    ax1.set_ylabel('电压 (μV)')
    ax1.set_xlabel('时间 (ms)')

    # 生成原始信号（微弱）
    t = np.linspace(0, 100, 1000)  # 100ms
    original_signal = 0.1 * np.sin(2 * np.pi * 50 * t / 1000)  # 50Hz主频
    original_signal += 0.05 * np.random.randn(len(t))  # 添加噪声
    line1, = ax1.plot(t, original_signal * 1000, linewidth=0.8)  # 转换为μV
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-200, 200)

    # 2. 电极检测
    ax2 = plt.subplot(3, 2, 2)
    ax2.set_title('2. 电极检测到的信号（含噪声）')
    ax2.set_ylabel('电压 (μV)')
    ax2.set_xlabel('时间 (ms)')

    # 电极会引入额外噪声
    electrode_noise = 0.02 * np.random.randn(len(t))
    electrode_signal = original_signal + electrode_noise
    line2, = ax2.plot(t, electrode_signal * 1000, linewidth=0.8, color='orange')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-200, 200)

    # 3. 放大器输出
    ax3 = plt.subplot(3, 2, 3)
    ax3.set_title('3. 放大器输出（放大1000倍）')
    ax3.set_ylabel('电压 (mV)')
    ax3.set_xlabel('时间 (ms)')

    gain = 1000
    amplified_signal = electrode_signal * gain
    line3, = ax3.plot(t, amplified_signal, linewidth=0.8, color='green')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(0, color='black', linewidth=0.5)

    # 4. 硬件滤波后
    ax4 = plt.subplot(3, 2, 4)
    ax4.set_title('4. 硬件滤波后（20-500Hz）')
    ax4.set_ylabel('电压 (mV)')
    ax4.set_xlabel('时间 (ms)')

    # 简单的滤波效果（去除部分高频噪声）
    from scipy.signal import butter, filtfilt
    b, a = butter(4, [20, 500], btype='band', fs=10000)
    filtered_signal = filtfilt(b, a, amplified_signal)
    line4, = ax4.plot(t, filtered_signal, linewidth=0.8, color='purple')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(0, color='black', linewidth=0.5)

    # 5. ADC采样
    ax5 = plt.subplot(3, 2, 5)
    ax5.set_title('5. ADC数字化（采样率1000Hz）')
    ax5.set_ylabel('电压 (mV)')
    ax5.set_xlabel('时间 (ms)')

    # 模拟采样（降采样）
    sampling_rate = 100  # 每100ms采样10个点
    sampled_indices = np.arange(0, len(t), len(t)//sampling_rate)
    sampled_t = t[sampled_indices]
    sampled_signal = filtered_signal[sampled_indices]
    ax5.plot(t, filtered_signal, linewidth=0.5, alpha=0.3, color='gray', label='原始')
    ax5.plot(sampled_t, sampled_signal, 'ro-', markersize=4, linewidth=1, label='采样点')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    ax5.axhline(0, color='black', linewidth=0.5)

    # 6. 最终数字信号
    ax6 = plt.subplot(3, 2, 6)
    ax6.set_title('6. 最终数字信号（可供计算机处理）')
    ax6.set_ylabel('电压 (mV)')
    ax6.set_xlabel('时间 (ms)')

    ax6.stem(sampled_t, sampled_signal, basefmt=' ', linefmt='b-', markerfmt='bo')
    ax6.grid(True, alpha=0.3)
    ax6.axhline(0, color='black', linewidth=0.5)

    plt.tight_layout()
    plt.savefig('code/week02_device/emg_pipeline_demo.png', dpi=150)
    print("✓ 图片已保存到: code/week02_device/emg_pipeline_demo.png")
    plt.show()


def explain_components():
    """打印各组件的说明"""

    print("\n" + "="*70)
    print("EMG采集系统组件详解".center(70))
    print("="*70)

    components = [
        {
            "name": "1. 电极 (Electrode)",
            "function": "将肌肉的电信号传导到设备",
            "types": "表面电极（本课程使用）、针电极（医疗）",
            "note": "电极质量直接影响信号质量"
        },
        {
            "name": "2. 放大器 (Amplifier)",
            "function": "将微弱信号放大到可处理范围",
            "types": "增益通常为1000-10000倍",
            "note": "使用差分放大器减少共模干扰"
        },
        {
            "name": "3. 滤波器 (Filter)",
            "function": "去除不需要的频率成分",
            "types": "带通滤波器（20-500Hz）、陷波滤波器（50/60Hz）",
            "note": "硬件滤波在采集阶段，软件滤波在后处理"
        },
        {
            "name": "4. ADC转换器",
            "function": "将模拟信号转换为数字信号",
            "types": "分辨率12-24位，采样率1000-2000Hz",
            "note": "采样率必须满足奈奎斯特定理（≥2倍信号最高频率）"
        },
        {
            "name": "5. 微控制器/处理器",
            "function": "控制采集、传输数据到计算机",
            "types": "Arduino, STM32, ESP32等",
            "note": "负责数据打包和通信"
        },
        {
            "name": "6. 通信接口",
            "function": "将数据传输到计算机",
            "types": "USB、蓝牙、WiFi、串口",
            "note": "需要足够的带宽保证实时传输"
        }
    ]

    for comp in components:
        print(f"\n{comp['name']}")
        print(f"  功能: {comp['function']}")
        print(f"  类型: {comp['types']}")
        print(f"  注意: {comp['note']}")

    print("\n" + "="*70 + "\n")


def key_parameters():
    """讲解关键参数"""

    print("\n" + "="*70)
    print("EMG设备关键参数".center(70))
    print("="*70)

    params = [
        {
            "name": "采样率 (Sampling Rate)",
            "unit": "Hz（次/秒）",
            "recommended": "1000-2000 Hz",
            "reason": "EMG信号主要频率在20-500Hz，采样率应≥1000Hz",
            "warning": "太低会丢失信号细节（欠采样），太高会增加数据量"
        },
        {
            "name": "增益 (Gain)",
            "unit": "倍数",
            "recommended": "1000-10000倍",
            "reason": "EMG信号只有50μV-5mV，需放大到ADC可识别范围",
            "warning": "增益过大会导致信号饱和（削波），过小则淹没在噪声中"
        },
        {
            "name": "分辨率 (Resolution)",
            "unit": "位(bit)",
            "recommended": "12-16位",
            "reason": "分辨率决定能表示多少个不同的电压值",
            "warning": "16位分辨率可以表示65536个不同的值"
        },
        {
            "name": "带宽 (Bandwidth)",
            "unit": "Hz",
            "recommended": "20-500 Hz",
            "reason": "EMG信号的有效频率范围",
            "warning": "带宽外的信号通常是噪声"
        },
        {
            "name": "输入阻抗",
            "unit": "MΩ（兆欧）",
            "recommended": ">10 MΩ",
            "reason": "高输入阻抗减少对皮肤电极的影响",
            "warning": "低阻抗会导致信号衰减"
        }
    ]

    for param in params:
        print(f"\n【{param['name']}】")
        print(f"  单位: {param['unit']}")
        print(f"  推荐值: {param['recommended']}")
        print(f"  原因: {param['reason']}")
        print(f"  注意: {param['warning']}")

    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("EMG设备组件演示程序".center(70))
    print("="*70)

    # 打印组件说明
    explain_components()

    # 打印关键参数
    key_parameters()

    # 显示可视化
    print("\n正在生成EMG采集流水线可视化...")
    print("你将看到信号从原始肌肉电信号到最终数字信号的完整过程。\n")
    simulate_emg_pipeline()

    print("\n演示完成！")
    print("现在你应该理解了EMG采集系统中每个组件的作用。")
    print("\n下一步：运行 sampling_demo.py 理解采样率的重要性\n")
