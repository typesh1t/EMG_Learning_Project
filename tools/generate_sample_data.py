#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
样本EMG数据生成器

用于生成模拟的EMG信号数据，用于学习和测试。
模拟手部抓紧和松开的动作。

使用方法:
    python generate_sample_data.py --output ../data/sample/ --subjects 5 --trials 10

作者: EMG Learning Project
日期: 2026-01-29
"""

import numpy as np
import pandas as pd
import os
import argparse
from pathlib import Path


class EMGSignalGenerator:
    """EMG信号生成器"""

    def __init__(self, fs=1000, duration=5.0):
        """
        初始化

        参数:
            fs: 采样率 (Hz)
            duration: 信号时长 (秒)
        """
        self.fs = fs
        self.duration = duration
        self.n_samples = int(fs * duration)
        self.time = np.linspace(0, duration, self.n_samples)

    def generate_noise(self, amplitude=0.05):
        """
        生成基础噪声

        参数:
            amplitude: 噪声幅度

        返回:
            noise: 噪声信号
        """
        # 白噪声
        white_noise = np.random.normal(0, amplitude, self.n_samples)

        # 添加一些低频漂移（模拟基线漂移）
        drift_freq = 0.5  # Hz
        drift = 0.02 * np.sin(2 * np.pi * drift_freq * self.time)

        return white_noise + drift

    def generate_powerline_interference(self, freq=50, amplitude=0.01):
        """
        生成工频干扰

        参数:
            freq: 工频频率 (Hz), 50 或 60
            amplitude: 干扰幅度

        返回:
            interference: 工频干扰信号
        """
        interference = amplitude * np.sin(2 * np.pi * freq * self.time)
        # 添加谐波
        interference += 0.5 * amplitude * np.sin(2 * np.pi * 2 * freq * self.time)
        return interference

    def generate_muscle_activation(self, start_time, end_time, intensity=1.0):
        """
        生成肌肉激活信号（突发性随机信号）

        参数:
            start_time: 激活开始时间 (秒)
            end_time: 激活结束时间 (秒)
            intensity: 激活强度 (0-1)

        返回:
            activation: 激活信号
        """
        activation = np.zeros(self.n_samples)

        # 确定激活区间的样本索引
        start_idx = int(start_time * self.fs)
        end_idx = int(end_time * self.fs)

        # 生成激活区间的信号
        activation_length = end_idx - start_idx

        # 使用多个频率成分的随机信号模拟EMG
        for freq in range(50, 200, 10):
            amplitude = intensity * np.random.uniform(0.1, 0.3)
            phase = np.random.uniform(0, 2*np.pi)
            activation[start_idx:end_idx] += amplitude * np.sin(
                2 * np.pi * freq * self.time[start_idx:end_idx] + phase
            )

        # 添加随机突发
        burst_noise = np.random.normal(0, 0.3 * intensity, activation_length)
        activation[start_idx:end_idx] += burst_noise

        # 平滑过渡（避免突变）
        window_size = int(0.1 * self.fs)  # 100ms过渡

        # 上升沿
        if start_idx + window_size < end_idx:
            ramp_up = np.linspace(0, 1, window_size)
            activation[start_idx:start_idx+window_size] *= ramp_up

        # 下降沿
        if end_idx - window_size > start_idx:
            ramp_down = np.linspace(1, 0, window_size)
            activation[end_idx-window_size:end_idx] *= ramp_down

        return activation

    def generate_gesture(self, gesture_type, intensity=None):
        """
        生成特定手势的EMG信号

        参数:
            gesture_type: 手势类型
                - 'rest': 静息
                - 'fist': 握拳
                - 'open': 张开
                - 'flex': 腕屈
                - 'extend': 腕伸
            intensity: 激活强度，None则随机生成

        返回:
            signal: EMG信号
        """
        # 基础噪声
        signal = self.generate_noise(amplitude=0.05)

        # 添加工频干扰
        signal += self.generate_powerline_interference(freq=50, amplitude=0.01)

        # 根据手势类型添加激活
        if gesture_type == 'rest':
            # 静息：只有噪声，无激活
            pass

        elif gesture_type == 'fist':
            # 握拳：在中间时段有强烈激活
            if intensity is None:
                intensity = np.random.uniform(0.7, 1.0)

            # 持续收缩
            signal += self.generate_muscle_activation(
                start_time=1.0,
                end_time=4.0,
                intensity=intensity
            )

        elif gesture_type == 'open':
            # 张开：相对较弱的激活
            if intensity is None:
                intensity = np.random.uniform(0.3, 0.6)

            signal += self.generate_muscle_activation(
                start_time=1.0,
                end_time=4.0,
                intensity=intensity
            )

        elif gesture_type == 'flex':
            # 腕屈
            if intensity is None:
                intensity = np.random.uniform(0.5, 0.8)

            signal += self.generate_muscle_activation(
                start_time=1.5,
                end_time=3.5,
                intensity=intensity
            )

        elif gesture_type == 'extend':
            # 腕伸
            if intensity is None:
                intensity = np.random.uniform(0.5, 0.8)

            signal += self.generate_muscle_activation(
                start_time=1.5,
                end_time=3.5,
                intensity=intensity
            )

        else:
            raise ValueError(f"未知的手势类型: {gesture_type}")

        return signal

    def generate_multichannel(self, gesture_type, n_channels=4, intensity=None):
        """
        生成多通道EMG信号

        参数:
            gesture_type: 手势类型
            n_channels: 通道数
            intensity: 激活强度

        返回:
            signals: 多通道信号 (n_samples, n_channels)
        """
        signals = np.zeros((self.n_samples, n_channels))

        for ch in range(n_channels):
            # 每个通道有轻微不同的强度
            if intensity is not None:
                ch_intensity = intensity * np.random.uniform(0.8, 1.2)
            else:
                ch_intensity = None

            signals[:, ch] = self.generate_gesture(gesture_type, ch_intensity)

        return signals


def generate_dataset(output_dir, n_subjects=5, n_trials_per_gesture=10,
                     gestures=['rest', 'fist', 'open'], n_channels=4,
                     fs=1000, duration=5.0):
    """
    生成完整的样本数据集

    参数:
        output_dir: 输出目录
        n_subjects: 受试者数量
        n_trials_per_gesture: 每种手势的试验次数
        gestures: 手势列表
        n_channels: 通道数
        fs: 采样率
        duration: 每个试验的时长
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    generator = EMGSignalGenerator(fs=fs, duration=duration)

    print(f"开始生成样本数据集...")
    print(f"  受试者数: {n_subjects}")
    print(f"  手势类型: {gestures}")
    print(f"  每种手势试验数: {n_trials_per_gesture}")
    print(f"  通道数: {n_channels}")
    print(f"  采样率: {fs} Hz")
    print(f"  时长: {duration} 秒")
    print(f"  输出目录: {output_path}")
    print()

    for subject_id in range(1, n_subjects + 1):
        subject_dir = output_path / f"subject_{subject_id:02d}"

        for gesture in gestures:
            gesture_dir = subject_dir / gesture
            gesture_dir.mkdir(parents=True, exist_ok=True)

            for trial in range(1, n_trials_per_gesture + 1):
                # 生成多通道信号
                signals = generator.generate_multichannel(
                    gesture_type=gesture,
                    n_channels=n_channels
                )

                # 创建DataFrame
                time = generator.time
                columns = ['time'] + [f'channel_{i}' for i in range(n_channels)] + ['label']

                data = np.column_stack([
                    time,
                    signals,
                    np.full(len(time), gesture)
                ])

                df = pd.DataFrame(data, columns=columns)

                # 保存为CSV
                filename = f"trial_{trial:03d}.csv"
                filepath = gesture_dir / filename
                df.to_csv(filepath, index=False)

                print(f"  生成: subject_{subject_id:02d}/{gesture}/{filename}")

    print(f"\n✅ 数据集生成完成！")
    print(f"总文件数: {n_subjects * len(gestures) * n_trials_per_gesture}")

    # 生成README
    readme_content = f"""# 样本EMG数据集说明

## 数据集信息

- **生成日期**: 自动生成
- **受试者数**: {n_subjects}
- **手势类型**: {', '.join(gestures)}
- **每种手势试验数**: {n_trials_per_gesture}
- **通道数**: {n_channels}
- **采样率**: {fs} Hz
- **每个试验时长**: {duration} 秒

## 目录结构

```
sample/
├── subject_01/
│   ├── rest/
│   │   ├── trial_001.csv
│   │   ├── trial_002.csv
│   │   └── ...
│   ├── fist/
│   │   └── ...
│   └── open/
│       └── ...
├── subject_02/
│   └── ...
└── README.md (本文件)
```

## 数据格式

每个CSV文件包含以下列:
- `time`: 时间戳（秒）
- `channel_0` ~ `channel_{n_channels-1}`: EMG信号（mV）
- `label`: 手势标签

## 信号特征

- **静息 (rest)**: 只有基础噪声和工频干扰
- **握拳 (fist)**: 强烈的肌肉激活（1-4秒）
- **张开 (open)**: 中等强度的肌肉激活（1-4秒）

## 使用示例

```python
import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_csv('subject_01/fist/trial_001.csv')

# 绘制第一个通道
plt.figure(figsize=(12, 4))
plt.plot(data['time'], data['channel_0'])
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (mV)')
plt.title('EMG Signal - Fist Gesture')
plt.grid(True)
plt.show()
```

## 注意事项

⚠️ **这是模拟数据**: 本数据集是通过算法生成的模拟EMG信号，用于教学和测试目的。
真实的EMG信号会有更多复杂性和变化。

## 生成脚本

使用 `generate_sample_data.py` 生成，可自定义参数：

```bash
python generate_sample_data.py \\
    --output ../data/sample/ \\
    --subjects 5 \\
    --trials 10 \\
    --gestures rest fist open \\
    --channels 4 \\
    --fs 1000 \\
    --duration 5.0
```
"""

    readme_path = output_path / "README.md"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)

    print(f"\n📄 生成 README: {readme_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='生成模拟的EMG样本数据集'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='../data/sample/',
        help='输出目录路径'
    )

    parser.add_argument(
        '--subjects',
        type=int,
        default=5,
        help='生成的受试者数量'
    )

    parser.add_argument(
        '--trials',
        type=int,
        default=10,
        help='每种手势的试验次数'
    )

    parser.add_argument(
        '--gestures',
        nargs='+',
        default=['rest', 'fist', 'open'],
        help='要生成的手势类型'
    )

    parser.add_argument(
        '--channels',
        type=int,
        default=4,
        help='EMG通道数'
    )

    parser.add_argument(
        '--fs',
        type=int,
        default=1000,
        help='采样率 (Hz)'
    )

    parser.add_argument(
        '--duration',
        type=float,
        default=5.0,
        help='每个试验的时长 (秒)'
    )

    args = parser.parse_args()

    # 生成数据集
    generate_dataset(
        output_dir=args.output,
        n_subjects=args.subjects,
        n_trials_per_gesture=args.trials,
        gestures=args.gestures,
        n_channels=args.channels,
        fs=args.fs,
        duration=args.duration
    )


if __name__ == '__main__':
    main()
