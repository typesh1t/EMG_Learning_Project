#!/usr/bin/env python3
"""
EMG信号查看器
用于可视化EMG数据文件
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def load_emg_data(filepath):
    """
    加载EMG数据

    参数:
        filepath: CSV文件路径

    返回:
        data: pandas DataFrame
    """
    data = pd.read_csv(filepath)
    print(f"✓ 加载数据: {filepath}")
    print(f"  形状: {data.shape}")
    print(f"  列名: {data.columns.tolist()}")

    # 识别通道列
    channel_cols = [col for col in data.columns if col.startswith('channel_')]
    print(f"  检测到 {len(channel_cols)} 个EMG通道")

    return data

def plot_multichannel_signal(data, save_path=None):
    """
    绘制多通道EMG信号

    参数:
        data: pandas DataFrame
        save_path: 保存路径（可选）
    """
    # 识别通道列
    channel_cols = [col for col in data.columns if col.startswith('channel_')]
    n_channels = len(channel_cols)

    # 时间轴
    if 'time' in data.columns:
        time = data['time'].values
        xlabel = '时间 (秒)'
    else:
        time = np.arange(len(data))
        xlabel = '样本索引'

    # 创建子图
    fig, axes = plt.subplots(n_channels, 1, figsize=(14, 3*n_channels))

    if n_channels == 1:
        axes = [axes]

    # 获取标签（如果有）
    label = data['label'].iloc[0] if 'label' in data.columns else '未知'
    fig.suptitle(f'EMG多通道信号 - 手势: {label}', fontsize=16, fontweight='bold')

    for i, (ax, ch_name) in enumerate(zip(axes, channel_cols)):
        signal = data[ch_name].values

        ax.plot(time, signal, linewidth=0.5)
        ax.set_ylabel(f'{ch_name}\n(mV)', fontsize=10)
        ax.grid(True, alpha=0.3)

        # 计算统计量
        rms = np.sqrt(np.mean(signal**2))
        peak = np.max(np.abs(signal))
        ax.text(0.02, 0.95, f'RMS={rms:.3f} | Peak={peak:.3f}',
               transform=ax.transAxes,
               verticalalignment='top',
               fontsize=9,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # 只在最后一个子图显示x轴标签
        if i < n_channels - 1:
            ax.set_xticks([])
        else:
            ax.set_xlabel(xlabel, fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ 图表已保存到: {save_path}")

    plt.show()


def plot_signal_statistics(data, save_path=None):
    """
    绘制信号统计信息

    参数:
        data: pandas DataFrame
        save_path: 保存路径（可选）
    """
    # 识别通道列
    channel_cols = [col for col in data.columns if col.startswith('channel_')]

    # 计算统计量
    stats = {
        'Channel': [],
        'Mean': [],
        'Std': [],
        'RMS': [],
        'Peak': [],
        'Min': [],
        'Max': []
    }

    for ch_name in channel_cols:
        signal = data[ch_name].values
        stats['Channel'].append(ch_name)
        stats['Mean'].append(np.mean(signal))
        stats['Std'].append(np.std(signal))
        stats['RMS'].append(np.sqrt(np.mean(signal**2)))
        stats['Peak'].append(np.max(np.abs(signal)))
        stats['Min'].append(np.min(signal))
        stats['Max'].append(np.max(signal))

    stats_df = pd.DataFrame(stats)

    # 绘制柱状图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('信号统计特征', fontsize=16, fontweight='bold')

    metrics = ['RMS', 'Peak', 'Std', 'Mean']
    for ax, metric in zip(axes.flatten(), metrics):
        ax.bar(stats_df['Channel'], stats_df[metric])
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} 各通道对比')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ 统计图已保存到: {save_path}")

    plt.show()

    # 打印统计表
    print("\n信号统计信息:")
    print(stats_df.to_string(index=False))
    print()


def plot_frequency_spectrum(data, fs=1000, save_path=None):
    """
    绘制频谱

    参数:
        data: pandas DataFrame
        fs: 采样率
        save_path: 保存路径（可选）
    """
    from scipy.fft import fft, fftfreq

    # 识别通道列
    channel_cols = [col for col in data.columns if col.startswith('channel_')]
    n_channels = len(channel_cols)

    # 创建子图
    fig, axes = plt.subplots(n_channels, 1, figsize=(14, 3*n_channels))

    if n_channels == 1:
        axes = [axes]

    fig.suptitle('功率谱分析', fontsize=16, fontweight='bold')

    for i, (ax, ch_name) in enumerate(zip(axes, channel_cols)):
        signal = data[ch_name].values
        N = len(signal)

        # FFT
        yf = fft(signal)
        xf = fftfreq(N, 1/fs)[:N//2]
        power = 2.0/N * np.abs(yf[:N//2])

        ax.semilogy(xf, power, linewidth=1)
        ax.set_ylabel(f'{ch_name}\n功率')
        ax.set_xlim(0, 300)  # 只显示0-300Hz
        ax.grid(True, alpha=0.3)

        # 标注主要频率
        peak_idx = np.argmax(power)
        peak_freq = xf[peak_idx]
        ax.axvline(peak_freq, color='r', linestyle='--', alpha=0.5)
        ax.text(peak_freq + 10, power[peak_idx],
               f'Peak: {peak_freq:.1f}Hz',
               fontsize=9)

        if i == n_channels - 1:
            ax.set_xlabel('频率 (Hz)')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ 频谱图已保存到: {save_path}")

    plt.show()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='EMG信号可视化工具')

    parser.add_argument('filepath', type=str, help='EMG数据文件路径 (CSV)')
    parser.add_argument('--fs', type=int, default=1000, help='采样率 (Hz)')
    parser.add_argument('--stats', action='store_true', help='显示统计图')
    parser.add_argument('--spectrum', action='store_true', help='显示频谱')
    parser.add_argument('--save', type=str, help='保存图片的前缀路径')

    args = parser.parse_args()

    # 加载数据
    data = load_emg_data(args.filepath)

    # 绘制信号
    save_signal = f"{args.save}_signal.png" if args.save else None
    plot_multichannel_signal(data, save_path=save_signal)

    # 绘制统计（可选）
    if args.stats:
        save_stats = f"{args.save}_stats.png" if args.save else None
        plot_signal_statistics(data, save_path=save_stats)

    # 绘制频谱（可选）
    if args.spectrum:
        save_spectrum = f"{args.save}_spectrum.png" if args.save else None
        plot_frequency_spectrum(data, fs=args.fs, save_path=save_spectrum)


if __name__ == '__main__':
    main()
