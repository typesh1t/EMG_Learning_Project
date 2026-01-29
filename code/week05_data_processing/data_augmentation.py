#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EMG数据增强

提供多种数据增强方法，用于扩充训练数据集，提高模型泛化能力。

作者: EMG Learning Project
日期: 2026-01-29
"""

import numpy as np
from typing import Tuple, Optional


class EMGDataAugmentation:
    """EMG数据增强类"""

    @staticmethod
    def add_noise(signal: np.ndarray, noise_level: float = 0.05) -> np.ndarray:
        """
        添加高斯白噪声

        参数:
            signal: 输入信号
            noise_level: 噪声水平（标准差相对于信号标准差的比例）

        返回:
            augmented: 增强后的信号
        """
        signal_std = np.std(signal)
        noise = np.random.normal(0, noise_level * signal_std, signal.shape)
        return signal + noise

    @staticmethod
    def amplitude_scaling(signal: np.ndarray, scale_range: Tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
        """
        幅度缩放

        参数:
            signal: 输入信号
            scale_range: 缩放范围 (min_scale, max_scale)

        返回:
            augmented: 增强后的信号
        """
        scale_factor = np.random.uniform(scale_range[0], scale_range[1])
        return signal * scale_factor

    @staticmethod
    def time_shift(signal: np.ndarray, max_shift: int = 100) -> np.ndarray:
        """
        时间平移

        参数:
            signal: 输入信号
            max_shift: 最大平移样本数

        返回:
            augmented: 增强后的信号
        """
        shift = np.random.randint(-max_shift, max_shift)

        if shift > 0:
            # 右移
            augmented = np.pad(signal, (shift, 0), mode='edge')[:-shift]
        elif shift < 0:
            # 左移
            augmented = np.pad(signal, (0, -shift), mode='edge')[-shift:]
        else:
            augmented = signal.copy()

        return augmented

    @staticmethod
    def time_stretch(signal: np.ndarray, stretch_range: Tuple[float, float] = (0.9, 1.1)) -> np.ndarray:
        """
        时间拉伸/压缩

        参数:
            signal: 输入信号
            stretch_range: 拉伸范围 (min_rate, max_rate)

        返回:
            augmented: 增强后的信号
        """
        rate = np.random.uniform(stretch_range[0], stretch_range[1])
        indices = np.arange(0, len(signal), rate)
        indices = np.clip(indices, 0, len(signal) - 1).astype(int)

        # 截断或填充到原始长度
        augmented = signal[indices]
        if len(augmented) < len(signal):
            # 填充
            augmented = np.pad(augmented, (0, len(signal) - len(augmented)), mode='edge')
        elif len(augmented) > len(signal):
            # 截断
            augmented = augmented[:len(signal)]

        return augmented

    @staticmethod
    def baseline_wander(signal: np.ndarray, amplitude: float = 0.1, freq: float = 0.5) -> np.ndarray:
        """
        添加基线漂移

        参数:
            signal: 输入信号
            amplitude: 漂移幅度
            freq: 漂移频率 (Hz，假设采样率为1000Hz)

        返回:
            augmented: 增强后的信号
        """
        t = np.arange(len(signal)) / 1000.0  # 假设采样率1000Hz
        wander = amplitude * np.sin(2 * np.pi * freq * t)
        return signal + wander

    @staticmethod
    def channel_dropout(signals: np.ndarray, dropout_prob: float = 0.1) -> np.ndarray:
        """
        通道dropout（用于多通道信号）

        参数:
            signals: 多通道信号 (n_samples, n_channels)
            dropout_prob: dropout概率

        返回:
            augmented: 增强后的信号
        """
        if signals.ndim == 1:
            return signals  # 单通道信号不做dropout

        augmented = signals.copy()
        n_channels = signals.shape[1]

        for ch in range(n_channels):
            if np.random.random() < dropout_prob:
                # 将该通道置零或替换为噪声
                augmented[:, ch] = np.random.normal(0, 0.01, len(signals))

        return augmented

    @staticmethod
    def random_crop(signal: np.ndarray, crop_size: int) -> np.ndarray:
        """
        随机裁剪

        参数:
            signal: 输入信号
            crop_size: 裁剪后的大小

        返回:
            cropped: 裁剪后的信号
        """
        if len(signal) <= crop_size:
            return signal

        start_idx = np.random.randint(0, len(signal) - crop_size)
        return signal[start_idx:start_idx + crop_size]

    @staticmethod
    def mixup(signal1: np.ndarray, signal2: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """
        Mixup数据增强（混合两个信号）

        参数:
            signal1: 第一个信号
            signal2: 第二个信号
            alpha: 混合系数范围

        返回:
            mixed: 混合后的信号
        """
        lam = np.random.beta(alpha, alpha)
        return lam * signal1 + (1 - lam) * signal2

    def augment_pipeline(self, signal: np.ndarray,
                        add_noise: bool = True,
                        scale_amplitude: bool = True,
                        shift_time: bool = False,
                        add_baseline: bool = False) -> np.ndarray:
        """
        数据增强流水线

        参数:
            signal: 输入信号
            add_noise: 是否添加噪声
            scale_amplitude: 是否缩放幅度
            shift_time: 是否时间平移
            add_baseline: 是否添加基线漂移

        返回:
            augmented: 增强后的信号
        """
        augmented = signal.copy()

        if add_noise:
            augmented = self.add_noise(augmented)

        if scale_amplitude:
            augmented = self.amplitude_scaling(augmented)

        if shift_time:
            augmented = self.time_shift(augmented)

        if add_baseline:
            augmented = self.baseline_wander(augmented)

        return augmented

    def augment_dataset(self, X: np.ndarray, y: np.ndarray,
                       n_augmentations: int = 1,
                       methods: Optional[list] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        对整个数据集进行增强

        参数:
            X: 特征矩阵 (n_samples, n_features)
            y: 标签向量 (n_samples,)
            n_augmentations: 每个样本生成的增强样本数
            methods: 使用的增强方法列表

        返回:
            X_aug: 增强后的特征矩阵
            y_aug: 增强后的标签向量
        """
        if methods is None:
            methods = ['noise', 'scale']

        X_list = [X]
        y_list = [y]

        for _ in range(n_augmentations):
            X_aug_batch = []

            for sample in X:
                # 应用随机增强方法
                augmented = sample.copy()

                if 'noise' in methods and np.random.random() > 0.5:
                    augmented = self.add_noise(augmented)

                if 'scale' in methods and np.random.random() > 0.5:
                    augmented = self.amplitude_scaling(augmented)

                if 'shift' in methods and np.random.random() > 0.5:
                    augmented = self.time_shift(augmented, max_shift=50)

                X_aug_batch.append(augmented)

            X_list.append(np.array(X_aug_batch))
            y_list.append(y)

        X_aug = np.vstack(X_list)
        y_aug = np.hstack(y_list)

        return X_aug, y_aug


def demo():
    """演示数据增强的效果"""
    import matplotlib.pyplot as plt

    # 配置中文字体
    try:
        import sys
        from pathlib import Path
        project_root = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(project_root))
        from code.utils.chinese_font_config import setup_chinese_font
        setup_chinese_font()
    except:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False

    print("EMG数据增强演示")
    print("="*60)

    # 生成模拟EMG信号
    fs = 1000
    t = np.linspace(0, 2, 2000)
    signal = np.random.normal(0, 0.1, len(t))

    # 添加肌肉激活
    activation_start = int(0.5 * fs)
    activation_end = int(1.5 * fs)
    for freq in range(60, 150, 20):
        signal[activation_start:activation_end] += 0.3 * np.sin(
            2 * np.pi * freq * t[activation_start:activation_end]
        )

    # 创建增强器
    augmentor = EMGDataAugmentation()

    # 应用不同的增强方法
    fig, axes = plt.subplots(4, 2, figsize=(14, 10))
    fig.suptitle('EMG数据增强方法演示', fontsize=16, fontweight='bold')

    # 原始信号
    axes[0, 0].plot(t, signal, linewidth=0.8, color='blue')
    axes[0, 0].set_title('原始信号', fontsize=12)
    axes[0, 0].set_ylabel('幅度 (mV)')
    axes[0, 0].grid(True, alpha=0.3)

    # 添加噪声
    noisy = augmentor.add_noise(signal, noise_level=0.1)
    axes[0, 1].plot(t, noisy, linewidth=0.8, color='orange')
    axes[0, 1].set_title('添加噪声', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)

    # 幅度缩放
    scaled = augmentor.amplitude_scaling(signal, scale_range=(0.7, 0.7))
    axes[1, 0].plot(t, scaled, linewidth=0.8, color='green')
    axes[1, 0].set_title('幅度缩放 (×0.7)', fontsize=12)
    axes[1, 0].set_ylabel('幅度 (mV)')
    axes[1, 0].grid(True, alpha=0.3)

    # 时间平移
    shifted = augmentor.time_shift(signal, max_shift=200)
    axes[1, 1].plot(t, shifted, linewidth=0.8, color='red')
    axes[1, 1].set_title('时间平移', fontsize=12)
    axes[1, 1].grid(True, alpha=0.3)

    # 时间拉伸
    stretched = augmentor.time_stretch(signal, stretch_range=(1.1, 1.1))
    axes[2, 0].plot(t, stretched, linewidth=0.8, color='purple')
    axes[2, 0].set_title('时间拉伸 (×1.1)', fontsize=12)
    axes[2, 0].set_ylabel('幅度 (mV)')
    axes[2, 0].grid(True, alpha=0.3)

    # 基线漂移
    wandered = augmentor.baseline_wander(signal, amplitude=0.2, freq=1.0)
    axes[2, 1].plot(t, wandered, linewidth=0.8, color='brown')
    axes[2, 1].set_title('基线漂移', fontsize=12)
    axes[2, 1].grid(True, alpha=0.3)

    # 组合增强
    combined = augmentor.augment_pipeline(signal, add_noise=True, scale_amplitude=True, add_baseline=True)
    axes[3, 0].plot(t, combined, linewidth=0.8, color='teal')
    axes[3, 0].set_title('组合增强 (噪声+缩放+漂移)', fontsize=12)
    axes[3, 0].set_xlabel('时间 (秒)')
    axes[3, 0].set_ylabel('幅度 (mV)')
    axes[3, 0].grid(True, alpha=0.3)

    # Mixup
    signal2 = np.random.normal(0, 0.05, len(t))
    mixed = augmentor.mixup(signal, signal2, alpha=0.5)
    axes[3, 1].plot(t, mixed, linewidth=0.8, color='magenta')
    axes[3, 1].set_title('Mixup混合', fontsize=12)
    axes[3, 1].set_xlabel('时间 (秒)')
    axes[3, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('code/week05_data_processing/augmentation_demo.png', dpi=150, bbox_inches='tight')
    print("\n✓ 图片已保存到: code/week05_data_processing/augmentation_demo.png")
    plt.show()

    print("\n演示完成！")
    print("\n数据增强的作用:")
    print("  1. 扩充训练数据，防止过拟合")
    print("  2. 提高模型对噪声和变化的鲁棒性")
    print("  3. 模拟真实环境中的信号变化")


if __name__ == '__main__':
    demo()
