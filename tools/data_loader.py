#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用EMG数据加载器

支持加载多种格式的EMG数据集：
- UCI EMG Gestures (CSV/TXT)
- Ninapro Database (MAT)
- GRABMyo (HDF5)
- 自定义样本数据 (CSV)

使用示例:
    from data_loader import EMGDataLoader

    # 加载UCI数据
    loader = EMGDataLoader(dataset_type='uci')
    emg, label = loader.load('path/to/uci_data.txt')

    # 加载Ninapro数据
    loader = EMGDataLoader(dataset_type='ninapro')
    emg, label, rep = loader.load('path/to/ninapro_S1_E1_A1.mat')

作者: EMG Learning Project
日期: 2026-01-29
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path
import warnings

# 尝试导入scipy（用于MATLAB文件）
try:
    import scipy.io as sio
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("scipy未安装，无法加载MATLAB格式数据")

# 尝试导入h5py（用于HDF5文件）
try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False
    warnings.warn("h5py未安装，无法加载HDF5格式数据")


class EMGDataLoader:
    """通用EMG数据加载器"""

    def __init__(self, dataset_type='auto'):
        """
        初始化

        参数:
            dataset_type: 数据集类型
                - 'auto': 自动识别（根据文件扩展名）
                - 'uci': UCI EMG Gestures
                - 'ninapro': Ninapro Database
                - 'grabmyo': GRABMyo
                - 'sample': 自定义样本数据
        """
        self.dataset_type = dataset_type
        self.supported_types = ['auto', 'uci', 'ninapro', 'grabmyo', 'sample']

        if dataset_type not in self.supported_types:
            raise ValueError(
                f"不支持的数据集类型: {dataset_type}. "
                f"支持的类型: {self.supported_types}"
            )

    def auto_detect_type(self, file_path):
        """
        根据文件扩展名自动检测数据集类型

        参数:
            file_path: 文件路径

        返回:
            dataset_type: 检测到的数据集类型
        """
        ext = Path(file_path).suffix.lower()

        if ext in ['.txt']:
            return 'uci'
        elif ext in ['.mat']:
            return 'ninapro'
        elif ext in ['.h5', '.hdf5']:
            return 'grabmyo'
        elif ext in ['.csv']:
            return 'sample'
        else:
            raise ValueError(f"无法识别的文件类型: {ext}")

    def load_uci(self, file_path):
        """
        加载UCI EMG Gestures数据

        文件格式: TAB分隔的TXT文件
        列: channel1 channel2 label(可选)

        参数:
            file_path: 数据文件路径

        返回:
            emg: EMG信号数组 (n_samples, n_channels)
            label: 标签数组 (n_samples,), 如果没有标签则返回None
        """
        try:
            # 尝试用TAB分隔读取
            data = pd.read_csv(file_path, sep='\t', header=None)
        except:
            # 如果失败，尝试自动识别分隔符
            data = pd.read_csv(file_path, sep=None, header=None, engine='python')

        # 前N-1列是EMG信号，最后一列可能是标签
        if data.shape[1] >= 3:
            # 假设最后一列是标签
            emg = data.iloc[:, :-1].values
            label = data.iloc[:, -1].values
        else:
            # 没有标签
            emg = data.values
            label = None

        print(f"✅ 加载UCI数据: {file_path}")
        print(f"   形状: {emg.shape}")
        print(f"   通道数: {emg.shape[1]}")
        print(f"   样本数: {emg.shape[0]}")
        if label is not None:
            print(f"   标签: {np.unique(label)}")

        return emg, label

    def load_ninapro(self, file_path):
        """
        加载Ninapro Database数据

        文件格式: MATLAB .mat文件
        变量:
            - emg: EMG信号 (n_samples, n_channels)
            - restimulus: 手势标签 (n_samples, 1)
            - rerepetition: 重复次数 (n_samples, 1)

        参数:
            file_path: 数据文件路径

        返回:
            emg: EMG信号数组 (n_samples, n_channels)
            label: 手势标签数组 (n_samples,)
            repetition: 重复次数数组 (n_samples,)
        """
        if not SCIPY_AVAILABLE:
            raise ImportError("需要安装scipy来加载MATLAB文件: pip install scipy")

        # 加载MATLAB文件
        data = sio.loadmat(file_path)

        # 提取EMG信号
        emg = data['emg']

        # 提取标签
        if 'restimulus' in data:
            label = data['restimulus'].flatten()
        elif 'stimulus' in data:
            label = data['stimulus'].flatten()
        else:
            label = None
            warnings.warn("未找到标签字段 (restimulus/stimulus)")

        # 提取重复次数
        if 'rerepetition' in data:
            repetition = data['rerepetition'].flatten()
        elif 'repetition' in data:
            repetition = data['repetition'].flatten()
        else:
            repetition = None
            warnings.warn("未找到重复次数字段 (rerepetition/repetition)")

        print(f"✅ 加载Ninapro数据: {file_path}")
        print(f"   EMG形状: {emg.shape}")
        print(f"   通道数: {emg.shape[1]}")
        print(f"   样本数: {emg.shape[0]}")
        if label is not None:
            print(f"   手势类型: {len(np.unique(label))} 种")
            print(f"   手势标签: {np.unique(label)}")
        if repetition is not None:
            print(f"   重复次数: {np.unique(repetition)}")

        return emg, label, repetition

    def load_grabmyo(self, file_path):
        """
        加载GRABMyo数据

        文件格式: HDF5 .h5文件

        参数:
            file_path: 数据文件路径

        返回:
            emg: EMG信号数组 (n_samples, n_channels)
            label: 标签数组 (n_samples,)
        """
        if not H5PY_AVAILABLE:
            raise ImportError("需要安装h5py来加载HDF5文件: pip install h5py")

        with h5py.File(file_path, 'r') as f:
            # 打印文件结构
            print(f"HDF5文件结构: {list(f.keys())}")

            # 尝试提取EMG和标签
            if 'emg' in f:
                emg = f['emg'][:]
            elif 'data' in f:
                emg = f['data'][:]
            else:
                raise KeyError("未找到EMG数据字段")

            if 'label' in f:
                label = f['label'][:]
            elif 'stimulus' in f:
                label = f['stimulus'][:]
            else:
                label = None
                warnings.warn("未找到标签字段")

        print(f"✅ 加载GRABMyo数据: {file_path}")
        print(f"   EMG形状: {emg.shape}")
        print(f"   通道数: {emg.shape[1] if emg.ndim > 1 else 1}")
        print(f"   样本数: {emg.shape[0]}")
        if label is not None:
            print(f"   标签类型: {len(np.unique(label))} 种")

        return emg, label

    def load_sample(self, file_path):
        """
        加载自定义样本数据 (CSV格式)

        文件格式: CSV文件
        列: time, channel_0, channel_1, ..., label

        参数:
            file_path: 数据文件路径

        返回:
            emg: EMG信号数组 (n_samples, n_channels)
            label: 标签数组 (n_samples,)
            time: 时间戳数组 (n_samples,)
        """
        data = pd.read_csv(file_path)

        # 提取时间
        if 'time' in data.columns:
            time = data['time'].values
        else:
            time = None

        # 提取标签
        if 'label' in data.columns:
            label = data['label'].values
        else:
            label = None

        # 提取EMG信号（所有以channel_开头的列）
        channel_cols = [col for col in data.columns if col.startswith('channel_')]

        if len(channel_cols) == 0:
            raise ValueError("未找到EMG通道列（应以'channel_'开头）")

        emg = data[channel_cols].values

        print(f"✅ 加载样本数据: {file_path}")
        print(f"   EMG形状: {emg.shape}")
        print(f"   通道数: {emg.shape[1]}")
        print(f"   样本数: {emg.shape[0]}")
        if label is not None:
            unique_labels = pd.Series(label).unique()
            print(f"   标签: {unique_labels}")
        if time is not None:
            print(f"   时长: {time[-1] - time[0]:.2f} 秒")

        return emg, label, time

    def load(self, file_path):
        """
        根据数据集类型加载数据

        参数:
            file_path: 数据文件路径

        返回:
            根据数据集类型返回不同的值
        """
        file_path = str(file_path)

        # 检查文件是否存在
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        # 自动检测类型
        if self.dataset_type == 'auto':
            detected_type = self.auto_detect_type(file_path)
            print(f"🔍 自动检测数据集类型: {detected_type}")
            return self._load_by_type(file_path, detected_type)
        else:
            return self._load_by_type(file_path, self.dataset_type)

    def _load_by_type(self, file_path, dataset_type):
        """根据类型加载数据"""
        if dataset_type == 'uci':
            return self.load_uci(file_path)
        elif dataset_type == 'ninapro':
            return self.load_ninapro(file_path)
        elif dataset_type == 'grabmyo':
            return self.load_grabmyo(file_path)
        elif dataset_type == 'sample':
            return self.load_sample(file_path)
        else:
            raise ValueError(f"不支持的数据集类型: {dataset_type}")

    def load_multiple(self, file_pattern):
        """
        加载多个文件（使用通配符）

        参数:
            file_pattern: 文件模式（支持通配符）
                例如: 'data/sample/subject_01/fist/*.csv'

        返回:
            all_emg: 所有EMG信号列表
            all_labels: 所有标签列表
            all_files: 文件路径列表
        """
        from glob import glob

        files = sorted(glob(file_pattern))

        if len(files) == 0:
            raise ValueError(f"未找到匹配的文件: {file_pattern}")

        print(f"📁 找到 {len(files)} 个文件")

        all_emg = []
        all_labels = []
        all_files = []

        for file in files:
            try:
                result = self.load(file)

                # 提取EMG和label（处理不同返回格式）
                if len(result) >= 2:
                    emg = result[0]
                    label = result[1]
                else:
                    emg = result[0]
                    label = None

                all_emg.append(emg)
                all_labels.append(label)
                all_files.append(file)

            except Exception as e:
                warnings.warn(f"加载文件失败 {file}: {e}")
                continue

        print(f"✅ 成功加载 {len(all_emg)} 个文件")

        return all_emg, all_labels, all_files


def demo():
    """演示如何使用数据加载器"""
    print("=" * 60)
    print("EMG数据加载器演示")
    print("=" * 60)

    # 创建加载器
    loader = EMGDataLoader(dataset_type='auto')

    # 示例：加载样本数据
    print("\n1. 尝试加载样本数据...")
    try:
        # 这里需要替换为实际的文件路径
        sample_file = '../data/sample/subject_01/fist/trial_001.csv'
        if os.path.exists(sample_file):
            emg, label, time = loader.load(sample_file)
            print(f"   样本数据形状: {emg.shape}")
        else:
            print(f"   ⚠️  样本文件不存在: {sample_file}")
            print("   请先运行 generate_sample_data.py 生成样本数据")
    except Exception as e:
        print(f"   ❌ 加载失败: {e}")

    # 示例：批量加载
    print("\n2. 演示批量加载...")
    try:
        pattern = '../data/sample/subject_01/fist/*.csv'
        all_emg, all_labels, all_files = loader.load_multiple(pattern)
        print(f"   加载了 {len(all_emg)} 个文件")
        if len(all_emg) > 0:
            print(f"   第一个文件形状: {all_emg[0].shape}")
    except Exception as e:
        print(f"   ℹ️  {e}")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    demo()
