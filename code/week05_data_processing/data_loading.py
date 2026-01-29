#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EMG数据加载器

提供便捷的数据加载功能，支持多种数据格式和批量加载。

作者: EMG Learning Project
日期: 2026-01-29
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import warnings


class EMGDataLoader:
    """EMG数据加载器类"""

    def __init__(self, data_dir: str, fs: int = 1000):
        """
        初始化数据加载器

        参数:
            data_dir: 数据目录路径
            fs: 采样率 (Hz)
        """
        self.data_dir = Path(data_dir)
        self.fs = fs

        if not self.data_dir.exists():
            raise FileNotFoundError(f"数据目录不存在: {self.data_dir}")

    def load_single_trial(self, file_path: str) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        加载单个试验数据

        参数:
            file_path: CSV文件路径

        返回:
            time: 时间序列
            signals: 信号数据 (n_samples, n_channels)
            label: 手势标签
        """
        df = pd.read_csv(file_path)

        # 提取时间
        time = df['time'].values

        # 提取信号通道
        signal_columns = [col for col in df.columns if col.startswith('channel_')]
        signals = df[signal_columns].values

        # 提取标签
        label = df['label'].iloc[0] if 'label' in df.columns else 'unknown'

        return time, signals, label

    def load_gesture_trials(self, subject_id: str, gesture: str) -> List[Tuple]:
        """
        加载某个受试者的某种手势的所有试验

        参数:
            subject_id: 受试者ID (如 'subject_01')
            gesture: 手势类型 (如 'fist', 'open', 'rest')

        返回:
            trials: [(time, signals, label), ...] 的列表
        """
        gesture_dir = self.data_dir / subject_id / gesture

        if not gesture_dir.exists():
            warnings.warn(f"目录不存在: {gesture_dir}")
            return []

        trials = []
        for csv_file in sorted(gesture_dir.glob('*.csv')):
            time, signals, label = self.load_single_trial(csv_file)
            trials.append((time, signals, label))

        return trials

    def load_subject_data(self, subject_id: str, gestures: Optional[List[str]] = None) -> Dict:
        """
        加载某个受试者的所有数据

        参数:
            subject_id: 受试者ID
            gestures: 手势列表，None则加载所有手势

        返回:
            data: {gesture: [(time, signals, label), ...], ...}
        """
        subject_dir = self.data_dir / subject_id

        if not subject_dir.exists():
            raise FileNotFoundError(f"受试者目录不存在: {subject_dir}")

        # 如果未指定手势，则自动检测
        if gestures is None:
            gestures = [d.name for d in subject_dir.iterdir() if d.is_dir()]

        data = {}
        for gesture in gestures:
            trials = self.load_gesture_trials(subject_id, gesture)
            if trials:
                data[gesture] = trials

        return data

    def load_all_subjects(self, gestures: Optional[List[str]] = None) -> Dict:
        """
        加载所有受试者的数据

        参数:
            gestures: 手势列表

        返回:
            all_data: {subject_id: {gesture: [trials], ...}, ...}
        """
        all_data = {}

        # 查找所有受试者目录
        subject_dirs = sorted([d for d in self.data_dir.iterdir()
                              if d.is_dir() and d.name.startswith('subject_')])

        for subject_dir in subject_dirs:
            subject_id = subject_dir.name
            print(f"加载 {subject_id}...")

            try:
                subject_data = self.load_subject_data(subject_id, gestures)
                all_data[subject_id] = subject_data
            except Exception as e:
                warnings.warn(f"加载 {subject_id} 失败: {e}")

        return all_data

    def prepare_ml_dataset(self, subjects: Optional[List[str]] = None,
                          gestures: Optional[List[str]] = None,
                          flatten_channels: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        准备用于机器学习的数据集

        参数:
            subjects: 受试者列表，None则加载所有
            gestures: 手势列表，None则加载所有
            flatten_channels: 是否展平通道维度

        返回:
            X: 特征数组 (n_trials, n_features)
            y: 标签数组 (n_trials,)
            gesture_names: 手势名称列表
        """
        X_list = []
        y_list = []
        gesture_names = []

        # 确定要加载的受试者
        if subjects is None:
            subject_dirs = sorted([d for d in self.data_dir.iterdir()
                                  if d.is_dir() and d.name.startswith('subject_')])
            subjects = [d.name for d in subject_dirs]

        # 加载数据
        for subject_id in subjects:
            subject_data = self.load_subject_data(subject_id, gestures)

            for gesture, trials in subject_data.items():
                if gesture not in gesture_names:
                    gesture_names.append(gesture)

                gesture_idx = gesture_names.index(gesture)

                for time, signals, label in trials:
                    if flatten_channels:
                        # 展平所有通道: (n_samples, n_channels) -> (n_samples * n_channels,)
                        features = signals.flatten()
                    else:
                        # 保持通道分离: (n_samples, n_channels)
                        features = signals

                    X_list.append(features)
                    y_list.append(gesture_idx)

        X = np.array(X_list)
        y = np.array(y_list)

        return X, y, gesture_names

    def get_dataset_info(self) -> Dict:
        """
        获取数据集的统计信息

        返回:
            info: 数据集信息字典
        """
        info = {
            'data_dir': str(self.data_dir),
            'n_subjects': 0,
            'gestures': set(),
            'n_trials_per_gesture': {},
            'total_trials': 0
        }

        subject_dirs = [d for d in self.data_dir.iterdir()
                       if d.is_dir() and d.name.startswith('subject_')]

        info['n_subjects'] = len(subject_dirs)

        for subject_dir in subject_dirs:
            gesture_dirs = [d for d in subject_dir.iterdir() if d.is_dir()]

            for gesture_dir in gesture_dirs:
                gesture = gesture_dir.name
                info['gestures'].add(gesture)

                n_trials = len(list(gesture_dir.glob('*.csv')))

                if gesture not in info['n_trials_per_gesture']:
                    info['n_trials_per_gesture'][gesture] = []

                info['n_trials_per_gesture'][gesture].append(n_trials)
                info['total_trials'] += n_trials

        # 转换为列表
        info['gestures'] = sorted(list(info['gestures']))

        # 计算每种手势的平均试验数
        for gesture in info['gestures']:
            trials = info['n_trials_per_gesture'][gesture]
            info['n_trials_per_gesture'][gesture] = {
                'mean': np.mean(trials),
                'min': np.min(trials),
                'max': np.max(trials)
            }

        return info

    def print_dataset_info(self):
        """打印数据集信息"""
        info = self.get_dataset_info()

        print("\n" + "="*60)
        print("数据集信息".center(60))
        print("="*60)

        print(f"\n数据目录: {info['data_dir']}")
        print(f"受试者数: {info['n_subjects']}")
        print(f"手势类型: {', '.join(info['gestures'])}")
        print(f"总试验数: {info['total_trials']}")

        print("\n每种手势的试验数统计:")
        for gesture, stats in info['n_trials_per_gesture'].items():
            print(f"  {gesture}: 平均 {stats['mean']:.1f}, "
                  f"范围 [{stats['min']} - {stats['max']}]")

        print("\n" + "="*60 + "\n")


def demo():
    """演示数据加载器的使用"""
    print("EMG数据加载器演示")
    print("="*60)

    # 创建数据加载器
    try:
        loader = EMGDataLoader(data_dir='../../data/sample/', fs=1000)
        print("✓ 数据加载器初始化成功")
    except FileNotFoundError as e:
        print(f"✗ 错误: {e}")
        print("\n请先运行 tools/generate_sample_data.py 生成样本数据")
        return

    # 打印数据集信息
    loader.print_dataset_info()

    # 加载单个试验
    print("\n【示例1: 加载单个试验】")
    try:
        time, signals, label = loader.load_single_trial(
            '../../data/sample/subject_01/fist/trial_001.csv'
        )
        print(f"  时间序列形状: {time.shape}")
        print(f"  信号形状: {signals.shape}")
        print(f"  标签: {label}")
    except Exception as e:
        print(f"  无法加载: {e}")

    # 加载某个受试者的某种手势
    print("\n【示例2: 加载某个受试者的某种手势】")
    trials = loader.load_gesture_trials('subject_01', 'fist')
    print(f"  加载了 {len(trials)} 个试验")

    # 加载所有数据并准备ML数据集
    print("\n【示例3: 准备机器学习数据集】")
    X, y, gesture_names = loader.prepare_ml_dataset()
    print(f"  特征矩阵 X: {X.shape}")
    print(f"  标签向量 y: {y.shape}")
    print(f"  手势类别: {gesture_names}")

    # 显示标签分布
    print("\n  标签分布:")
    for i, gesture in enumerate(gesture_names):
        count = np.sum(y == i)
        print(f"    {gesture}: {count} 个样本")

    print("\n" + "="*60)
    print("演示完成！")


if __name__ == '__main__':
    demo()
