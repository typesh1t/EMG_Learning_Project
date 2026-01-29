#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时EMG分类器

实时处理EMG信号并进行手势识别

作者: EMG Learning Project
日期: 2026-01-29
"""

import sys
from pathlib import Path
import numpy as np
import time
from collections import deque

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from code.week06_preprocessing.filters import EMGFilters
from code.week07_feature_extraction.features import EMGFeatures
from code.week08_pattern_recognition.classifier import EMGClassifier


class RealtimeClassifier:
    """实时EMG分类器"""

    def __init__(self,
                 model_path,
                 scaler_path,
                 window_size=200,
                 step_size=50,
                 n_channels=4,
                 fs=1000):
        """
        初始化实时分类器

        参数:
            model_path: 训练好的模型路径
            scaler_path: 特征归一化器路径
            window_size: 窗口大小（样本数）
            step_size: 步长（样本数）
            n_channels: 通道数
            fs: 采样率 (Hz)
        """
        self.window_size = window_size
        self.step_size = step_size
        self.n_channels = n_channels
        self.fs = fs

        # 加载模型
        self.classifier = EMGClassifier.load_model(model_path, scaler_path)
        print(f"✓ 模型已加载: {model_path}")
        print(f"  手势类别: {self.classifier.gesture_names}")

        # 初始化滤波器
        self.filters = EMGFilters(fs=fs)

        # 数据窗口缓冲区（每个通道一个）
        self.windows = [deque(maxlen=window_size) for _ in range(n_channels)]

        # 预测历史（用于平滑）
        self.prediction_history = deque(maxlen=5)

        # 统计信息
        self.prediction_count = 0
        self.last_prediction = None
        self.last_probabilities = None
        self.last_update_time = None

    def update(self, new_data):
        """
        更新数据并进行预测

        参数:
            new_data: 新数据点 (n_channels,) 或 (n_samples, n_channels)

        返回:
            prediction: 预测的手势类别（如果有新预测）
            confidence: 预测置信度
        """
        # 确保数据是2D的
        if new_data.ndim == 1:
            new_data = new_data.reshape(1, -1)

        # 将新数据添加到窗口
        for sample in new_data:
            for ch in range(self.n_channels):
                self.windows[ch].append(sample[ch])

        # 检查是否有足够的数据进行预测
        if len(self.windows[0]) < self.window_size:
            return None, None

        # 检查是否应该进行新的预测（根据步长）
        if self.prediction_count > 0:
            samples_since_last = len(self.windows[0]) - self.window_size
            if samples_since_last < self.step_size:
                return self.last_prediction, np.max(self.last_probabilities)

        # 提取窗口数据
        window_data = np.zeros((self.window_size, self.n_channels))
        for ch in range(self.n_channels):
            window_data[:, ch] = list(self.windows[ch])

        # 进行预测
        prediction, confidence = self._predict_window(window_data)

        self.prediction_count += 1
        self.last_update_time = time.time()

        return prediction, confidence

    def _predict_window(self, window_data):
        """
        对一个窗口的数据进行预测

        参数:
            window_data: (window_size, n_channels) 数组

        返回:
            prediction: 预测类别
            confidence: 置信度
        """
        # 1. 预处理每个通道
        filtered_data = np.zeros_like(window_data)
        for ch in range(self.n_channels):
            filtered_data[:, ch] = self.filters.preprocess_emg(
                window_data[:, ch],
                remove_powerline=True,
                powerline_freq=50
            )

        # 2. 提取特征
        features = self._extract_features(filtered_data)

        # 3. 预测
        predictions, probabilities = self.classifier.predict(features)

        # 4. 存储到历史
        self.last_prediction = predictions[0]
        self.last_probabilities = probabilities[0]
        self.prediction_history.append(predictions[0])

        # 5. 获取置信度
        confidence = np.max(probabilities[0])

        return predictions[0], confidence

    def _extract_features(self, window_data):
        """
        从窗口数据提取特征

        参数:
            window_data: (window_size, n_channels) 数组

        返回:
            features: (1, n_features) 特征向量
        """
        all_features = []

        for ch in range(self.n_channels):
            # 时域特征
            time_feat = EMGFeatures.extract_time_features(window_data[:, ch])
            # 频域特征
            freq_feat = EMGFeatures.extract_freq_features(window_data[:, ch], self.fs)

            # 合并特征
            all_features.extend(list(time_feat.values()))
            all_features.extend(list(freq_feat.values()))

        return np.array([all_features])

    def get_smoothed_prediction(self):
        """
        获取平滑后的预测（多数投票）

        返回:
            prediction: 平滑后的预测类别
        """
        if len(self.prediction_history) == 0:
            return None

        # 多数投票
        unique, counts = np.unique(list(self.prediction_history), return_counts=True)
        return unique[np.argmax(counts)]

    def get_gesture_name(self, class_idx):
        """获取手势名称"""
        if class_idx is None:
            return "未知"
        return self.classifier.gesture_names[class_idx]

    def reset(self):
        """重置分类器状态"""
        for window in self.windows:
            window.clear()
        self.prediction_history.clear()
        self.prediction_count = 0
        self.last_prediction = None
        self.last_probabilities = None
        print("✓ 分类器已重置")

    def get_statistics(self):
        """获取统计信息"""
        return {
            'prediction_count': self.prediction_count,
            'buffer_fill': len(self.windows[0]) / self.window_size * 100,
            'last_prediction': self.get_gesture_name(self.last_prediction),
            'last_confidence': np.max(self.last_probabilities) if self.last_probabilities is not None else 0,
            'smoothed_prediction': self.get_gesture_name(self.get_smoothed_prediction())
        }


def demo():
    """演示实时分类（需要预先训练的模型）"""
    print("="*60)
    print("实时EMG分类器演示")
    print("="*60)

    # 检查模型是否存在
    model_path = 'data/models/emg_classifier.pkl'
    scaler_path = 'data/models/emg_scaler.pkl'

    try:
        # 创建分类器
        rt_clf = RealtimeClassifier(
            model_path=model_path,
            scaler_path=scaler_path,
            window_size=200,
            step_size=50,
            n_channels=4,
            fs=1000
        )

        print("\n开始实时分类演示...")
        print("（使用模拟数据）\n")

        # 模拟实时数据流
        from realtime_acquisition import RealtimeEMGAcquisition

        # 创建采集器
        acq = RealtimeEMGAcquisition(
            source='simulator',
            n_channels=4,
            fs=1000,
            buffer_size=5000
        )

        # 定义回调函数
        prediction_count = [0]
        def on_new_data(data):
            # 更新分类器
            prediction, confidence = rt_clf.update(data.reshape(1, -1))

            if prediction is not None:
                prediction_count[0] += 1
                if prediction_count[0] % 10 == 0:
                    gesture = rt_clf.get_gesture_name(prediction)
                    smoothed = rt_clf.get_gesture_name(rt_clf.get_smoothed_prediction())
                    print(f"预测 {prediction_count[0]}: {gesture} (置信度: {confidence:.2%}), "
                          f"平滑后: {smoothed}")

        acq.register_callback(on_new_data)

        # 开始采集
        acq.start()

        try:
            # 运行10秒
            print("运行10秒...\n")
            time.sleep(10)

            # 显示统计信息
            stats = rt_clf.get_statistics()
            print("\n" + "-"*60)
            print("统计信息:")
            print(f"  总预测次数: {stats['prediction_count']}")
            print(f"  最后预测: {stats['last_prediction']} ({stats['last_confidence']:.2%})")
            print(f"  平滑预测: {stats['smoothed_prediction']}")

        except KeyboardInterrupt:
            print("\n用户中断")

        finally:
            acq.stop()

    except FileNotFoundError:
        print(f"\n✗ 错误: 未找到训练好的模型")
        print(f"  模型路径: {model_path}")
        print("\n请先训练模型:")
        print("  1. 生成样本数据: python tools/generate_sample_data.py")
        print("  2. 运行完整流程: python examples/complete_pipeline.py")
        print("  3. 模型会自动保存到 data/models/\n")

    print("\n演示完成！")


if __name__ == '__main__':
    demo()
