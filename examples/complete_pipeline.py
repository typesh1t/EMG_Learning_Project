#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EMG信号处理完整流程示例

这是一个端到端的示例，展示如何：
1. 加载EMG数据
2. 信号预处理（滤波）
3. 特征提取
4. 训练分类器
5. 评估性能

运行前确保已生成样本数据：
    python tools/generate_sample_data.py --output data/sample/ --subjects 3 --trials 5

作者: EMG Learning Project
日期: 2026-01-29
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 配置中文字体
try:
    from code.utils.chinese_font_config import setup_chinese_font
    setup_chinese_font()
except:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False

# 导入我们的模块
from code.week05_data_processing.data_loading import EMGDataLoader
from code.week06_preprocessing.filters import EMGFilters
from code.week07_feature_extraction.features import EMGFeatures
from code.week08_pattern_recognition.classifier import EMGClassifier


def main():
    """主函数：完整的EMG处理流程"""

    print("\n" + "="*70)
    print("EMG信号处理完整流程示例".center(70))
    print("="*70)

    # ========================================================================
    # 步骤1: 加载数据
    # ========================================================================
    print("\n【步骤1/5】加载EMG数据...")
    print("-" * 70)

    try:
        loader = EMGDataLoader(data_dir='data/sample/', fs=1000)
        loader.print_dataset_info()

        # 准备机器学习数据集
        X_raw, y, gesture_names = loader.prepare_ml_dataset()
        print(f"\n✓ 数据加载成功！")
        print(f"  原始数据形状: {X_raw.shape}")
        print(f"  标签数量: {len(y)}")
        print(f"  手势类别: {gesture_names}")

    except FileNotFoundError:
        print("\n✗ 错误: 未找到样本数据！")
        print("\n请先运行以下命令生成样本数据：")
        print("  python tools/generate_sample_data.py --output data/sample/ --subjects 3 --trials 5\n")
        return

    # ========================================================================
    # 步骤2: 信号预处理（滤波）
    # ========================================================================
    print("\n【步骤2/5】信号预处理（滤波）...")
    print("-" * 70)

    filters = EMGFilters(fs=1000)

    # 对每个样本进行预处理
    X_filtered = []
    for i, signal in enumerate(X_raw):
        # 将展平的信号重塑为多通道格式（假设4通道）
        n_channels = 4
        signal_length = len(signal) // n_channels
        signal_multichannel = signal.reshape(n_channels, signal_length)

        # 对每个通道进行滤波
        filtered_channels = []
        for ch in range(n_channels):
            filtered = filters.preprocess_emg(
                signal_multichannel[ch],
                remove_powerline=True,
                powerline_freq=50
            )
            filtered_channels.append(filtered)

        # 重新展平
        X_filtered.append(np.array(filtered_channels).flatten())

        if (i + 1) % 10 == 0:
            print(f"  已处理 {i+1}/{len(X_raw)} 个样本...")

    X_filtered = np.array(X_filtered)
    print(f"\n✓ 预处理完成！")
    print(f"  滤波后数据形状: {X_filtered.shape}")

    # ========================================================================
    # 步骤3: 特征提取
    # ========================================================================
    print("\n【步骤3/5】特征提取...")
    print("-" * 70)

    # 从每个样本提取特征
    X_features = []
    feature_names = None

    for i, signal in enumerate(X_filtered):
        # 重塑为多通道
        signal_multichannel = signal.reshape(n_channels, signal_length)

        # 对每个通道提取特征
        all_features = []
        for ch in range(n_channels):
            # 提取时域特征
            time_features = EMGFeatures.extract_time_features(signal_multichannel[ch])
            # 提取频域特征
            freq_features = EMGFeatures.extract_freq_features(signal_multichannel[ch], fs=1000)

            # 合并特征
            combined = np.concatenate([
                list(time_features.values()),
                list(freq_features.values())
            ])
            all_features.extend(combined)

        X_features.append(all_features)

        # 保存特征名称（只需要一次）
        if feature_names is None:
            feature_names = []
            for ch in range(n_channels):
                for name in time_features.keys():
                    feature_names.append(f'ch{ch}_{name}')
                for name in freq_features.keys():
                    feature_names.append(f'ch{ch}_{name}')

        if (i + 1) % 10 == 0:
            print(f"  已提取 {i+1}/{len(X_filtered)} 个样本的特征...")

    X_features = np.array(X_features)
    print(f"\n✓ 特征提取完成！")
    print(f"  特征矩阵形状: {X_features.shape}")
    print(f"  每个样本的特征数: {X_features.shape[1]}")

    # ========================================================================
    # 步骤4: 训练分类器
    # ========================================================================
    print("\n【步骤4/5】训练分类器...")
    print("-" * 70)

    # 创建分类器
    clf = EMGClassifier(classifier_type='random_forest', n_estimators=100)

    # 准备训练和测试数据
    X_train, X_test, y_train, y_test = clf.prepare_data(
        X_features, y, test_size=0.2, random_state=42
    )

    print(f"  训练集大小: {len(X_train)}")
    print(f"  测试集大小: {len(X_test)}")

    # 训练模型
    print("\n  开始训练...")
    clf.train(X_train, y_train, feature_names=feature_names, gesture_names=gesture_names)
    print("  ✓ 训练完成！")

    # ========================================================================
    # 步骤5: 评估性能
    # ========================================================================
    print("\n【步骤5/5】评估模型性能...")
    print("-" * 70)

    # 在测试集上评估
    accuracy = clf.evaluate(X_test, y_test)

    print(f"\n✓ 模型评估完成！")
    print(f"  测试集准确率: {accuracy:.2%}")

    # 绘制混淆矩阵
    print("\n  正在生成混淆矩阵...")
    clf.plot_confusion_matrix(X_test, y_test, save_path='examples/confusion_matrix.png')
    print("  ✓ 混淆矩阵已保存到: examples/confusion_matrix.png")

    # 绘制特征重要性（仅对Random Forest）
    if clf.classifier_type == 'random_forest':
        print("\n  正在生成特征重要性图...")
        clf.plot_feature_importance(top_n=20, save_path='examples/feature_importance.png')
        print("  ✓ 特征重要性图已保存到: examples/feature_importance.png")

    # ========================================================================
    # 步骤6: 保存模型
    # ========================================================================
    print("\n【额外步骤】保存训练好的模型...")
    print("-" * 70)

    clf.save_model('data/models/emg_classifier.pkl', 'data/models/emg_scaler.pkl')
    print("  ✓ 模型已保存到: data/models/")

    # ========================================================================
    # 完成
    # ========================================================================
    print("\n" + "="*70)
    print("🎉 完整流程演示完成！".center(70))
    print("="*70)

    print("\n总结:")
    print(f"  1. 加载了 {len(y)} 个EMG样本")
    print(f"  2. 对信号进行了滤波预处理")
    print(f"  3. 提取了 {X_features.shape[1]} 个特征")
    print(f"  4. 训练了 {clf.classifier_type} 分类器")
    print(f"  5. 达到了 {accuracy:.2%} 的测试准确率")

    print("\n下一步:")
    print("  - 查看生成的可视化图表")
    print("  - 尝试不同的分类器（SVM、KNN）")
    print("  - 调整预处理和特征提取参数")
    print("  - 使用真实的EMG数据进行实验")

    print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    main()
