#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EMG信号处理教学演示脚本

完整流程演示：
1. 原始信号 → 消噪/滤波
2. 特征提取（18种经典特征）
3. 结果可视化
4. 不同群体差异分析
5. 多种分类器对比

作者: EMG Learning Project
日期: 2026-01-30
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from code.week06_preprocessing.filters import EMGFilters
from code.week07_feature_extraction.features import EMGFeatures
from code.week08_pattern_recognition.classifier import EMGClassifier

# 配置中文显示
try:
    from code.utils.chinese_font_config import setup_chinese_font
    setup_chinese_font()
except:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False


def generate_patient_signal(patient_type='normal', duration=5, fs=1000):
    """
    生成不同类型患者的模拟EMG信号

    参数:
        patient_type: 'normal', 'mild_disorder', 'severe_disorder'
        duration: 信号时长（秒）
        fs: 采样率
    """
    t = np.linspace(0, duration, int(duration * fs))
    signal = np.zeros(len(t))

    # 静息期噪声
    signal += np.random.normal(0, 0.02, len(t))

    # 收缩期（1-4秒）
    start = int(1 * fs)
    end = int(4 * fs)

    if patient_type == 'normal':
        # 正常人：强有力的收缩，频率集中在70-140 Hz
        for freq in range(70, 140, 15):
            signal[start:end] += 0.4 * np.sin(2 * np.pi * freq * t[start:end])
        signal[start:end] += np.random.normal(0, 0.15, end - start)

    elif patient_type == 'mild_disorder':
        # 轻度障碍：收缩较弱，频率略低
        for freq in range(60, 120, 15):
            signal[start:end] += 0.25 * np.sin(2 * np.pi * freq * t[start:end])
        signal[start:end] += np.random.normal(0, 0.12, end - start)

    elif patient_type == 'severe_disorder':
        # 重度障碍：收缩很弱，频率更低，更多噪声
        for freq in range(50, 100, 20):
            signal[start:end] += 0.15 * np.sin(2 * np.pi * freq * t[start:end])
        signal[start:end] += np.random.normal(0, 0.1, end - start)

    # 添加工频干扰
    signal += 0.05 * np.sin(2 * np.pi * 50 * t)

    return t, signal


def demo_step1_denoising():
    """步骤1：展示消噪过程"""
    print("\n" + "="*70)
    print("步骤1：信号消噪/滤波".center(70))
    print("="*70)

    # 生成带噪声的信号
    t, signal_noisy = generate_patient_signal('normal')
    fs = 1000

    # 创建滤波器
    filters = EMGFilters(fs=fs)

    # 应用预处理
    signal_clean = filters.preprocess_emg(
        signal_noisy,
        remove_powerline=True,
        powerline_freq=50
    )

    # 可视化对比
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # 原始信号
    axes[0].plot(t, signal_noisy, linewidth=0.5, color='red', alpha=0.7)
    axes[0].set_ylabel('幅度 (mV)', fontsize=12)
    axes[0].set_title('原始信号（含噪声和50Hz工频干扰）', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(-2, 2)

    # 滤波后
    axes[1].plot(t, signal_clean, linewidth=0.5, color='green')
    axes[1].set_ylabel('幅度 (mV)', fontsize=12)
    axes[1].set_title('滤波后信号（带通20-500Hz + 陷波50Hz）', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(-2, 2)

    # 噪声成分
    noise = signal_noisy - signal_clean
    axes[2].plot(t, noise, linewidth=0.5, color='orange', alpha=0.7)
    axes[2].set_xlabel('时间 (秒)', fontsize=12)
    axes[2].set_ylabel('幅度 (mV)', fontsize=12)
    axes[2].set_title('被去除的噪声成分', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('teaching_demo_1_denoising.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"✓ 消噪完成")
    print(f"  原始信号标准差: {np.std(signal_noisy):.4f} mV")
    print(f"  滤波后标准差: {np.std(signal_clean):.4f} mV")
    print(f"  噪声标准差: {np.std(noise):.4f} mV")

    return signal_clean, fs


def demo_step2_feature_extraction(signal, fs):
    """步骤2：特征提取并可视化"""
    print("\n" + "="*70)
    print("步骤2：特征提取（18种经典特征）".center(70))
    print("="*70)

    # 提取时域特征
    time_features = EMGFeatures.extract_time_features(signal)
    print("\n时域特征（10种）：")
    for name, value in time_features.items():
        print(f"  {name:10s}: {value:10.6f}")

    # 提取频域特征
    freq_features = EMGFeatures.extract_freq_features(signal, fs=fs)
    print("\n频域特征（8种）：")
    for name, value in freq_features.items():
        print(f"  {name:15s}: {value:10.6f}")

    # 可视化所有特征
    all_features = {**time_features, **freq_features}

    fig, axes = plt.subplots(3, 6, figsize=(18, 12))
    fig.suptitle('18种EMG经典特征', fontsize=16, fontweight='bold')

    for idx, (name, value) in enumerate(all_features.items()):
        row = idx // 6
        col = idx % 6

        axes[row, col].bar([name], [value], color=f'C{idx % 10}')
        axes[row, col].set_title(name, fontsize=10, fontweight='bold')
        axes[row, col].set_ylabel('数值', fontsize=9)
        axes[row, col].tick_params(axis='x', labelsize=8, rotation=45)
        axes[row, col].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('teaching_demo_2_features.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\n✓ 特征提取完成，共{len(all_features)}种特征")

    return all_features


def demo_step3_group_comparison():
    """步骤3：不同群体的差异分析"""
    print("\n" + "="*70)
    print("步骤3：不同群体差异分析".center(70))
    print("="*70)

    fs = 1000
    filters = EMGFilters(fs=fs)

    # 生成三组数据：正常人、轻度障碍、重度障碍
    groups = {
        '正常人': 'normal',
        '轻度障碍': 'mild_disorder',
        '重度障碍': 'severe_disorder'
    }

    n_samples = 20  # 每组20个样本

    all_data = []

    for group_name, patient_type in groups.items():
        print(f"\n生成 {group_name} 的数据...")

        for i in range(n_samples):
            # 生成信号
            t, signal = generate_patient_signal(patient_type)

            # 预处理
            signal_clean = filters.preprocess_emg(signal, remove_powerline=True)

            # 提取特征
            time_feat = EMGFeatures.extract_time_features(signal_clean)
            freq_feat = EMGFeatures.extract_freq_features(signal_clean, fs=fs)

            # 合并特征
            features = {**time_feat, **freq_feat}
            features['group'] = group_name
            features['patient_type'] = patient_type

            all_data.append(features)

    # 转换为DataFrame
    df = pd.DataFrame(all_data)

    print(f"\n✓ 数据生成完成：{len(df)}个样本")
    print(f"  每组样本数: {n_samples}")

    # 选择关键特征进行对比
    key_features = ['MAV', 'RMS', 'VAR', 'WL', 'MNF', 'MDF']

    # 箱线图对比
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('不同群体的特征对比（箱线图）', fontsize=16, fontweight='bold')

    for idx, feat in enumerate(key_features):
        row = idx // 3
        col = idx % 3

        df.boxplot(column=feat, by='group', ax=axes[row, col])
        axes[row, col].set_title(feat, fontsize=12, fontweight='bold')
        axes[row, col].set_xlabel('群体', fontsize=10)
        axes[row, col].set_ylabel('数值', fontsize=10)
        axes[row, col].get_figure().suptitle('')  # 移除自动标题

    plt.tight_layout()
    plt.savefig('teaching_demo_3_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

    # 统计检验（ANOVA）
    print("\n统计检验结果（ANOVA）：")
    print("-" * 60)

    for feat in key_features:
        groups_data = [df[df['group'] == g][feat].values for g in groups.keys()]
        f_stat, p_value = stats.f_oneway(*groups_data)

        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
        print(f"{feat:10s}: F={f_stat:8.3f}, p={p_value:.6f} {significance}")

    print("\n说明: *** p<0.001, ** p<0.01, * p<0.05, ns 不显著")

    # 热图展示均值差异
    fig, ax = plt.subplots(figsize=(10, 6))

    pivot_table = df.groupby('group')[key_features].mean()
    sns.heatmap(pivot_table.T, annot=True, fmt='.4f', cmap='YlOrRd', ax=ax)
    ax.set_title('不同群体的特征均值热图', fontsize=14, fontweight='bold')
    ax.set_xlabel('群体', fontsize=12)
    ax.set_ylabel('特征', fontsize=12)

    plt.tight_layout()
    plt.savefig('teaching_demo_3_heatmap.png', dpi=150, bbox_inches='tight')
    plt.show()

    return df


def demo_step4_classifiers(df):
    """步骤4：多种分类器对比"""
    print("\n" + "="*70)
    print("步骤4：分类器对比（SVM, KNN, Random Forest）".center(70))
    print("="*70)

    # 准备数据
    feature_cols = [col for col in df.columns if col not in ['group', 'patient_type']]
    X = df[feature_cols].values
    y = df['group'].values

    # 转换标签为数字
    label_map = {'正常人': 0, '轻度障碍': 1, '重度障碍': 2}
    y_numeric = np.array([label_map[label] for label in y])

    classifiers_config = [
        ('random_forest', {'n_estimators': 100}, 'Random Forest'),
        ('svm', {'kernel': 'rbf', 'C': 1.0}, 'SVM (RBF核)'),
        ('knn', {'n_neighbors': 5}, 'KNN (k=5)')
    ]

    results = []

    for clf_type, params, name in classifiers_config:
        print(f"\n训练 {name}...")

        # 创建分类器
        clf = EMGClassifier(classifier_type=clf_type, **params)

        # 划分数据
        X_train, X_test, y_train, y_test = clf.prepare_data(
            X, y_numeric, test_size=0.3, random_state=42
        )

        # 训练
        clf.train(X_train, y_train,
                 feature_names=feature_cols,
                 gesture_names=list(label_map.keys()))

        # 评估
        accuracy = clf.evaluate(X_test, y_test)

        print(f"  测试集准确率: {accuracy:.2%}")

        results.append({
            'classifier': name,
            'accuracy': accuracy
        })

    # 可视化对比
    results_df = pd.DataFrame(results)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(results_df['classifier'], results_df['accuracy'] * 100,
                  color=['#2ecc71', '#3498db', '#e74c3c'])

    ax.set_ylabel('准确率 (%)', fontsize=12)
    ax.set_title('不同分类器性能对比', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')

    # 在柱子上标注数值
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}%',
               ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('teaching_demo_4_classifiers.png', dpi=150, bbox_inches='tight')
    plt.show()

    # 详细报告
    print("\n" + "="*70)
    print("分类器性能总结：")
    print("-" * 70)
    for result in results:
        print(f"  {result['classifier']:20s}: {result['accuracy']:.2%}")

    return results


def main():
    """主函数：运行完整演示"""
    print("\n" + "="*70)
    print("EMG信号处理完整教学演示".center(70))
    print("="*70)
    print("\n本演示包含以下内容：")
    print("  1. 信号消噪/滤波")
    print("  2. 特征提取（18种经典方法）")
    print("  3. 不同群体差异分析")
    print("  4. 多种分类器对比")
    print("\n开始演示...\n")

    # 步骤1：消噪
    signal_clean, fs = demo_step1_denoising()

    input("\n按Enter继续到步骤2...")

    # 步骤2：特征提取
    features = demo_step2_feature_extraction(signal_clean, fs)

    input("\n按Enter继续到步骤3...")

    # 步骤3：群体差异分析
    df = demo_step3_group_comparison()

    input("\n按Enter继续到步骤4...")

    # 步骤4：分类器对比
    results = demo_step4_classifiers(df)

    print("\n" + "="*70)
    print("演示完成！".center(70))
    print("="*70)
    print("\n生成的图片：")
    print("  - teaching_demo_1_denoising.png")
    print("  - teaching_demo_2_features.png")
    print("  - teaching_demo_3_comparison.png")
    print("  - teaching_demo_3_heatmap.png")
    print("  - teaching_demo_4_classifiers.png")
    print("\n这些图片可以直接用于教学PPT！\n")


if __name__ == '__main__':
    main()
