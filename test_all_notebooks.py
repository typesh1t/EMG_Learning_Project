#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试所有notebooks的核心功能
"""

import sys
from pathlib import Path
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("=" * 70)
print("开始测试所有Notebook功能".center(70))
print("=" * 70)

# ========== 测试导入 ==========
print("\n[1/5] 测试模块导入...")
try:
    # 方法3: 直接添加模块路径
    sys.path.insert(0, str(project_root / 'code' / 'week06_preprocessing'))
    sys.path.insert(0, str(project_root / 'code' / 'week07_feature_extraction'))
    sys.path.insert(0, str(project_root / 'code' / 'week08_pattern_recognition'))

    from filters import EMGFilters
    from features import EMGFeatures
    from classifier import EMGClassifier

    print("  [OK] 模块导入成功")
except Exception as e:
    print(f"  [FAIL] 模块导入失败: {e}")
    sys.exit(1)

# ========== 测试滤波器（修复后） ==========
print("\n[2/5] 测试滤波器功能...")
try:
    fs = 1000
    t = np.linspace(0, 2, 2000)
    signal = np.random.normal(0, 0.1, len(t))

    # 添加60-120Hz成分
    for freq in range(60, 120, 20):
        signal += 0.3 * np.sin(2 * np.pi * freq * t)

    filters = EMGFilters(fs=fs)

    # 测试带通滤波（修复后应该能工作）
    signal_bp = filters.bandpass_filter(signal, lowcut=20, highcut=500)
    assert len(signal_bp) == len(signal), "滤波后长度不匹配"

    # 测试陷波滤波
    signal_notch = filters.notch_filter(signal_bp, freq=50)
    assert len(signal_notch) == len(signal), "陷波滤波后长度不匹配"

    # 测试完整预处理
    signal_processed = filters.preprocess_emg(signal, remove_powerline=True)
    assert len(signal_processed) == len(signal), "预处理后长度不匹配"

    print("  [OK] 滤波器功能正常")
    print(f"    - 带通滤波: OK")
    print(f"    - 陷波滤波: OK")
    print(f"    - 完整预处理: OK")
except Exception as e:
    print(f"  [FAIL] 滤波器测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ========== 测试特征提取 ==========
print("\n[3/5] 测试特征提取...")
try:
    # 生成测试信号
    signal_clean = signal_processed[:1000]

    # 时域特征
    time_features = EMGFeatures.extract_time_features(signal_clean)
    assert len(time_features) == 10, f"时域特征数量错误: {len(time_features)}"

    # 频域特征
    freq_features = EMGFeatures.extract_freq_features(signal_clean, fs=1000)
    assert len(freq_features) == 8, f"频域特征数量错误: {len(freq_features)}"

    # 滑动窗口特征
    features, feature_names, time_points = EMGFeatures.sliding_window_features(
        signal_clean, window_size=100, step=50, fs=1000
    )
    assert features.shape[1] == 18, f"滑动窗口特征数量错误: {features.shape[1]}"

    print("  [OK] 特征提取功能正常")
    print(f"    - 时域特征: {len(time_features)}个")
    print(f"    - 频域特征: {len(freq_features)}个")
    print(f"    - 滑动窗口: {features.shape[0]}个窗口")
except Exception as e:
    print(f"  [FAIL] 特征提取测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ========== 测试分类器 ==========
print("\n[4/5] 测试分类器...")
try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # 生成测试数据
    n_samples = 30
    n_features = 18
    X = np.random.randn(n_samples, n_features)
    y = np.array([0] * 10 + [1] * 10 + [2] * 10)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # 测试Random Forest
    clf = EMGClassifier(classifier_type='random_forest', n_estimators=10)
    clf.prepare_data(X, y, test_size=0.3, random_state=42)
    clf.train(X_train, y_train, feature_names=[f'feat_{i}' for i in range(n_features)])
    acc = clf.evaluate(X_test, y_test)

    assert np.isfinite(acc) and 0.0 <= acc <= 1.0, "准确率应为[0,1]之间的有限值"

    print("  [OK] 分类器功能正常")
    print(f"    - Random Forest训练: OK")
    print(f"    - 测试准确率: {acc:.2%}")
except Exception as e:
    print(f"  [FAIL] 分类器测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ========== 测试完整流程 ==========
print("\n[5/5] 测试完整流程...")
try:
    # 模拟三种手势
    def generate_gesture(gesture_type, n_samples=10):
        samples = []
        for _ in range(n_samples):
            t = np.linspace(0, 1, 1000)
            signal = np.random.normal(0, 0.02, len(t))

            if gesture_type == 'fist':
                for freq in range(70, 130, 15):
                    signal += 0.4 * np.sin(2 * np.pi * freq * t)
            elif gesture_type == 'open':
                for freq in range(60, 110, 15):
                    signal += 0.2 * np.sin(2 * np.pi * freq * t)

            # 预处理
            signal_clean = filters.preprocess_emg(signal, remove_powerline=True)

            # 特征提取
            time_feat = EMGFeatures.extract_time_features(signal_clean)
            freq_feat = EMGFeatures.extract_freq_features(signal_clean, fs=1000)

            sample = list(time_feat.values()) + list(freq_feat.values())
            samples.append(sample)

        return samples

    # 生成数据
    data_rest = generate_gesture('rest', 10)
    data_fist = generate_gesture('fist', 10)
    data_open = generate_gesture('open', 10)

    X = np.array(data_rest + data_fist + data_open)
    y = np.array([0]*10 + [1]*10 + [2]*10)

    # 训练分类器
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    clf = EMGClassifier(classifier_type='random_forest', n_estimators=50)
    clf.prepare_data(X, y, test_size=0.3, random_state=42)
    clf.train(X_train, y_train)
    acc = clf.evaluate(X_test, y_test)

    print("  [OK] 完整流程测试通过")
    print(f"    - 数据生成: {len(X)}个样本")
    print(f"    - 特征维度: {X.shape[1]}")
    print(f"    - 分类准确率: {acc:.2%}")
except Exception as e:
    print(f"  [FAIL] 完整流程测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ========== 总结 ==========
print("\n" + "=" * 70)
print("[OK] 所有测试通过！".center(70))
print("=" * 70)
print("\n测试总结:")
print("  1. 模块导入        OK")
print("  2. 滤波器功能      OK")
print("  3. 特征提取        OK")
print("  4. 分类器训练      OK")
print("  5. 完整流程        OK")
print("\n所有notebooks应该都能正常运行了！")
print("\n建议:")
print("  1. 打开任意notebook")
print("  2. 重启内核（Kernel → Restart）")
print("  3. 从头到尾运行所有单元格（Run All）")
