#!/usr/bin/env python3
"""
EMG模式识别/分类模块
实现手势识别的完整流程
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                            accuracy_score)
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

class EMGClassifier:
    """EMG手势分类器"""

    def __init__(self, classifier_type='random_forest', **kwargs):
        """
        初始化分类器

        参数:
            classifier_type: 分类器类型
                - 'random_forest': 随机森林(默认)
                - 'svm': 支持向量机
                - 'knn': K近邻
            **kwargs: 传递给分类器的参数
        """
        self.classifier_type = classifier_type
        self.scaler = StandardScaler()
        self.clf = None
        self.feature_names = None
        self.gesture_names = None

        # 创建分类器
        if classifier_type == 'random_forest':
            self.clf = RandomForestClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 10),
                random_state=kwargs.get('random_state', 42),
                n_jobs=-1
            )
        elif classifier_type == 'svm':
            self.clf = SVC(
                kernel=kwargs.get('kernel', 'rbf'),
                C=kwargs.get('C', 1.0),
                gamma=kwargs.get('gamma', 'scale'),
                random_state=kwargs.get('random_state', 42),
                probability=True
            )
        elif classifier_type == 'knn':
            self.clf = KNeighborsClassifier(
                n_neighbors=kwargs.get('n_neighbors', 5),
                weights=kwargs.get('weights', 'distance')
            )
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")

    def prepare_data(self, X, y, test_size=0.2, random_state=42):
        """
        准备训练和测试数据

        参数:
            X: 特征矩阵 (n_samples, n_features)
            y: 标签数组 (n_samples,)
            test_size: 测试集比例
            random_state: 随机种子

        返回:
            X_train, X_test, y_train, y_test: 划分后的数据
        """
        # 划分数据集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y  # 保持各类别比例
        )

        print(f"训练集大小: {X_train.shape}")
        print(f"测试集大小: {X_test.shape}")
        print(f"训练集标签分布: {np.bincount(y_train)}")
        print(f"测试集标签分布: {np.bincount(y_test)}")

        return X_train, X_test, y_train, y_test

    def train(self, X_train, y_train, feature_names=None, gesture_names=None):
        """
        训练分类器

        参数:
            X_train: 训练特征
            y_train: 训练标签
            feature_names: 特征名称列表
            gesture_names: 手势名称列表
        """
        print(f"\n训练 {self.classifier_type} 分类器...")

        # 保存特征和手势名称
        self.feature_names = feature_names
        self.gesture_names = gesture_names

        # 归一化特征
        X_train_scaled = self.scaler.fit_transform(X_train)

        # 训练模型
        self.clf.fit(X_train_scaled, y_train)

        # 训练集准确率
        train_pred = self.clf.predict(X_train_scaled)
        train_acc = accuracy_score(y_train, train_pred)

        print(f"训练完成！训练集准确率: {train_acc*100:.2f}%")

    def evaluate(self, X_test, y_test, show_report=True):
        """
        评估模型性能

        参数:
            X_test: 测试特征
            y_test: 测试标签
            show_report: 是否显示详细报告

        返回:
            accuracy: 准确率
        """
        # 归一化
        X_test_scaled = self.scaler.transform(X_test)

        # 预测
        y_pred = self.clf.predict(X_test_scaled)

        # 计算准确率
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n测试集准确率: {accuracy*100:.2f}%")

        # 显示详细报告
        if show_report:
            print("\n分类报告:")
            if self.gesture_names:
                print(classification_report(y_test, y_pred,
                                           target_names=self.gesture_names))
            else:
                print(classification_report(y_test, y_pred))

        return accuracy

    def plot_confusion_matrix(self, X_test, y_test, save_path=None):
        """
        绘制混淆矩阵

        参数:
            X_test: 测试特征
            y_test: 测试标签
            save_path: 保存路径
        """
        # 预测
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.clf.predict(X_test_scaled)

        # 计算混淆矩阵
        cm = confusion_matrix(y_test, y_pred)

        # 绘图
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.gesture_names if self.gesture_names else 'auto',
                   yticklabels=self.gesture_names if self.gesture_names else 'auto')
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        plt.title(f'混淆矩阵 - {self.classifier_type}')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ 混淆矩阵已保存到: {save_path}")

        plt.tight_layout()
        plt.show()

    def plot_feature_importance(self, top_n=10, save_path=None):
        """
        绘制特征重要性（仅适用于随机森林）

        参数:
            top_n: 显示前N个重要特征
            save_path: 保存路径
        """
        if self.classifier_type != 'random_forest':
            print("特征重要性仅适用于随机森林分类器")
            return

        if not self.feature_names:
            print("未提供特征名称")
            return

        # 获取特征重要性
        importances = self.clf.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]

        # 绘图
        plt.figure(figsize=(10, 6))
        plt.bar(range(top_n), importances[indices])
        plt.xticks(range(top_n),
                  [self.feature_names[i] for i in indices],
                  rotation=45, ha='right')
        plt.xlabel('特征')
        plt.ylabel('重要性')
        plt.title(f'Top {top_n} 重要特征')
        plt.grid(True, alpha=0.3, axis='y')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ 特征重要性图已保存到: {save_path}")

        plt.tight_layout()
        plt.show()

    def predict(self, X):
        """
        预测新数据

        参数:
            X: 特征矩阵

        返回:
            predictions: 预测标签
            probabilities: 预测概率
        """
        X_scaled = self.scaler.transform(X)
        predictions = self.clf.predict(X_scaled)

        if hasattr(self.clf, 'predict_proba'):
            probabilities = self.clf.predict_proba(X_scaled)
        else:
            probabilities = None

        return predictions, probabilities

    def save_model(self, model_path, scaler_path):
        """
        保存模型

        参数:
            model_path: 模型保存路径
            scaler_path: 归一化器保存路径
        """
        joblib.dump(self.clf, model_path)
        joblib.dump(self.scaler, scaler_path)
        print(f"✓ 模型已保存到: {model_path}")
        print(f"✓ 归一化器已保存到: {scaler_path}")

    @classmethod
    def load_model(cls, model_path, scaler_path, classifier_type='random_forest'):
        """
        加载模型

        参数:
            model_path: 模型路径
            scaler_path: 归一化器路径
            classifier_type: 分类器类型

        返回:
            classifier: 加载的分类器对象
        """
        classifier = cls(classifier_type=classifier_type)
        classifier.clf = joblib.load(model_path)
        classifier.scaler = joblib.load(scaler_path)
        print(f"✓ 模型已从 {model_path} 加载")
        return classifier


# 示例使用
if __name__ == "__main__":
    print("\n" + "="*60)
    print("EMG分类器模块测试".center(60))
    print("="*60 + "\n")

    # 生成模拟数据
    print("生成模拟数据...")
    n_samples_per_class = 100
    n_features = 14
    n_classes = 5

    X = []
    y = []

    for class_id in range(n_classes):
        # 每个类别有不同的特征分布
        class_features = np.random.randn(n_samples_per_class, n_features) + class_id
        X.append(class_features)
        y.extend([class_id] * n_samples_per_class)

    X = np.vstack(X)
    y = np.array(y)

    gesture_names = ['握拳', '张开', '左转', '右转', '静息']
    feature_names = ['MAV', 'RMS', 'VAR', 'WL', 'ZC', 'SSC',
                    'IEMG', 'DASDV', 'PEAK', 'MNF', 'MDF',
                    'Peak_Freq', 'Total_Power', 'Freq_Ratio']

    # 创建分类器
    clf = EMGClassifier(classifier_type='random_forest',
                       n_estimators=100,
                       max_depth=10)

    # 准备数据
    X_train, X_test, y_train, y_test = clf.prepare_data(X, y, test_size=0.2)

    # 训练
    clf.train(X_train, y_train,
             feature_names=feature_names,
             gesture_names=gesture_names)

    # 评估
    accuracy = clf.evaluate(X_test, y_test)

    # 测试预测
    print("\n测试单个样本预测...")
    test_sample = X_test[0:1]
    pred, prob = clf.predict(test_sample)
    print(f"预测结果: {gesture_names[pred[0]]}")
    if prob is not None:
        print(f"各类别概率: {prob[0]}")

    print("\n✓ 分类器模块测试完成\n")
