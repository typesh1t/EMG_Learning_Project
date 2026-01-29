# 第8周：模式识别与分类

## 学习目标
- 理解机器学习分类的基本流程
- 掌握3种常用分类器（随机森林、SVM、KNN）
- 学会评估分类模型性能
- 实现完整的手势识别系统

## 本周内容

### 1. 机器学习基础
- 训练集和测试集划分
- 特征归一化/标准化
- 交叉验证
- 过拟合和欠拟合

### 2. 分类器详解
- **Random Forest（随机森林）** - 推荐，稳定性好
- **SVM（支持向量机）** - 小样本效果好
- **KNN（K近邻）** - 简单直观

### 3. 性能评估
- 准确率（Accuracy）
- 混淆矩阵（Confusion Matrix）
- 精确率、召回率、F1分数
- 交叉验证分数

### 4. 完整流程
1. 数据准备（特征提取）
2. 数据划分（训练集/测试集）
3. 模型训练
4. 性能评估
5. 模型保存和加载

## 文件说明
- `classifier.py` - 分类器核心模块
- `train_model.py` - 模型训练脚本
- `evaluate_model.py` - 模型评估脚本
- `gesture_recognition_demo.py` - 完整演示
- `exercises.py` - 本周练习题

## 实践任务
1. 运行 `classifier.py` 查看分类器实现
2. 使用 `train_model.py` 训练手势识别模型
3. 使用 `evaluate_model.py` 评估模型性能
4. 完成 `exercises.py` 中的练习

## 作业
1. 对比3种分类器的性能
2. 调整分类器参数（如随机森林的树数量）
3. 尝试不同的特征组合
4. 实现自己的手势识别应用

## 学习资源

### 核心模块
- [classifier.py](classifier.py) - 本地，完整的分类器实现

### 理论基础
- scikit-learn官方文档（分类器教程）
- Week 7的特征提取文档（特征组合策略）

### 在线教程
- [EMG信号处理合集（含Python）](https://blog.csdn.net/YINTENAXIONGNAIER/article/details/134566397) - 分类实现
- [Overview of Processing Techniques](https://arxiv.org/pdf/2304.04098) - 分类方法综述
- [传统EMG信号预处理与分类综述](https://www.mdpi.com/1424-8220/13/9/12431) - 经典分类方法

### 学术论文
- Hudgins et al. "A New Strategy for Multifunction Myoelectric Control" (1993) - 经典手势识别
- Farina et al. "The Extraction of Neural Strategies from the Surface EMG" (2004) - 模式识别策略

### 机器学习资源
- [scikit-learn官方文档](https://scikit-learn.org/stable/) - 分类器详细说明
- [机器学习基础教程](https://www.coursera.org/learn/machine-learning) - Coursera课程

### 完整资源
查看 [../../docs/EMG学习资源汇总.md](../../docs/EMG学习资源汇总.md)

## 实用技巧

### 提高分类准确率的方法
1. **特征工程**
   - 选择互补的特征组合
   - 尝试不同的窗口大小
   - 标准化/归一化特征

2. **数据增强**
   - 增加训练样本数量
   - 使用交叉验证
   - 平衡各类别样本数

3. **模型调参**
   - 随机森林：调整树的数量（50-200）
   - SVM：调整C和gamma参数
   - KNN：调整K值（3-15）

4. **集成方法**
   - 多个模型投票
   - 加权平均
   - Stacking集成

## 性能基准

参考性能（基于本项目的示例数据）：
- 3类手势（rest, fist, open）
- 4通道EMG数据
- 标准特征集（MAV, RMS, VAR, WL, ZC, SSC）

预期准确率：
- Random Forest: 85-95%
- SVM: 80-90%
- KNN: 75-85%

如果准确率低于70%，检查：
- 特征是否正确提取
- 数据是否有标注错误
- 窗口大小是否合适
- 特征是否归一化

## 重要提示

本周是将前面所学应用到实际问题的关键：
1. 理解机器学习的完整流程
2. 学会评估和改进模型
3. 为Week 9-10的实时系统打基础
