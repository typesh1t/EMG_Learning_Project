# EMG学习项目代码目录

本目录包含10周课程的所有代码实现。

## 📁 目录结构

### 第1-2周：基础认知
- [week01_basics/](week01_basics/) - EMG基础概念和演示
- [week02_device/](week02_device/) - 设备组件和参数理解

### 第3周：信号特征
- [week03_signal_characteristics/](week03_signal_characteristics/) - 信号时域和频域特征

### 第4-5周：Python基础
- [week04_python_basics/](week04_python_basics/) - Python编程入门
- [week05_data_processing/](week05_data_processing/) - 数据处理和可视化

### 第6周：信号预处理 ⭐
- [week06_preprocessing/](week06_preprocessing/) - 滤波器实现
  - `filters.py` - 完整的滤波器模块（带通、陷波、高通、低通）
  - `preprocess_pipeline.py` - 预处理流程

### 第7周：特征提取 ⭐
- [week07_feature_extraction/](week07_feature_extraction/) - 特征提取
  - `features.py` - 时域和频域特征提取
  - 支持滑动窗口特征提取

### 第8周：模式识别 ⭐
- [week08_pattern_recognition/](week08_pattern_recognition/) - 机器学习分类
  - `classifier.py` - EMG分类器（随机森林、SVM、KNN）
  - 完整的训练、评估、保存/加载功能

### 第9-10周：实时系统
- [week09_realtime_system/](week09_realtime_system/) - 实时采集和处理
- [week10_final_project/](week10_final_project/) - 最终项目模板

## 🚀 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境
conda create -n emg_env python=3.9
conda activate emg_env

# 安装依赖
pip install -r ../requirements.txt
```

### 2. 生成示例数据

```bash
cd ../tools
python generate_sample_data.py --output ../data/sample/ --subjects 3 --trials 10
```

### 3. 运行示例

#### 第1周：理解EMG概念
```bash
cd week01_basics
python emg_concept_demo.py
```

#### 第2周：理解采样率
```bash
cd week02_device
python sampling_demo.py
```

#### 第6周：信号滤波
```python
from week06_preprocessing.filters import EMGFilters

# 创建滤波器
filters = EMGFilters(fs=1000)

# 应用预处理
filtered_signal = filters.preprocess_emg(raw_signal)
```

#### 第7周：特征提取
```python
from week07_feature_extraction.features import EMGFeatures

# 提取时域特征
time_features = EMGFeatures.extract_time_features(signal)

# 提取频域特征
freq_features = EMGFeatures.extract_freq_features(signal, fs=1000)

# 滑动窗口特征提取
feature_matrix, names, times = EMGFeatures.sliding_window_features(
    signal, window_size=200, step=100, fs=1000
)
```

#### 第8周：手势分类
```python
from week08_pattern_recognition.classifier import EMGClassifier

# 创建分类器
clf = EMGClassifier(classifier_type='random_forest', n_estimators=100)

# 准备数据
X_train, X_test, y_train, y_test = clf.prepare_data(X, y, test_size=0.2)

# 训练
clf.train(X_train, y_train, feature_names=names, gesture_names=gestures)

# 评估
accuracy = clf.evaluate(X_test, y_test)

# 保存模型
clf.save_model('model.pkl', 'scaler.pkl')
```

## 📊 核心模块说明

### filters.py - 信号预处理
提供EMG信号滤波的完整实现：

- `bandpass_filter()` - 带通滤波器（20-500Hz）
- `lowpass_filter()` - 低通滤波器
- `highpass_filter()` - 高通滤波器
- `notch_filter()` - 陷波滤波器（去除工频干扰）
- `preprocess_emg()` - 完整预处理流程
- `calculate_snr()` - 计算信噪比
- `normalize_signal()` - 信号归一化

### features.py - 特征提取
提供时域和频域特征提取：

**时域特征**:
- MAV (Mean Absolute Value) - 平均绝对值
- RMS (Root Mean Square) - 均方根
- VAR (Variance) - 方差
- WL (Waveform Length) - 波形长度
- ZC (Zero Crossing) - 过零率
- SSC (Slope Sign Change) - 斜率符号变化
- IEMG - 积分EMG
- DASDV - 标准差

**频域特征**:
- MNF (Mean Frequency) - 平均频率
- MDF (Median Frequency) - 中值频率
- Peak Frequency - 峰值频率
- Total Power - 总功率
- Frequency Ratio - 频率比

**滑动窗口**:
- `sliding_window_features()` - 连续信号的特征提取

### classifier.py - 模式识别
提供完整的分类流程：

**支持的分类器**:
- Random Forest（随机森林）- 推荐
- SVM（支持向量机）
- KNN（K近邻）

**主要功能**:
- `prepare_data()` - 数据划分
- `train()` - 模型训练
- `evaluate()` - 性能评估
- `plot_confusion_matrix()` - 混淆矩阵可视化
- `plot_feature_importance()` - 特征重要性分析
- `save_model()` / `load_model()` - 模型保存和加载

## 🎯 学习路径建议

### 初学者路径
1. 从第1周开始，依次学习
2. 重点掌握第6-8周的核心模块
3. 使用提供的示例数据练习

### 快速实践路径
1. 直接学习第6-8周（核心技术）
2. 使用核心模块处理数据
3. 完成手势识别项目

### 项目开发路径
1. 使用核心模块搭建基础框架
2. 根据需求调整参数
3. 集成实时采集和处理

## 🔧 常用命令

### 查看信号
```bash
python ../tools/signal_viewer.py data/sample/subject_01/fist/trial_001.csv --stats --spectrum
```

### 生成更多数据
```bash
python ../tools/generate_sample_data.py --subjects 10 --trials 20 --gestures rest fist open flex extend
```

### 运行测试
```bash
# 测试滤波器
python -m week06_preprocessing.filters

# 测试特征提取
python -m week07_feature_extraction.features

# 测试分类器
python -m week08_pattern_recognition.classifier
```

## 📚 进一步学习

### 优化建议
1. 尝试不同的滤波器参数
2. 探索更多特征组合
3. 调整分类器超参数
4. 实现交叉验证

### 扩展方向
1. 增加更多手势类型
2. 实现实时系统
3. 开发GUI界面
4. 集成硬件设备

## ❓ 常见问题

### Q: 如何导入模块？
A: 从项目根目录运行，或添加到PYTHONPATH：
```python
import sys
sys.path.append('path/to/EMG_Learning_Project')
from code.week06_preprocessing.filters import EMGFilters
```

### Q: 示例数据在哪里？
A: 运行 `tools/generate_sample_data.py` 生成，保存在 `data/sample/` 目录

### Q: 如何处理真实EMG数据？
A:
1. 确保数据格式为CSV，包含通道列
2. 使用 `filters.py` 进行预处理
3. 使用 `features.py` 提取特征
4. 使用 `classifier.py` 训练分类器

### Q: 准确率很低怎么办？
A:
1. 检查数据质量（SNR）
2. 调整特征提取参数（窗口大小）
3. 尝试不同的特征组合
4. 增加训练数据量
5. 优化分类器参数

## 📚 学习资源

### 本地资源（强烈推荐）
每个week目录都有详细的README和学习资源链接：

- [Week 1-2学习资源](week01_basics/README_详细.md#学习资源) - 基础理论、视频教程
- [Week 3学习资源](week03_signal_characteristics/README.md#学习资源) - 信号特性
- [Week 4学习资源](week04_python_basics/README.md#学习资源) - Python编程
- [Week 5学习资源](week05_data_processing/README.md#学习资源) - 数据可视化
- [Week 6学习资源](week06_preprocessing/README.md#学习资源) - 信号滤波
- [Week 7学习资源](week07_feature_extraction/README.md#学习资源) - 特征提取
- [Week 8学习资源](week08_pattern_recognition/README.md#学习资源) - 模式识别
- [Week 9学习资源](week09_realtime_system/README.md#学习资源) - 实时系统
- [Week 10项目指南](week10_final_project/README.md#学习资源) - 综合项目

### 核心教材文档
- [EMG设备与信号基础.md](week01_basics/EMG设备与信号基础.md) - 20,000字学术级文档
- [EMG信号特征分析详解.md](week02_device/EMG信号特征分析详解.md) - 25,000字详细教程

### 完整资源汇总
- [EMG学习资源汇总](../docs/EMG学习资源汇总.md) - 所有在线资源（视频、文章、论文）
- [视频资源清单](../resources/videos/视频资源清单.md) - YouTube和B站视频教程

### 视频教程推荐
- [Surface EMG Signal Processing](https://youtu.be/5AtAoO51vWQ) - YouTube，英文，入门必看
- [多通道肌电传感器演示](https://www.bilibili.com/opus/676738656717766660) - B站，中文，实际应用

### 在线文章推荐
- [EMG信号处理合集（Python）](https://blog.csdn.net/YINTENAXIONGNAIER/article/details/134566397) - 完整代码示例
- [肌电信号特点详解](https://blog.csdn.net/gitblog_06641/article/details/142570969) - 信号特性
- [Surface EMG Best Practices](https://colab.ws/articles/10.1016%2Fj.jelekin.2020.102440) - 最佳实践

### 学术资源
- [Merletti EMG教程合集](https://www.robertomerletti.it/it/emg/material/tutorials/) - 官方教程
- [EMG处理技术综述](https://arxiv.org/pdf/2304.04098) - arXiv论文
- SENIAM标准：http://seniam.org/

## 📧 获取帮助

- 查看每周的README文件（包含详细的学习资源）
- 参考 [docs/00_整体学习计划.md](../docs/00_整体学习计划.md)
- 查看代码中的详细注释
- 在GitHub Issues提问
- 查阅[完整资源汇总](../docs/EMG学习资源汇总.md)

---

**祝学习愉快！** 🎓
