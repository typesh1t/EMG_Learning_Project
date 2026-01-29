# 第6周：信号预处理

## 学习目标
- 理解滤波器的原理和作用
- 实现EMG信号的带通滤波和陷波滤波
- 评估信号质量（SNR）

## 本周内容

### 1. 滤波器基础
理解低通、高通、带通、陷波滤波器

### 2. EMG信号滤波实现
- 带通滤波器（20-500Hz）
- 陷波滤波器（去除50/60Hz工频干扰）

### 3. 信号质量评估
计算信噪比(SNR)，评估滤波效果

## 文件说明
- `filters.py` - 滤波器实现模块
- `preprocess_pipeline.py` - 完整预处理流程
- `filter_demo.py` - 滤波效果对比演示
- `exercises.py` - 本周练习题

## 实践任务
1. 运行 `filter_demo.py` 查看滤波效果
2. 使用 `preprocess_pipeline.py` 处理样本数据
3. 完成 `exercises.py` 中的练习

## 作业
1. 实现不同阶数的滤波器并对比效果
2. 处理含工频干扰的数据并验证去除效果
3. 编写批量处理脚本

## 学习资源

### 核心模块
- [filters.py](filters.py) - 本地，完整的滤波器实现

### 理论基础
- [EMG设备与信号基础.md](../week01_basics/EMG设备与信号基础.md) - 噪声类型和解决方案

### 在线教程
- [Surface EMG Signal Processing](https://youtu.be/5AtAoO51vWQ) - YouTube，滤波器设计
- [EMG信号处理合集（含Python）](https://blog.csdn.net/YINTENAXIONGNAIER/article/details/134566397) - 带通滤波、陷波滤波代码
- [Surface EMG Best Practices](https://colab.ws/articles/10.1016%2Fj.jelekin.2020.102440) - 预处理最佳实践
- [Merletti教程](https://www.robertomerletti.it/it/emg/material/tutorials/) - 信号调理和预处理

### 技术文档
- [Overview of Processing Techniques](https://arxiv.org/pdf/2304.04098) - 预处理技术综述

### 完整资源
查看 [../../docs/EMG学习资源汇总.md](../../docs/EMG学习资源汇总.md)
