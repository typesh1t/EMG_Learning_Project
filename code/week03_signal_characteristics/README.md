# 第3周：EMG信号特征认知

## 学习目标
- 认识真实EMG信号的时域和频域特征
- 理解信号的随机性、突发性、幅度范围
- 识别常见噪声类型（工频干扰、运动伪影等）
- 学会评估信号质量（SNR）

## 核心概念

### 时域特征
- 随机性：EMG信号看起来像噪声，无明显周期
- 突发性：肌肉收缩时信号幅度突然增大
- 幅度范围：静息约5-20μV，收缩可达500μV-5mV
- 零均值：信号在零点上下对称波动

### 频域特征
- 主要频率范围：20-500 Hz
- 能量集中区域：50-150 Hz
- 低频端（20-50Hz）：能量较少
- 高频端（150-500Hz）：能量逐渐衰减

## 本周内容

### 实践代码
1. `signal_viewer.py` - 可视化EMG信号的时域和频域特征
2. `noise_identification.py` - 识别和分析各种噪声
3. `signal_quality_assessment.py` - 评估信号质量（SNR计算）
4. `exercises.py` - 本周练习题

### 学习步骤
1. 运行signal_viewer.py加载样本数据
2. 观察静息期和激活期的信号差异
3. 使用noise_identification.py识别噪声
4. 完成exercises.py中的练习

## 作业
1. 加载提供的样本数据，绘制时域和频域图
2. 识别信号中的静息段、激活段、噪声段
3. 计算信号的RMS值、峰值、过零率
4. 写一段话描述你观察到的信号特征

## 参考资料
- docs/00_整体学习计划.md 第二阶段
- 样本数据：data/sample/

## 学习资源

### 核心教材
- [EMG设备与信号基础.md](../week01_basics/EMG设备与信号基础.md) - 信号特性详解
- [EMG信号特征分析详解.md](../week02_device/EMG信号特征分析详解.md) - 时域和频域特征

### 在线文章
- [肌电信号的特点、频率、幅值](https://blog.csdn.net/gitblog_06641/article/details/142570969) - 信号特性详解
- [简单讲肌电信号（时域、频域）](https://zhuanlan.zhihu.com/p/138204944) - 时频域理解

### 视频教程
- [Surface EMG Signal Processing](https://youtu.be/5AtAoO51vWQ) - YouTube，信号处理基础

### 完整资源
查看 [../../docs/EMG学习资源汇总.md](../../docs/EMG学习资源汇总.md)
