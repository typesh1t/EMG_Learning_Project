# 第7周：特征提取

## 学习目标
- 理解时域特征和频域特征的物理意义
- 掌握10种时域特征的计算方法
- 掌握8种频域特征的计算方法
- 实现滑动窗口特征提取

## 本周内容

### 1. 时域特征（10种）
- MAV（平均绝对值）- 反映信号强度
- RMS（均方根）- 反映信号能量
- VAR（方差）- 反映信号波动
- WL（波形长度）- 反映信号复杂度
- ZC（过零率）- 反映频率信息
- SSC（斜率符号变化）- 反映信号变化
- IEMG（积分EMG）- 反映总活动量
- DASDV（差值绝对标准差）- 反映变化率
- PEAK（峰值）- 最大幅度
- MEAN（均值）- 平均水平

### 2. 频域特征（8种）
- MNF（平均频率）- 频谱重心
- MDF（中值频率）- 频谱中位数
- Peak Frequency（峰值频率）- 能量最大频率
- Total Power（总功率）- 整体能量
- SM1-SM3（频谱矩）- 频谱统计特性
- Frequency Ratio（频率比）- 高低频比值

### 3. 滑动窗口特征提取
实现窗口滑动，提取时间序列特征

## 文件说明
- `features.py` - 特征提取核心模块
- `feature_demo.py` - 特征提取演示
- `feature_comparison.py` - 特征对比分析
- `exercises.py` - 本周练习题

## 实践任务
1. 运行 `features.py` 查看所有特征的计算
2. 使用 `feature_demo.py` 提取样本数据特征
3. 完成 `exercises.py` 中的练习

## 作业
1. 对同一信号提取所有18种特征并分析
2. 比较静息段和激活段的特征差异
3. 实现自定义的滑动窗口大小和步长

## 学习资源

### 核心模块
- [features.py](features.py) - 本地，完整的特征提取实现

### 理论基础（必读）
- [EMG信号特征分析详解.md](../week02_device/EMG信号特征分析详解.md) - 25,000字详细文档
  - 第一部分：10种时域特征详解（含公式、代码、典型值）
  - 第二部分：8种频域特征详解（含FFT、功率谱分析）
  - 第三部分：时频分析（STFT）

### 在线教程
- [肌电信号的特点、频率、幅值](https://blog.csdn.net/gitblog_06641/article/details/142570969) - 特征详解
- [简单讲肌电信号（时域、频域）](https://zhuanlan.zhihu.com/p/138204944) - 时频域理解
- [EMG信号处理合集（含Python）](https://blog.csdn.net/YINTENAXIONGNAIER/article/details/134566397) - Python特征提取代码

### 学术论文
- Hudgins et al. "A New Strategy for Multifunction Myoelectric Control" (1993) - 经典特征集
- 特征选择和组合策略参考Week 2文档

### 完整资源
查看 [../../docs/EMG学习资源汇总.md](../../docs/EMG学习资源汇总.md)

## 重要提示

本周是核心内容，特征提取的质量直接影响后续的分类效果。务必：
1. 理解每个特征的物理意义（不要只会计算）
2. 对比不同特征在不同信号上的表现
3. 思考哪些特征组合最有效
