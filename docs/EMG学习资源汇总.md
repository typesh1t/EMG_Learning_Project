# EMG学习资源汇总

本文档汇总了EMG（肌电信号）学习的优质在线资源，包括基础概念、视频教程、技术文档和学术论文。

---

## 第一部分 | EMG 是什么（概念与基本认识）

### 1. Wiki 基础介绍（英文，全面定义）

**EMG 是什么**
- 链接：https://en.wikipedia.org/wiki/Electromyography
- 语言：英文
- 内容：维基百科全面介绍EMG的定义、表面EMG vs 针刺EMG、信号用途等基础概念
- 适合：初学者建立基本认知

### 2. 中文基础解释（知乎讲解）

**EMG 肌电图原理与作用**
- 链接：https://zhuanlan.zhihu.com/p/408281822
- 语言：中文
- 内容：讲解肌电图的基本概念、信号指标（RMS、频域等）
- 适合：中文读者快速理解基础概念

---

## 第二部分 | 视频教程：从视频开始理解 EMG

### 视频 1 | EMG Signal Processing Tutorial（英文）

**Surface Electromyography Signal Processing | Part 1**
- 链接：https://youtu.be/5AtAoO51vWQ
- 语言：英文
- 内容：讲解表面肌电基本信号处理步骤，包括滤波、预处理等
- 适合：第一次接触EMG的学习者
- 推荐指数：⭐⭐⭐⭐⭐
- 备注：这是基础级别视频，非常适合入门

### 视频 2 | 多通道肌电应用演示（中文）

**B站：多通道肌电传感器与应用示例**
- 链接：https://www.bilibili.com/opus/676738656717766660
- 语言：中文
- 内容：演示多通道肌电模块采集数据并在实际场景中的应用案例
- 适合：想了解"EMG能做什么"的学习者
- 推荐指数：⭐⭐⭐⭐
- 备注：偏硬件和应用，让学生直观感受EMG的用途

---

## 第三部分 | EMG 信号采集 & 多通道系统

### 1. 多通道采集介绍（中文）

**多通道无线EMG采集系统介绍**
- 链接：https://www.sohu.com/a/915078813_121781227
- 语言：中文
- 内容：说明多通道系统的概念、用途、带宽、增益等关键参数
- 适合：了解多通道采集设备的技术细节

### 2. 第三方教程 PDF（英文）

**Surface EMG detection, conditioning and pre-processing: Best practices**
- 链接：https://colab.ws/articles/10.1016%2Fj.jelekin.2020.102440
- 语言：英文
- 内容：面向非工程背景读者的详尽教程，可免费阅读
- 适合：想系统学习EMG采集和预处理的学习者
- 推荐指数：⭐⭐⭐⭐⭐
- 备注：指南式内容，非常实用

### 3. 下载版教程合集（Merletti 教程汇总）

**官方教程集（多篇EMG教程）**
- 链接：https://www.robertomerletti.it/it/emg/material/tutorials/
- 语言：英文
- 内容：包含表面EMG采集、预处理、空间与时间检波、最佳规范等多个教程
- 适合：深入学习EMG采集和信号处理的研究者
- 推荐指数：⭐⭐⭐⭐⭐
- 备注：Roberto Merletti是EMG领域的权威专家

---

## 第四部分 | 肌电信号的特点与理解

### 1. 中文说明（博客）

**肌电信号的特点、频率、幅值等**
- 链接：https://blog.csdn.net/gitblog_06641/article/details/142570969
- 语言：中文
- 内容：详细介绍信号产生机制、频率范围、主要特征
- 适合：想深入理解EMG信号特性的学习者

### 2. 中文基础总结（知乎）

**简单讲肌电信号（时域、频域）**
- 链接：https://zhuanlan.zhihu.com/p/138204944
- 语言：中文
- 内容：讲解什么是肌电信号以及它在运动分析中的意义
- 适合：快速了解EMG信号时域和频域特性

---

## 第五部分 | 信号处理入门（预处理流程 & 代码思路）

### 1. Python 信号处理实例（中文博客）

**EMG肌电信号处理合集（含Python）**
- 链接：https://blog.csdn.net/YINTENAXIONGNAIER/article/details/134566397
- 语言：中文
- 内容：讲解DC去除、带通滤波、整流、线性包络、归一化等步骤，并提供代码示例
- 适合：想学习Python实现EMG信号处理的学习者
- 推荐指数：⭐⭐⭐⭐⭐
- 备注：包含完整的代码示例，非常实用

### 2. 推荐英文综述（入门理解处理技术）

**Overview of Processing Techniques for Surface EMG Signals**
- 链接：https://arxiv.org/pdf/2304.04098
- 语言：英文
- 内容：开源综述论文，介绍采集与信号处理技术概览
- 适合：想系统了解EMG信号处理技术的学习者
- 推荐指数：⭐⭐⭐⭐

---

## 第六部分 | 进阶阅读（选看）

这些资料不强制阅读，但如果想更系统理解可以参考。

### 1. 频域分析最佳实践教程（英文）

- 链接：https://research.chalmers.se/publication/543868/file/543868_Fulltext.pdf
- 语言：英文
- 内容：深入讲解信号频域分析
- 适合：进阶学习者

### 2. EMG预处理与检测标准（详细Tutorial）

- 链接：https://www.sciencedirect.com/science/article/pii/S1050641124000816
- 语言：英文
- 内容：解释EMG信号的频谱、滤波与特性
- 适合：需要规范化处理流程的研究者

### 3. 传统EMG信号预处理与分类综述（经典论文）

- 链接：https://www.mdpi.com/1424-8220/13/9/12431
- 语言：英文
- 内容：回顾预处理与分类方法
- 适合：想了解EMG分类方法的研究者

---

## 学习建议

### 第一阶段：建立基础认知（Week 1-2）

建议按以下顺序学习：

1. 先看Wiki和知乎文章，理解什么是EMG
2. 观看YouTube视频《Surface Electromyography Signal Processing | Part 1》
3. 观看B站多通道演示视频，了解实际应用
4. 阅读本项目的`EMG设备与信号基础.md`和`EMG信号特征分析详解.md`

### 第二阶段：深入信号特性（Week 3-4）

1. 阅读CSDN博客了解信号特点
2. 阅读知乎文章理解时域和频域
3. 学习本项目的Python基础教程
4. 运行本项目的信号模拟和可视化脚本

### 第三阶段：信号处理实践（Week 5-8）

1. 阅读CSDN的Python信号处理实例
2. 学习本项目的滤波、特征提取和分类模块
3. 参考arxiv综述了解处理技术全貌
4. 实践本项目的完整处理流程

### 第四阶段：规范化和进阶（Week 9-10）

1. 阅读Merletti教程合集
2. 学习best practices文档
3. 参考进阶论文深入理解
4. 完成本项目的实时系统开发

---

## 资源分类索引

### 按语言分类

**中文资源：**
- 知乎：EMG肌电图原理与作用
- B站：多通道肌电传感器与应用
- 搜狐：多通道无线EMG采集系统
- CSDN：肌电信号特点详解
- CSDN：EMG肌电信号处理合集（含Python代码）
- 知乎：简单讲肌电信号

**英文资源：**
- Wikipedia：Electromyography
- YouTube：Surface Electromyography Signal Processing
- Best practices：Surface EMG detection and conditioning
- Merletti教程：官方EMG教程合集
- arXiv：Overview of Processing Techniques
- 其他学术论文

### 按内容类型分类

**基础概念：**
- Wikipedia
- 知乎基础解释

**视频教程：**
- YouTube EMG Signal Processing
- B站多通道演示

**技术文档：**
- Best practices PDF
- Merletti教程合集
- 多通道系统介绍

**信号特性：**
- CSDN信号特点博客
- 知乎时域频域文章

**代码实践：**
- CSDN Python信号处理合集

**学术论文：**
- arXiv综述
- 频域分析教程
- 预处理与分类综述

### 按难度分级

**入门级（初学者必看）：**
- ⭐ Wikipedia EMG介绍
- ⭐ 知乎基础解释
- ⭐ YouTube信号处理视频
- ⭐ B站应用演示

**中级（有一定基础）：**
- ⭐⭐ CSDN信号特点
- ⭐⭐ CSDN Python代码实例
- ⭐⭐ 多通道系统介绍
- ⭐⭐ 知乎时域频域

**高级（深入研究）：**
- ⭐⭐⭐ Best practices文档
- ⭐⭐⭐ Merletti教程合集
- ⭐⭐⭐ arXiv综述
- ⭐⭐⭐ 学术论文合集

---

## 如何使用本资源汇总

### 对于零基础学习者

1. 从Wikipedia和知乎开始，建立基本概念
2. 观看YouTube和B站视频，直观感受EMG
3. 结合本项目Week 1-2的文档，深入理解
4. 按照本项目的10周计划循序渐进

### 对于有编程基础的学习者

1. 快速浏览概念性资源
2. 重点学习CSDN的Python代码实例
3. 直接上手本项目的代码模块
4. 参考best practices规范自己的代码

### 对于研究人员

1. 阅读Merletti教程合集建立系统认知
2. 研读arXiv综述和学术论文
3. 使用本项目作为实验平台
4. 参考best practices确保研究规范

---

## 资源更新

本文档会根据新的优质资源持续更新。

如果您发现了好的EMG学习资源，欢迎通过以下方式分享：
- 在项目GitHub页面提交Issue
- 发送邮件给项目维护者
- 提交Pull Request添加新资源

---

## 致谢

感谢以下资源的创作者和分享者：
- Wikipedia EMG条目的编辑者
- YouTube和B站的视频教程制作者
- CSDN和知乎的博主
- Roberto Merletti教授及其团队
- 各学术论文的作者

---

**最后更新：** 2026-01-29
**维护者：** EMG Learning Project Team
