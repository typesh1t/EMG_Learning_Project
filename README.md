# EMG肌电信号学习项目

> 从零开始的EMG信号处理完整教程（面向高中国际生）

## 项目简介

这是一个为期8-10周的EMG（肌电信号）学习项目，旨在帮助零基础的高中国际生掌握：
- EMG信号的基本原理和特征
- 多通道肌电采集设备的使用
- Python编程和科学计算
- 信号处理和特征提取
- 机器学习模式识别
- 实时系统搭建

## 快速开始

### 1. 环境配置

```bash
# 创建Python虚拟环境
conda create -n emg_env python=3.9
conda activate emg_env

# 安装依赖
pip install -r requirements.txt
```

### 2. 查看学习计划

请先阅读：[docs/00_整体学习计划.md](docs/00_整体学习计划.md)

### 3. 开始学习

按周次顺序学习，每周对应一个文件夹：
- 第1-2周：[code/week01_basics](code/week01_basics) 和 [code/week02_device](code/week02_device)
- 第3周：[code/week03_signal_characteristics](code/week03_signal_characteristics)
- 第4-5周：[code/week04_python_basics](code/week04_python_basics) 和 [code/week05_data_processing](code/week05_data_processing)
- 第6周：[code/week06_preprocessing](code/week06_preprocessing)
- 第7周：[code/week07_feature_extraction](code/week07_feature_extraction)
- 第8周：[code/week08_pattern_recognition](code/week08_pattern_recognition)
- 第9-10周：[code/week09_realtime_system](code/week09_realtime_system) 和 [code/week10_final_project](code/week10_final_project)

## 项目结构

```
EMG_Learning_Project/
│
├── README.md                          # 项目说明（本文件）
├── requirements.txt                   # Python依赖列表
│
├── docs/                             # 文档目录
│   ├── 00_整体学习计划.md             # 完整学习计划
│   ├── 01_EMG基础知识.md              # EMG理论知识
│   ├── 02_设备使用指南.md             # 硬件设备说明
│   ├── 03_Python快速入门.md           # Python教程
│   └── 04_常见问题FAQ.md              # 常见问题解答
│
├── code/                             # 代码目录（按周组织）
│   ├── week01_basics/                # 第1周：EMG基础认知
│   ├── week02_device/                # 第2周：设备认识
│   ├── week03_signal_characteristics/ # 第3周：信号特征
│   ├── week04_python_basics/         # 第4周：Python基础
│   ├── week05_data_processing/       # 第5周：数据处理
│   ├── week06_preprocessing/         # 第6周：信号预处理
│   ├── week07_feature_extraction/    # 第7周：特征提取
│   ├── week08_pattern_recognition/   # 第8周：模式识别
│   ├── week09_realtime_system/       # 第9周：实时系统
│   └── week10_final_project/         # 第10周：最终项目
│
├── data/                             # 数据目录
│   ├── raw/                          # 原始数据
│   ├── processed/                    # 处理后的数据
│   └── sample/                       # 样本数据（用于练习）
│
├── notebooks/                        # Jupyter笔记本
│   ├── 01_信号可视化入门.ipynb
│   ├── 02_滤波器设计实验.ipynb
│   └── 03_特征提取实践.ipynb
│
├── resources/                        # 资源目录
│   ├── images/                       # 图片资源
│   ├── videos/                       # 视频教程链接
│   └── papers/                       # 参考论文
│
└── tools/                            # 工具脚本
    ├── data_collector.py             # 数据采集工具
    ├── signal_viewer.py              # 信号查看器
    └── model_tester.py               # 模型测试工具
```

## 学习路径

### 第一阶段：基础认知（第1-2周）
**目标**: 理解EMG是什么，认识采集设备

**学习内容**:
- EMG信号的产生机制
- EMG的实际应用
- 采集设备的组成和工作原理
- 多通道采集的意义

**作业**:
- 绘制EMG采集系统流程图
- 识别设备各个部件

---

### 第二阶段：信号特征认知（第3周）
**目标**: 认识EMG信号的时域和频域特征

**学习内容**:
- 时域特征：幅度、随机性、突发性
- 频域特征：主要频率范围（20-500Hz）
- 常见噪声类型及识别

**作业**:
- 加载样本数据并可视化
- 识别信号中的噪声

---

### 第三阶段：Python基础（第4-5周）
**目标**: 掌握Python编程和科学计算库

**学习内容**:
- Python基础语法
- NumPy数组操作
- Matplotlib数据可视化
- Pandas数据管理

**作业**:
- 读取CSV文件
- 绘制多通道EMG信号
- 计算基本统计量

---

### 第四阶段：信号预处理（第6周）
**目标**: 实现EMG信号的滤波处理

**学习内容**:
- 滤波器原理和类型
- 带通滤波器实现（20-500Hz）
- 陷波滤波器去除工频干扰
- 信号质量评估（SNR）

**作业**:
- 实现完整预处理流程
- 对比滤波前后效果

---

### 第五阶段：特征提取（第7周）
**目标**: 提取EMG信号的时域和频域特征

**学习内容**:
- 时域特征：MAV, RMS, VAR, WL, ZC, SSC
- 频域特征：MNF, MDF, Peak Frequency
- 滑动窗口特征提取
- 特征归一化

**作业**:
- 提取所有特征
- 可视化特征随时间的变化

---

### 第六阶段：模式识别（第8周）
**目标**: 训练机器学习模型识别手势

**学习内容**:
- 机器学习基本流程
- 数据准备和划分
- 随机森林分类器
- 模型评估（混淆矩阵、准确率）

**作业**:
- 采集手势数据
- 训练分类模型
- 评估模型性能

---

### 第七阶段：实时系统（第9-10周）
**目标**: 搭建完整的实时EMG处理系统

**学习内容**:
- 实时数据采集
- 实时信号处理
- 实时分类和可视化
- 系统整合和优化

**最终项目**:
- 实时手势识别系统
- 或EMG控制游戏
- 或肌肉训练监测应用

---

## 所需硬件

### 入门级（推荐）
- **Arduino Uno** + **Grove EMG Sensor** (~$40)
- 或 **MyoWare Muscle Sensor** (~$40)

### 进阶级
- **OpenBCI Cyton** (8通道, ~$500)
- **Myo Armband** (8通道, 二手~$200)

### 配件
- 电极贴片（Ag/AgCl）
- 导电膏
- USB数据线

---

## 软件依赖

主要Python库：
```
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
scipy>=1.7.0
scikit-learn>=0.24.0
pyqtgraph>=0.12.0
pyserial>=3.5
```

完整依赖见：[requirements.txt](requirements.txt)

---

## 学习资源

### 在线课程
- Coursera: "Digital Signal Processing"
- edX: "Introduction to Biomedical Signals"

### 书籍
- "Biomedical Signal Analysis" by Rangaraj M. Rangayyan
- "Introduction to EMG Signal Processing" by Edward Clancy

### 数据集
- [Ninapro Database](http://ninapro.hevs.ch/) - 公开的EMG手势数据集
- [PhysioNet EMG Database](https://physionet.org/)

### 视频教程
- 链接在 [resources/videos/](resources/videos/) 文件夹

---

## 常见问题

### Q1: 我没有EMG设备怎么办？
A: 可以先使用提供的样本数据学习信号处理和机器学习部分。等掌握基础后再购买入门级设备。

### Q2: Python完全零基础可以学吗？
A: 可以！第4-5周有专门的Python快速入门，从零开始教学。

### Q3: 需要多长时间完成？
A: 建议每周投入10-12小时，8-10周完成全部内容。可以根据自己的节奏调整。

### Q4: 遇到问题怎么办？
A:
1. 查看 [docs/04_常见问题FAQ.md](docs/04_常见问题FAQ.md)
2. 在GitHub Issues提问
3. 联系导师

---

## 项目展示

完成项目后，你将拥有：
- ✅ 完整的代码仓库
- ✅ 实时手势识别演示视频
- ✅ 技术报告和文档
- ✅ 可运行的实时系统

这些成果可以用于：
- 大学申请作品集
- 科技竞赛参赛作品
- 个人技能展示

---

## 贡献指南

欢迎提交：
- 错误修正
- 代码改进
- 新的示例和教程
- 翻译（英文版）

请提交Pull Request或创建Issue。

---

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

---

## 致谢

感谢所有为EMG信号处理开源社区做出贡献的研究者和开发者。

---

## 联系方式

- 项目维护者：[导师姓名]
- 邮箱：[邮箱地址]
- GitHub Issues: [项目Issues页面]

---

**祝学习愉快！如有任何问题，随时联系！** 🚀
