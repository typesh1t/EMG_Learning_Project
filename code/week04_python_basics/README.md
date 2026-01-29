# 第4周：Python基础

## 学习目标
- 掌握Python基础语法（变量、循环、函数）
- 理解数据类型和结构（列表、字典）
- 学会使用NumPy进行数组操作
- 能够编写简单的数据处理脚本

## 前置要求
- 已安装Python 3.9+和Anaconda
- 已创建虚拟环境（参考GETTING_STARTED.md）

## 本周内容

### 1. Python语法基础
- `01_variables_and_types.py` - 变量和数据类型
- `02_control_flow.py` - 控制流（if/for/while）
- `03_functions.py` - 函数定义和使用
- `04_data_structures.py` - 列表、字典、元组

### 2. NumPy数组操作
- `05_numpy_basics.py` - NumPy基础操作
- `06_numpy_indexing.py` - 数组索引和切片
- `07_numpy_math.py` - 数学运算和统计

### 3. 综合练习
- `08_exercises.py` - 本周练习题
- `09_emg_data_exercise.py` - EMG数据处理练习

## 学习路径

### Day 1-2: Python基础语法
1. 运行01-04的脚本，理解基本概念
2. 修改示例代码，尝试不同的变量和操作
3. 完成每个脚本末尾的练习

### Day 3-4: NumPy数组
1. 学习05-07，掌握数组操作
2. 理解NumPy与Python列表的区别
3. 练习数组切片和数学运算

### Day 5: 综合练习
1. 完成08_exercises.py中的所有练习
2. 尝试09_emg_data_exercise.py处理简单EMG数据
3. 编写自己的小程序

## 重要概念

### Python基础
- **变量**: 存储数据的容器
- **函数**: 可重复使用的代码块
- **列表**: 有序的数据集合
- **字典**: 键值对的数据结构

### NumPy核心
- **数组**: 高效的多维数值数据结构
- **索引**: 访问数组中的特定元素
- **广播**: 不同形状数组之间的运算
- **向量化**: 避免Python循环的高效计算

## 作业

### 基础作业
1. 编写函数计算列表的最大值、最小值、平均值（不用内置函数）
2. 创建一个1000点的随机数组，统计大于0的数量
3. 实现数组归一化函数（转换到[0,1]范围）

### 进阶作业
1. 读取CSV文件，计算每列的统计量
2. 生成模拟EMG信号（随机数+正弦波）
3. 实现滑动窗口平均滤波器

## 参考资料
- Python官方教程: https://docs.python.org/3/tutorial/
- NumPy文档: https://numpy.org/doc/stable/
- 学习计划: docs/00_整体学习计划.md 第三阶段

## 常见问题

### Q: Python和其他语言有什么区别？
A: Python使用缩进表示代码块，语法简洁直观。不需要声明变量类型。

### Q: 为什么要用NumPy而不是Python列表？
A: NumPy数组运算速度快（C语言实现），支持向量化操作，内存效率高。

### Q: 如何查看函数的帮助文档？
A: 使用help()函数或在IPython中用?：
```python
help(np.array)
# 或
np.array?
```

### Q: 数组索引从0还是1开始？
A: 从0开始。第一个元素是arr[0]，最后一个是arr[-1]。

## 学习建议
- 边学边练，不要只看不动手
- 遇到错误不要慌，仔细阅读错误信息
- 使用print()查看中间结果
- 多写注释，帮助理解代码逻辑
- 参考官方文档，学习正确用法

## 学习资源

### Python基础教程
- [Python官方教程](https://docs.python.org/3/tutorial/) - 官方中文文档
- [NumPy官方文档](https://numpy.org/doc/stable/) - NumPy完整文档

### 在线文章（Python + EMG）
- [EMG肌电信号处理合集（含Python）](https://blog.csdn.net/YINTENAXIONGNAIER/article/details/134566397) - Python代码实例
- [Overview of Processing Techniques](https://arxiv.org/pdf/2304.04098) - EMG处理技术综述

### 本地教程
- 本周9个Python脚本（01-09）包含完整的教学内容
- Week 1-2的EMG理论文档

### 完整资源
查看 [../../docs/EMG学习资源汇总.md](../../docs/EMG学习资源汇总.md)

---

**重要**: 本周是后续学习的基础，务必打牢。如果遇到困难，不要跳过，花时间理解每个概念。
