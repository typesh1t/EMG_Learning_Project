#!/usr/bin/env python3
"""
Python基础 - 数据结构
学习列表、元组、字典、集合等数据结构
"""

print("="*60)
print("第四课：数据结构")
print("="*60)

# ============================================================
# 1. 列表 (List)
# ============================================================
print("\n【1. 列表 (List)】")
print("列表是可变的有序集合，用方括号[]表示")

# 创建列表
channels = [0, 1, 2, 3]
features = ['MAV', 'RMS', 'VAR', 'WL']
mixed = [1, 'EMG', 3.14, True]

print(f"\nchannels = {channels}")
print(f"features = {features}")
print(f"mixed = {mixed}")

# 列表索引
print("\n列表索引:")
print(f"  第一个元素: features[0] = {features[0]}")
print(f"  最后一个元素: features[-1] = {features[-1]}")
print(f"  第二个元素: features[1] = {features[1]}")

# 列表切片
print("\n列表切片:")
numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
print(f"  numbers = {numbers}")
print(f"  numbers[2:5] = {numbers[2:5]}")     # 索引2到4
print(f"  numbers[:3] = {numbers[:3]}")       # 前3个
print(f"  numbers[7:] = {numbers[7:]}")       # 从索引7到末尾
print(f"  numbers[::2] = {numbers[::2]}")     # 每隔一个
print(f"  numbers[::-1] = {numbers[::-1]}")   # 反转

# 列表方法
print("\n列表方法:")
emg_channels = [0, 1, 2]
print(f"  原始列表: {emg_channels}")

emg_channels.append(3)
print(f"  append(3) 后: {emg_channels}")

emg_channels.insert(1, 10)
print(f"  insert(1, 10) 后: {emg_channels}")

emg_channels.remove(10)
print(f"  remove(10) 后: {emg_channels}")

last = emg_channels.pop()
print(f"  pop() 返回 {last}，剩余: {emg_channels}")

emg_channels.extend([4, 5])
print(f"  extend([4, 5]) 后: {emg_channels}")

# 列表操作
print("\n列表操作:")
list1 = [1, 2, 3]
list2 = [4, 5, 6]
combined = list1 + list2
print(f"  {list1} + {list2} = {combined}")
print(f"  {list1} * 3 = {list1 * 3}")
print(f"  3 in {list1}: {3 in list1}")
print(f"  长度 len({list1}): {len(list1)}")

# ============================================================
# 2. 元组 (Tuple)
# ============================================================
print("\n【2. 元组 (Tuple)】")
print("元组是不可变的有序集合，用圆括号()表示")

# 创建元组
coordinates = (10, 20)
emg_params = (1000, 4, 5.0)  # (采样率, 通道数, 时长)
single_element = (42,)  # 单元素元组需要逗号

print(f"\ncoordinates = {coordinates}")
print(f"emg_params = {emg_params}")
print(f"single_element = {single_element}")

# 元组解包
sampling_rate, n_channels, duration = emg_params
print(f"\n解包: fs={sampling_rate}, ch={n_channels}, dur={duration}")

# 元组索引
print(f"\n元组索引:")
print(f"  coordinates[0] = {coordinates[0]}")
print(f"  coordinates[1] = {coordinates[1]}")

# 元组不可变
print("\n注意: 元组创建后不能修改")
print("  coordinates[0] = 100  # 这会报错")

# 元组的用途
print("\n元组的用途:")
print("  1. 返回多个值")
print("  2. 作为字典的键")
print("  3. 保护数据不被修改")

# ============================================================
# 3. 字典 (Dictionary)
# ============================================================
print("\n【3. 字典 (Dictionary)】")
print("字典是键值对的集合，用大括号{}表示")

# 创建字典
emg_config = {
    'sampling_rate': 1000,
    'channels': 4,
    'duration': 5.0,
    'gestures': ['rest', 'fist', 'open']
}

print(f"\nemg_config = {emg_config}")

# 访问字典
print("\n访问字典:")
print(f"  采样率: {emg_config['sampling_rate']}")
print(f"  通道数: {emg_config['channels']}")

# 使用get方法（更安全）
print(f"  get('duration'): {emg_config.get('duration')}")
print(f"  get('filter'): {emg_config.get('filter', 'None')}")  # 不存在时返回默认值

# 修改字典
emg_config['channels'] = 8
emg_config['filter'] = 'bandpass'
print(f"\n修改后: channels={emg_config['channels']}, filter={emg_config.get('filter')}")

# 删除键
del emg_config['filter']
print(f"删除filter后的键: {list(emg_config.keys())}")

# 字典方法
print("\n字典方法:")
print(f"  keys(): {list(emg_config.keys())}")
print(f"  values(): {list(emg_config.values())}")
print(f"  items(): {list(emg_config.items())}")

# 遍历字典
print("\n遍历字典:")
for key, value in emg_config.items():
    print(f"  {key}: {value}")

# ============================================================
# 4. 集合 (Set)
# ============================================================
print("\n【4. 集合 (Set)】")
print("集合是无序的、不重复元素的集合")

# 创建集合
channels_set = {0, 1, 2, 3, 2, 1}  # 重复元素会被自动去除
print(f"\nchannels_set = {channels_set}")

gestures1 = {'rest', 'fist', 'open'}
gestures2 = {'fist', 'open', 'point'}

print(f"gestures1 = {gestures1}")
print(f"gestures2 = {gestures2}")

# 集合操作
print("\n集合操作:")
print(f"  并集: {gestures1 | gestures2}")
print(f"  交集: {gestures1 & gestures2}")
print(f"  差集: {gestures1 - gestures2}")
print(f"  对称差: {gestures1 ^ gestures2}")

# 集合方法
print("\n集合方法:")
my_set = {1, 2, 3}
my_set.add(4)
print(f"  add(4): {my_set}")

my_set.remove(2)
print(f"  remove(2): {my_set}")

print(f"  3 in my_set: {3 in my_set}")

# ============================================================
# 5. 数据结构选择
# ============================================================
print("\n【5. 数据结构选择指南】")
print("\n什么时候用什么:")
print("  列表 []: 需要有序、可修改的集合")
print("    - 存储EMG数据点")
print("    - 保存特征列表")
print("  ")
print("  元组 (): 需要不可变的有序集合")
print("    - 函数返回多个值")
print("    - 配置参数（保护不被修改）")
print("  ")
print("  字典 {}: 需要键值对映射")
print("    - 配置文件")
print("    - 存储实验参数")
print("  ")
print("  集合 {}: 需要去重或集合运算")
print("    - 统计不同的手势类型")
print("    - 查找交集、并集")

# ============================================================
# 6. 嵌套数据结构
# ============================================================
print("\n【6. 嵌套数据结构】")
print("数据结构可以相互嵌套")

# 列表的列表
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]
print(f"\n矩阵:")
for row in matrix:
    print(f"  {row}")

print(f"\n访问元素: matrix[1][2] = {matrix[1][2]}")

# 字典的列表
experiments = [
    {'subject': 1, 'gesture': 'fist', 'accuracy': 0.95},
    {'subject': 2, 'gesture': 'open', 'accuracy': 0.89},
    {'subject': 3, 'gesture': 'rest', 'accuracy': 0.99}
]

print("\n实验数据:")
for exp in experiments:
    print(f"  受试者{exp['subject']}: {exp['gesture']} - {exp['accuracy']:.2%}")

# 列表的字典
subject_data = {
    'subject_1': [0.95, 0.92, 0.88],
    'subject_2': [0.89, 0.91, 0.85],
    'subject_3': [0.99, 0.97, 0.96]
}

print("\n受试者准确率:")
for subject, accuracies in subject_data.items():
    avg = sum(accuracies) / len(accuracies)
    print(f"  {subject}: {accuracies} -> 平均: {avg:.2%}")

# ============================================================
# 7. 列表推导式（高级）
# ============================================================
print("\n【7. 列表推导式】")
print("用简洁的方式创建列表")

# 基本列表推导式
squares = [x**2 for x in range(10)]
print(f"\n平方数: {squares}")

# 带条件的列表推导式
even_squares = [x**2 for x in range(10) if x % 2 == 0]
print(f"偶数的平方: {even_squares}")

# 字符串处理
features = ['MAV', 'RMS', 'VAR']
lowercase = [f.lower() for f in features]
print(f"\n转小写: {lowercase}")

# 二维列表推导式
matrix = [[i*3 + j for j in range(3)] for i in range(3)]
print(f"\n3x3矩阵:")
for row in matrix:
    print(f"  {row}")

# 字典推导式
feature_names = ['MAV', 'RMS', 'VAR']
feature_values = [0.5, 0.7, 0.3]
feature_dict = {name: value for name, value in zip(feature_names, feature_values)}
print(f"\n特征字典: {feature_dict}")

# ============================================================
# 实践练习
# ============================================================
print("\n" + "="*60)
print("实践练习")
print("="*60)

print("\n练习1: 统计列表中的最大、最小、平均值")
data = [0.5, 0.8, 1.2, 0.3, 0.9, 1.5, 0.7]
print(f"数据: {data}")
print(f"最大值: {max(data)}")
print(f"最小值: {min(data)}")
print(f"平均值: {sum(data) / len(data):.2f}")
print(f"排序: {sorted(data)}")

print("\n练习2: 使用字典存储EMG特征")
emg_features = {
    'MAV': 0.52,
    'RMS': 0.68,
    'VAR': 0.31,
    'WL': 12.5
}
print("EMG特征:")
for name, value in emg_features.items():
    print(f"  {name}: {value:.2f}")

print("\n练习3: 去除重复的手势标签")
labels = ['fist', 'open', 'fist', 'rest', 'open', 'fist', 'rest']
unique_labels = list(set(labels))
print(f"原始标签: {labels}")
print(f"唯一标签: {unique_labels}")
print(f"统计: {len(labels)}个标签，{len(unique_labels)}种手势")

print("\n练习4: 嵌套字典表示实验配置")
experiment = {
    'name': 'EMG手势识别',
    'parameters': {
        'sampling_rate': 1000,
        'channels': 4,
        'duration': 5.0
    },
    'subjects': ['S1', 'S2', 'S3'],
    'gestures': ['rest', 'fist', 'open']
}

print(f"实验名称: {experiment['name']}")
print(f"采样率: {experiment['parameters']['sampling_rate']} Hz")
print(f"受试者: {', '.join(experiment['subjects'])}")
print(f"手势: {', '.join(experiment['gestures'])}")

print("\n练习5: 使用列表推导式处理数据")
raw_data = [0.1, 0.5, 0.9, 1.3, 0.7]
threshold = 0.5

# 归一化到[0, 1]
normalized = [(x - min(raw_data)) / (max(raw_data) - min(raw_data)) for x in raw_data]
print(f"\n原始数据: {raw_data}")
print(f"归一化: {[f'{x:.2f}' for x in normalized]}")

# 筛选超过阈值的数据
above_threshold = [x for x in raw_data if x > threshold]
print(f"阈值: {threshold}")
print(f"超过阈值: {above_threshold}")

# ============================================================
# 课后作业
# ============================================================
print("\n" + "="*60)
print("课后作业")
print("="*60)

print("\n请完成以下作业:")
print("1. 创建一个列表存储10个随机数，")
print("   编写代码找出其中的最大值、最小值和平均值（不用内置函数）")
print("2. 创建字典存储3个受试者的信息（姓名、年龄、性别）")
print("3. 给定两个列表：")
print("   gestures1 = ['rest', 'fist', 'open', 'point']")
print("   gestures2 = ['fist', 'open', 'pinch', 'wave']")
print("   找出它们的交集、并集和差集")
print("4. 创建一个嵌套字典表示一个EMG数据集：")
print("   包含受试者ID、手势类型、试验编号、准确率")
print("5. 使用列表推导式:")
print("   a. 创建1-100中所有能被3整除的数")
print("   b. 将列表['MAV', 'RMS', 'VAR']转换为小写")
print("   c. 从列表[1,2,3,4,5]创建平方值列表")

# ============================================================
# 总结
# ============================================================
print("\n" + "="*60)
print("本课总结")
print("="*60)

print("\n核心要点:")
print("1. 列表[]: 有序、可变、允许重复")
print("2. 元组(): 有序、不可变、允许重复")
print("3. 字典{}: 键值对、无序（Python 3.7+保持插入顺序）")
print("4. 集合{}: 无序、不重复、用于集合运算")
print("5. 可以嵌套使用各种数据结构")
print("6. 列表推导式提供简洁的创建方式")

print("\n使用场景:")
print("- 列表: 存储EMG信号、特征序列")
print("- 元组: 返回多个值、不可变配置")
print("- 字典: 实验参数、特征名称-值映射")
print("- 集合: 统计唯一标签、查找重复")

print("\n下一课: 05_numpy_basics.py - NumPy数组基础")
print("="*60)
