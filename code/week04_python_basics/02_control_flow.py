#!/usr/bin/env python3
"""
Python基础 - 控制流
学习条件判断(if)、循环(for/while)和流程控制
"""

print("="*60)
print("第二课：控制流")
print("="*60)

# ============================================================
# 1. if条件判断
# ============================================================
print("\n【1. if条件判断】")
print("根据条件执行不同的代码块")

# 基本if语句
signal_amplitude = 0.8
threshold = 0.5

print(f"\n信号幅度: {signal_amplitude}, 阈值: {threshold}")

if signal_amplitude > threshold:
    print("检测到肌肉激活")

# if-else
print("\n示例：判断信号强度")
rms_value = 0.3

if rms_value > 0.5:
    print("强信号")
else:
    print("弱信号")

# if-elif-else（多条件）
print("\n示例：信号质量分级")
snr = 15  # 信噪比(dB)

if snr > 20:
    quality = "优秀"
elif snr > 10:
    quality = "良好"
elif snr > 5:
    quality = "一般"
else:
    quality = "较差"

print(f"SNR: {snr} dB -> 质量: {quality}")

# 复合条件
print("\n示例：复合条件判断")
frequency = 120  # Hz
amplitude = 0.7

if frequency > 20 and frequency < 500:
    print(f"频率{frequency}Hz在EMG有效范围内")

if amplitude > 0.5 and frequency > 50:
    print("检测到有效的肌肉激活信号")

# ============================================================
# 2. for循环
# ============================================================
print("\n【2. for循环】")
print("遍历序列（列表、范围等）中的每个元素")

# 基本for循环
print("\n示例：遍历通道")
channels = [0, 1, 2, 3]

for channel in channels:
    print(f"  处理通道 {channel}")

# range()函数
print("\n示例：使用range()生成数字序列")
for i in range(5):  # 0到4
    print(f"  迭代 {i}")

# range(start, stop, step)
print("\n示例：自定义范围")
for i in range(0, 10, 2):  # 0,2,4,6,8
    print(f"  偶数: {i}")

# 遍历字符串
print("\n示例：遍历字符串")
text = "EMG"
for char in text:
    print(f"  字符: {char}")

# enumerate() - 同时获取索引和值
print("\n示例：使用enumerate获取索引")
features = ['MAV', 'RMS', 'VAR', 'WL']
for idx, feature in enumerate(features):
    print(f"  {idx}: {feature}")

# ============================================================
# 3. while循环
# ============================================================
print("\n【3. while循环】")
print("当条件为True时持续执行")

# 基本while循环
print("\n示例：倒计时")
count = 5
while count > 0:
    print(f"  {count}...")
    count -= 1
print("  开始采集!")

# 注意：避免无限循环
print("\n注意事项：")
print("  while循环要确保条件最终变为False")
print("  否则会造成无限循环，程序卡住")

# 实际应用：处理数据直到满足条件
print("\n示例：数据处理")
sample_count = 0
target_samples = 5

while sample_count < target_samples:
    print(f"  采集样本 {sample_count + 1}")
    sample_count += 1
    # 实际应用中这里会有采集数据的代码

# ============================================================
# 4. break和continue
# ============================================================
print("\n【4. break和continue】")

# break - 立即退出循环
print("\n示例：找到第一个大于阈值的值就停止")
values = [0.1, 0.3, 0.6, 0.9, 1.2]
threshold = 0.5

for value in values:
    print(f"  检查: {value}")
    if value > threshold:
        print(f"  找到! {value} > {threshold}")
        break  # 退出循环
    print(f"  {value} <= {threshold}, 继续")

# continue - 跳过本次迭代，继续下一次
print("\n示例：跳过无效数据")
measurements = [0.5, -999, 0.8, -999, 1.2]  # -999表示无效数据

print("有效测量值:")
for measurement in measurements:
    if measurement == -999:
        continue  # 跳过无效值
    print(f"  {measurement}")

# ============================================================
# 5. 嵌套循环
# ============================================================
print("\n【5. 嵌套循环】")
print("循环内部可以再包含循环")

# 示例：多通道多试验
print("\n示例：处理多通道多试验数据")
num_channels = 3
num_trials = 2

for channel in range(num_channels):
    for trial in range(num_trials):
        print(f"  通道{channel}, 试验{trial}")

# 实际应用：生成数据矩阵
print("\n示例：创建3x3矩阵")
for row in range(3):
    row_data = []
    for col in range(3):
        value = row * 3 + col
        row_data.append(value)
    print(f"  第{row}行: {row_data}")

# ============================================================
# 6. 列表推导式
# ============================================================
print("\n【6. 列表推导式】")
print("创建列表的简洁方式")

# 传统方式
print("\n传统方式：创建平方数列表")
squares = []
for i in range(5):
    squares.append(i ** 2)
print(f"  {squares}")

# 列表推导式（更简洁）
print("\n列表推导式：")
squares = [i ** 2 for i in range(5)]
print(f"  {squares}")

# 带条件的列表推导式
print("\n示例：筛选偶数")
numbers = range(10)
evens = [n for n in numbers if n % 2 == 0]
print(f"  偶数: {evens}")

# 实际应用：处理EMG数据
print("\n示例：计算多个信号的RMS")
signals = [0.5, 0.8, 1.2, 0.3, 0.9]
rms_values = [sig ** 2 for sig in signals]
print(f"  原始: {signals}")
print(f"  平方: {rms_values}")

# ============================================================
# 7. pass语句
# ============================================================
print("\n【7. pass语句】")
print("占位符，什么都不做")

# 用于暂时不想实现的代码
print("\n示例：暂未实现的函数")
for i in range(3):
    if i == 1:
        pass  # 稍后实现
    else:
        print(f"  处理{i}")

# ============================================================
# 实践练习
# ============================================================
print("\n" + "="*60)
print("实践练习")
print("="*60)

print("\n练习1: 计算阶乘")
n = 5
factorial = 1
for i in range(1, n + 1):
    factorial *= i
print(f"{n}! = {factorial}")

print("\n练习2: 找出列表中的最大值和最小值")
data = [3.2, 1.5, 4.8, 2.1, 5.0, 0.9]
max_val = data[0]
min_val = data[0]

for value in data:
    if value > max_val:
        max_val = value
    if value < min_val:
        min_val = value

print(f"数据: {data}")
print(f"最大值: {max_val}")
print(f"最小值: {min_val}")

print("\n练习3: 计算平均值")
data = [0.5, 0.8, 1.2, 0.9, 1.1]
total = 0
count = 0

for value in data:
    total += value
    count += 1

average = total / count
print(f"数据: {data}")
print(f"平均值: {average:.2f}")

print("\n练习4: 计数和过滤")
values = [0.3, 0.6, 0.2, 0.9, 0.4, 0.8]
threshold = 0.5
count_above = 0
above_threshold = []

for value in values:
    if value > threshold:
        count_above += 1
        above_threshold.append(value)

print(f"阈值: {threshold}")
print(f"超过阈值的数量: {count_above}")
print(f"超过阈值的值: {above_threshold}")

print("\n练习5: 模拟信号检测")
signal = [0.1, 0.2, 0.8, 0.9, 1.1, 0.3, 0.2]
threshold = 0.5
state = "rest"  # rest或active

print("信号状态变化:")
for i, amplitude in enumerate(signal):
    if amplitude > threshold and state == "rest":
        print(f"  时刻{i}: 激活开始 (幅度={amplitude})")
        state = "active"
    elif amplitude <= threshold and state == "active":
        print(f"  时刻{i}: 返回静息 (幅度={amplitude})")
        state = "rest"

# ============================================================
# 课后作业
# ============================================================
print("\n" + "="*60)
print("课后作业")
print("="*60)

print("\n请完成以下作业:")
print("1. 使用for循环打印1-100之间所有3的倍数")
print("2. 使用while循环计算1+2+3+...+100的和")
print("3. 给定列表[1,2,3,4,5,6,7,8,9,10]，")
print("   使用列表推导式创建：")
print("   a. 所有元素的平方")
print("   b. 所有大于5的元素")
print("   c. 所有偶数的平方")
print("4. 模拟EMG数据采集：")
print("   生成10个随机数(0-1)，如果值>0.7就停止，")
print("   打印采集到的所有数据")
print("5. 使用嵌套循环打印九九乘法表前5行")

# ============================================================
# 总结
# ============================================================
print("\n" + "="*60)
print("本课总结")
print("="*60)

print("\n核心要点:")
print("1. if-elif-else: 根据条件执行不同代码")
print("2. for循环: 遍历序列中的每个元素")
print("3. while循环: 条件为True时重复执行")
print("4. break: 立即退出循环")
print("5. continue: 跳过本次迭代")
print("6. 列表推导式: 简洁地创建列表")
print("7. 嵌套循环: 循环中包含循环")

print("\n使用场景:")
print("- if: 判断信号是否超过阈值")
print("- for: 遍历所有EMG通道")
print("- while: 持续采集直到满足条件")
print("- break: 找到异常值后停止")
print("- continue: 跳过无效数据")

print("\n下一课: 03_functions.py - 函数定义和使用")
print("="*60)
