#!/usr/bin/env python3
"""
Python基础 - 函数
学习如何定义和使用函数，理解参数和返回值
"""

print("="*60)
print("第三课：函数")
print("="*60)

# ============================================================
# 1. 函数基础
# ============================================================
print("\n【1. 函数基础】")
print("函数是可重复使用的代码块")

# 定义最简单的函数
def greet():
    """打招呼函数"""
    print("Hello, EMG!")

# 调用函数
print("\n调用greet():")
greet()

# 带参数的函数
def greet_person(name):
    """打招呼，包含姓名"""
    print(f"Hello, {name}!")

print("\n调用greet_person():")
greet_person("Student")
greet_person("Researcher")

# 带返回值的函数
def calculate_square(number):
    """计算平方"""
    result = number ** 2
    return result

print("\n调用calculate_square():")
square = calculate_square(5)
print(f"5的平方 = {square}")

# ============================================================
# 2. 参数类型
# ============================================================
print("\n【2. 参数类型】")

# 位置参数
def calculate_rms(signal_values):
    """计算RMS值"""
    import math
    sum_squares = sum([x**2 for x in signal_values])
    rms = math.sqrt(sum_squares / len(signal_values))
    return rms

data = [0.5, 0.8, 1.2, 0.9]
rms_result = calculate_rms(data)
print(f"\n信号: {data}")
print(f"RMS: {rms_result:.4f}")

# 默认参数
def bandpass_filter(signal, lowcut=20, highcut=500):
    """带通滤波器（带默认参数）"""
    print(f"  滤波器范围: {lowcut}-{highcut} Hz")
    return signal  # 这里简化，实际会进行滤波

print("\n使用默认参数:")
signal = [1, 2, 3]
bandpass_filter(signal)

print("\n指定参数:")
bandpass_filter(signal, lowcut=30, highcut=400)

# 关键字参数
def process_signal(signal, method="filter", threshold=0.5):
    """处理信号（演示关键字参数）"""
    print(f"  方法: {method}, 阈值: {threshold}")

print("\n使用关键字参数:")
process_signal([1, 2, 3], method="normalize")
process_signal([1, 2, 3], threshold=0.8, method="threshold")

# 可变参数 (*args)
def calculate_average(*numbers):
    """计算任意数量数字的平均值"""
    if len(numbers) == 0:
        return 0
    return sum(numbers) / len(numbers)

print("\n可变参数示例:")
print(f"3个数的平均: {calculate_average(1, 2, 3)}")
print(f"5个数的平均: {calculate_average(10, 20, 30, 40, 50)}")

# 关键字可变参数 (**kwargs)
def configure_system(**settings):
    """配置系统（任意关键字参数）"""
    print("  系统配置:")
    for key, value in settings.items():
        print(f"    {key} = {value}")

print("\n关键字可变参数:")
configure_system(sampling_rate=1000, channels=4, duration=5)

# ============================================================
# 3. 返回值
# ============================================================
print("\n【3. 返回值】")

# 返回单个值
def get_max(values):
    """返回最大值"""
    return max(values)

print(f"\n最大值: {get_max([1, 5, 3, 9, 2])}")

# 返回多个值（元组）
def get_stats(values):
    """返回统计信息"""
    return min(values), max(values), sum(values) / len(values)

data = [1, 2, 3, 4, 5]
min_val, max_val, avg_val = get_stats(data)
print(f"\n数据: {data}")
print(f"最小值: {min_val}, 最大值: {max_val}, 平均值: {avg_val}")

# 返回字典
def analyze_signal(signal):
    """分析信号，返回字典"""
    return {
        'length': len(signal),
        'mean': sum(signal) / len(signal),
        'max': max(signal),
        'min': min(signal)
    }

signal = [0.5, 0.8, 1.2, 0.3]
stats = analyze_signal(signal)
print(f"\n信号统计: {stats}")

# 没有返回值（返回None）
def log_message(message):
    """只打印消息，不返回值"""
    print(f"[LOG] {message}")

result = log_message("处理完成")
print(f"返回值: {result}")  # None

# ============================================================
# 4. 函数文档字符串
# ============================================================
print("\n【4. 函数文档字符串】")
print("使用三引号添加函数说明")

def calculate_snr(signal, noise):
    """
    计算信噪比

    参数:
        signal: 信号功率
        noise: 噪声功率

    返回:
        float: 信噪比(dB)
    """
    import math
    if noise == 0:
        return float('inf')
    return 10 * math.log10(signal / noise)

# 查看函数文档
print("\n函数文档:")
print(calculate_snr.__doc__)

# ============================================================
# 5. 作用域
# ============================================================
print("\n【5. 作用域】")
print("变量的可见范围")

# 全局变量
SAMPLING_RATE = 1000  # 全局常量

def print_sampling_rate():
    """访问全局变量"""
    print(f"  采样率: {SAMPLING_RATE} Hz")

print("\n访问全局变量:")
print_sampling_rate()

# 局部变量
def calculate():
    """局部变量只在函数内有效"""
    local_var = 100  # 局部变量
    print(f"  函数内: {local_var}")

print("\n局部变量:")
calculate()
# print(local_var)  # 错误！函数外无法访问

# 修改全局变量（不推荐）
counter = 0

def increment():
    """修改全局变量"""
    global counter
    counter += 1

print("\n修改全局变量:")
print(f"初始: {counter}")
increment()
print(f"之后: {counter}")

# ============================================================
# 6. Lambda函数
# ============================================================
print("\n【6. Lambda函数】")
print("简单的匿名函数")

# 普通函数
def square(x):
    return x ** 2

# Lambda函数（等价）
square_lambda = lambda x: x ** 2

print(f"\n使用普通函数: {square(5)}")
print(f"使用lambda: {square_lambda(5)}")

# Lambda常用于sort、map、filter等
values = [3, 1, 4, 1, 5, 9, 2]
print(f"\n原始列表: {values}")
sorted_values = sorted(values, key=lambda x: x)
print(f"排序后: {sorted_values}")

# 实际应用：按绝对值排序
numbers = [-5, 2, -3, 7, -1]
sorted_abs = sorted(numbers, key=lambda x: abs(x))
print(f"\n按绝对值排序: {numbers} -> {sorted_abs}")

# ============================================================
# 实践练习
# ============================================================
print("\n" + "="*60)
print("实践练习")
print("="*60)

print("\n练习1: 温度转换函数")
def celsius_to_fahrenheit(celsius):
    """摄氏度转华氏度"""
    return celsius * 9/5 + 32

def fahrenheit_to_celsius(fahrenheit):
    """华氏度转摄氏度"""
    return (fahrenheit - 32) * 5/9

temp_c = 25
temp_f = celsius_to_fahrenheit(temp_c)
print(f"{temp_c}°C = {temp_f}°F")
print(f"{temp_f}°F = {fahrenheit_to_celsius(temp_f):.1f}°C")

print("\n练习2: 计算EMG特征")
def calculate_mav(signal):
    """计算平均绝对值"""
    return sum([abs(x) for x in signal]) / len(signal)

def calculate_peak(signal):
    """计算峰值"""
    return max([abs(x) for x in signal])

signal = [0.5, -0.8, 1.2, -0.3, 0.9]
print(f"信号: {signal}")
print(f"MAV: {calculate_mav(signal):.3f}")
print(f"峰值: {calculate_peak(signal):.3f}")

print("\n练习3: 数据验证函数")
def is_valid_signal(signal, min_length=100, max_amplitude=5.0):
    """
    检查信号是否有效

    参数:
        signal: 信号列表
        min_length: 最小长度
        max_amplitude: 最大幅度

    返回:
        bool: 是否有效
        str: 错误消息
    """
    if len(signal) < min_length:
        return False, f"信号太短: {len(signal)} < {min_length}"

    if max([abs(x) for x in signal]) > max_amplitude:
        return False, f"幅度过大: > {max_amplitude}"

    return True, "有效"

# 测试
test_signal = [0.5, 0.8, 1.2]
valid, message = is_valid_signal(test_signal)
print(f"信号长度{len(test_signal)}: {message}")

test_signal_long = [0.5] * 150
valid, message = is_valid_signal(test_signal_long)
print(f"信号长度{len(test_signal_long)}: {message}")

print("\n练习4: 过滤函数")
def filter_by_threshold(values, threshold=0.5):
    """筛选大于阈值的值"""
    return [v for v in values if v > threshold]

data = [0.3, 0.6, 0.2, 0.9, 0.4, 0.8]
filtered = filter_by_threshold(data)
print(f"原始: {data}")
print(f"筛选(>0.5): {filtered}")

filtered_custom = filter_by_threshold(data, threshold=0.7)
print(f"筛选(>0.7): {filtered_custom}")

# ============================================================
# 课后作业
# ============================================================
print("\n" + "="*60)
print("课后作业")
print("="*60)

print("\n请完成以下作业:")
print("1. 编写函数计算列表的标准差")
print("   公式: std = sqrt(sum((x - mean)^2) / n)")
print("2. 编写函数判断一个数是否为质数")
print("3. 编写函数接收任意数量的EMG通道数据，")
print("   返回每个通道的RMS值（字典格式）")
print("4. 编写函数模拟带通滤波器：")
print("   - 接收信号列表和频率范围")
print("   - 返回'滤波'后的信号（可以简单处理）")
print("   - 打印滤波参数")
print("5. 编写函数生成模拟EMG数据：")
print("   - 参数: 时长(秒), 采样率, 噪声水平")
print("   - 返回: 时间数组和信号数组（元组）")

# ============================================================
# 总结
# ============================================================
print("\n" + "="*60)
print("本课总结")
print("="*60)

print("\n核心要点:")
print("1. 函数用def定义，可以接收参数和返回值")
print("2. 参数类型：位置参数、默认参数、*args、**kwargs")
print("3. 返回值：可以返回单个值、多个值（元组）、字典等")
print("4. 文档字符串：使用三引号说明函数功能")
print("5. 作用域：全局变量 vs 局部变量")
print("6. Lambda函数：简短的匿名函数")

print("\n函数设计原则:")
print("- 单一职责：一个函数只做一件事")
print("- 有意义的函数名：动词开头，描述功能")
print("- 添加文档字符串：说明参数和返回值")
print("- 避免修改全局变量：使用参数和返回值")
print("- 合理使用默认参数：提高灵活性")

print("\n下一课: 04_data_structures.py - 数据结构")
print("="*60)
