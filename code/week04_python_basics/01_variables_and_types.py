#!/usr/bin/env python3
"""
Python基础 - 变量和数据类型
理解Python中的基本数据类型和变量使用
"""

print("="*60)
print("第一课：变量和数据类型")
print("="*60)

# ============================================================
# 1. 变量定义
# ============================================================
print("\n【1. 变量定义】")
print("变量是存储数据的容器，Python会自动推断类型")

# 数字类型
age = 17                    # 整数 (int)
height = 1.75               # 浮点数 (float)
temperature = -5.5          # 可以是负数

print(f"\nage = {age}, 类型: {type(age)}")
print(f"height = {height}, 类型: {type(height)}")
print(f"temperature = {temperature}, 类型: {type(temperature)}")

# 字符串类型
name = "EMG Project"        # 双引号
description = 'Signal Processing'  # 单引号都可以
multi_line = """这是一个
多行字符串"""

print(f"\nname = '{name}', 类型: {type(name)}")
print(f"长度: {len(name)} 个字符")

# 布尔类型
is_student = True
has_data = False

print(f"\nis_student = {is_student}, 类型: {type(is_student)}")
print(f"has_data = {has_data}, 类型: {type(has_data)}")

# ============================================================
# 2. 变量命名规则
# ============================================================
print("\n【2. 变量命名规则】")
print("规则:")
print("  - 只能包含字母、数字、下划线")
print("  - 不能以数字开头")
print("  - 区分大小写")
print("  - 不能使用Python关键字（如if, for, class等）")

# 好的命名示例
emg_signal = 100            # 使用下划线分隔单词（推荐）
channelCount = 4            # 驼峰命名法也可以
SAMPLING_RATE = 1000        # 常量用大写

# 不好的命名（虽然合法）
x = 100                     # 太简短，不知道是什么
data1 = 50                  # 数字后缀不直观
temp_var_for_calculation = 5  # 太长

print("\n推荐命名风格：")
print("  变量: emg_signal, channel_count")
print("  常量: SAMPLING_RATE, MAX_CHANNELS")
print("  函数: calculate_rms(), filter_signal()")

# ============================================================
# 3. 数字运算
# ============================================================
print("\n【3. 数字运算】")

a = 10
b = 3

print(f"\na = {a}, b = {b}")
print(f"加法: a + b = {a + b}")
print(f"减法: a - b = {a - b}")
print(f"乘法: a * b = {a * b}")
print(f"除法: a / b = {a / b}")          # 结果是浮点数
print(f"整除: a // b = {a // b}")        # 结果是整数
print(f"取余: a % b = {a % b}")
print(f"幂运算: a ** b = {a ** b}")      # a的b次方

# 复合赋值运算符
count = 0
count += 1      # 等同于 count = count + 1
print(f"\ncount += 1 后: {count}")

count *= 2      # 等同于 count = count * 2
print(f"count *= 2 后: {count}")

# ============================================================
# 4. 字符串操作
# ============================================================
print("\n【4. 字符串操作】")

text = "EMG Signal"

# 字符串方法
print(f"\n原始字符串: '{text}'")
print(f"转大写: '{text.upper()}'")
print(f"转小写: '{text.lower()}'")
print(f"替换: '{text.replace('Signal', 'Data')}'")
print(f"分割: {text.split()}")          # 返回列表

# 字符串拼接
first_name = "EMG"
last_name = "Project"
full_name = first_name + " " + last_name
print(f"\n拼接: '{full_name}'")

# 格式化字符串（推荐使用f-string）
value = 123.456
print(f"\n格式化示例:")
print(f"默认: {value}")
print(f"保留2位小数: {value:.2f}")
print(f"填充到10位: {value:10.2f}")

# ============================================================
# 5. 类型转换
# ============================================================
print("\n【5. 类型转换】")

# 数字转字符串
num = 100
num_str = str(num)
print(f"\nint转str: {num} -> '{num_str}' (类型: {type(num_str)})")

# 字符串转数字
str_num = "456"
num_from_str = int(str_num)
print(f"str转int: '{str_num}' -> {num_from_str} (类型: {type(num_from_str)})")

# 浮点数转整数
float_num = 3.7
int_num = int(float_num)        # 直接截断小数部分
print(f"float转int: {float_num} -> {int_num}")

# 整数转浮点数
int_val = 5
float_val = float(int_val)
print(f"int转float: {int_val} -> {float_val}")

# ============================================================
# 6. 布尔运算
# ============================================================
print("\n【6. 布尔运算】")

# 比较运算符
x = 10
y = 20

print(f"\nx = {x}, y = {y}")
print(f"x == y: {x == y}")      # 等于
print(f"x != y: {x != y}")      # 不等于
print(f"x < y: {x < y}")        # 小于
print(f"x > y: {x > y}")        # 大于
print(f"x <= y: {x <= y}")      # 小于等于
print(f"x >= y: {x >= y}")      # 大于等于

# 逻辑运算符
a = True
b = False

print(f"\na = {a}, b = {b}")
print(f"a and b: {a and b}")    # 与
print(f"a or b: {a or b}")      # 或
print(f"not a: {not a}")        # 非

# ============================================================
# 7. None类型
# ============================================================
print("\n【7. None类型】")
print("None表示'没有值'，类似其他语言的null")

result = None
print(f"result = {result}, 类型: {type(result)}")

# 检查None
if result is None:
    print("result是None")

# ============================================================
# 实践练习
# ============================================================
print("\n" + "="*60)
print("实践练习")
print("="*60)

print("\n练习1: 变量交换")
print("将两个变量的值互换")
a = 10
b = 20
print(f"交换前: a={a}, b={b}")
# 方法1: 使用临时变量
temp = a
a = b
b = temp
print(f"交换后: a={a}, b={b}")

# 方法2: Python的元组解包（更简洁）
a = 10
b = 20
a, b = b, a
print(f"使用元组解包: a={a}, b={b}")

print("\n练习2: 温度转换")
print("将摄氏度转换为华氏度")
celsius = 25
fahrenheit = celsius * 9/5 + 32
print(f"{celsius}°C = {fahrenheit}°F")

print("\n练习3: 字符串操作")
sentence = "EMG signal processing is fun"
print(f"原句: {sentence}")
print(f"单词数: {len(sentence.split())}")
print(f"字符数: {len(sentence)}")
print(f"全部大写: {sentence.upper()}")

print("\n练习4: 数学计算")
print("计算圆的面积和周长")
radius = 5
pi = 3.14159
area = pi * radius ** 2
circumference = 2 * pi * radius
print(f"半径: {radius}")
print(f"面积: {area:.2f}")
print(f"周长: {circumference:.2f}")

# ============================================================
# 课后作业
# ============================================================
print("\n" + "="*60)
print("课后作业")
print("="*60)

print("\n请完成以下作业（修改此脚本添加代码）:")
print("1. 定义变量存储你的姓名、年龄、身高")
print("2. 计算你的BMI（体重指数）= 体重(kg) / 身高(m)²")
print("3. 编写代码判断BMI是否在正常范围（18.5-24.9）")
print("4. 创建一个包含5个EMG特征名称的字符串变量，用逗号分隔")
print("5. 将该字符串分割成列表并打印")

print("\n提示：使用f-string格式化输出结果")

# ============================================================
# 总结
# ============================================================
print("\n" + "="*60)
print("本课总结")
print("="*60)

print("\n核心要点:")
print("1. 变量用于存储数据，Python自动推断类型")
print("2. 基本数据类型: int, float, str, bool, None")
print("3. 运算符: 算术运算、比较运算、逻辑运算")
print("4. 字符串可以拼接、格式化、调用各种方法")
print("5. 类型可以相互转换（int(), float(), str()）")
print("6. 使用有意义的变量名，遵循命名规范")

print("\n下一课: 02_control_flow.py - 控制流（if/for/while）")
print("="*60)
