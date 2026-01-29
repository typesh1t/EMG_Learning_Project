# 第5周：数据处理与可视化

## 学习目标
- 掌握使用Matplotlib绘制EMG信号
- 学会数据加载和保存（CSV、NPY格式）
- 理解数据预处理流程
- 能够制作专业的信号分析图表

## 本周内容

### 1. 数据加载
- 读取CSV格式的EMG数据
- 读取NumPy的.npy格式
- 处理多通道数据
- 数据格式转换

### 2. 数据可视化
- 时域信号绘制
- 频域谱图
- 多通道对比图
- 时频谱图（Spectrogram）

### 3. 数据处理
- 数据清洗（去除异常值）
- 数据分段
- 数据标注
- 批量处理

## 文件说明
- `data_loader.py` - 数据加载工具
- `visualization.py` - 可视化函数集
- `data_preprocessing.py` - 预处理工具
- `batch_processing.py` - 批量处理脚本
- `exercises.py` - 本周练习题

## 实践任务
1. 加载sample数据并绘制图表
2. 制作多通道对比图
3. 实现自动批量处理脚本
4. 完成exercises.py中的练习

## 作业
1. 绘制完整的EMG信号分析报告（包含时域、频域、统计信息）
2. 实现数据自动分段和标注
3. 批量处理多个受试者的数据

## 学习资源

### Python可视化
- [Matplotlib官方教程](https://matplotlib.org/stable/tutorials/index.html) - 官方完整教程
- [Matplotlib中文文档](https://www.matplotlib.org.cn/) - 中文参考
- 本地：`code/utils/chinese_font_config.py` - 中文字体配置

### EMG数据处理
- [EMG信号处理合集（含Python）](https://blog.csdn.net/YINTENAXIONGNAIER/article/details/134566397) - 可视化代码
- Week 3的signal_viewer.py - 参考实现

### 数据处理基础
- Week 4的NumPy教程 - 数组操作基础
- Pandas官方文档 - 数据处理工具

### 完整资源
查看 [../../docs/EMG学习资源汇总.md](../../docs/EMG学习资源汇总.md)

## 绘图技巧

### 制作专业图表的要点
1. **图表标题和标签**
   - 清晰的标题
   - 轴标签包含单位
   - 图例位置合适

2. **颜色和样式**
   - 使用对比明显的颜色
   - 线条粗细适当
   - 网格辅助阅读

3. **中文支持**
   - 使用chinese_font_config.py
   - 设置合适的中文字体

4. **多子图布局**
   - 使用subplots合理布局
   - 调整图表间距
   - 统一坐标轴范围

## 常见问题

### Q: 中文显示乱码怎么办？
A: 使用本项目的中文字体配置模块：
```python
from code.utils.chinese_font_config import setup_chinese_font
setup_chinese_font()
```

### Q: 如何保存高分辨率图片？
A: 使用savefig时设置dpi参数：
```python
plt.savefig('figure.png', dpi=300, bbox_inches='tight')
```

### Q: 数据量太大绘图很慢怎么办？
A: 考虑降采样或只绘制部分数据：
```python
signal_downsampled = signal[::10]  # 每10个点取1个
```

## 实用工具

本周完成后，你将拥有一套完整的数据处理工具：
- 数据加载器
- 可视化函数库
- 批量处理脚本
- 报告生成工具

这些工具将在后续周次中反复使用。
