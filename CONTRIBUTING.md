# 贡献指南

感谢您对EMG学习项目的关注！我们欢迎各种形式的贡献。

## 🤝 如何贡献

### 报告问题 (Issues)

如果您发现了bug或有新功能建议：

1. 在[GitHub Issues](../../issues)中搜索，确保问题未被报告
2. 创建新Issue，使用清晰的标题
3. 详细描述问题或建议
4. 如果是bug，提供重现步骤

**Issue标题示例：**
- `[Bug] 滤波器在处理短信号时出错`
- `[Feature] 希望添加小波变换特征`
- `[Docs] Week 3 README中的链接失效`

### 提交代码 (Pull Request)

1. **Fork项目**
   ```bash
   # 克隆你的fork
   git clone https://github.com/your-username/EMG_Learning_Project.git
   cd EMG_Learning_Project
   ```

2. **创建分支**
   ```bash
   git checkout -b feature/your-feature-name
   # 或
   git checkout -b fix/your-bug-fix
   ```

3. **进行修改**
   - 遵循代码风格指南（见下文）
   - 添加必要的注释和文档
   - 确保代码可以运行

4. **提交更改**
   ```bash
   git add .
   git commit -m "feat: 添加小波变换特征提取"
   ```

5. **推送并创建PR**
   ```bash
   git push origin feature/your-feature-name
   ```
   然后在GitHub上创建Pull Request

### 改进文档

文档改进也非常重要！

- 修正拼写或语法错误
- 添加更多示例
- 改进解释说明
- 翻译文档（如英文版本）

## 📝 代码风格指南

### Python代码规范

1. **PEP 8风格**
   ```python
   # 好的示例
   def calculate_rms(signal):
       """计算信号的RMS值"""
       return np.sqrt(np.mean(signal ** 2))

   # 避免
   def Calculate_RMS(Signal):
       return np.sqrt(np.mean(Signal**2))
   ```

2. **文档字符串**
   ```python
   def bandpass_filter(signal, lowcut, highcut, fs, order=4):
       """
       带通滤波器

       参数:
           signal: 输入信号
           lowcut: 低截止频率 (Hz)
           highcut: 高截止频率 (Hz)
           fs: 采样率 (Hz)
           order: 滤波器阶数，默认4

       返回:
           filtered: 滤波后的信号
       """
       # 实现代码...
   ```

3. **命名规范**
   - 变量和函数：`snake_case`
   - 类名：`PascalCase`
   - 常量：`UPPER_CASE`
   ```python
   # 好的示例
   sampling_rate = 1000
   class EMGFilters:
       MAX_FREQUENCY = 500
   ```

4. **类型提示（推荐）**
   ```python
   from typing import Tuple, Optional

   def extract_features(signal: np.ndarray,
                       fs: int = 1000) -> Tuple[dict, dict]:
       """提取特征"""
       # ...
   ```

### 注释规范

1. **中文注释**（本项目主要面向中文学习者）
   ```python
   # 计算信号的均方根值（RMS）
   rms = np.sqrt(np.mean(signal ** 2))
   ```

2. **关键步骤说明**
   ```python
   # 第一步：去除直流分量
   signal = signal - np.mean(signal)

   # 第二步：应用巴特沃斯带通滤波器
   filtered = butter_bandpass_filter(signal, 20, 500, fs)
   ```

3. **复杂算法解释**
   ```python
   # 使用Welch方法估计功率谱密度
   # 参数说明：
   #   - nperseg: 每段的长度，设为采样率的1/4以平衡频率分辨率和方差
   #   - noverlap: 重叠50%以提高估计精度
   frequencies, psd = welch(signal, fs=fs,
                           nperseg=fs//4,
                           noverlap=fs//8)
   ```

## 🧪 测试

在提交PR之前，请确保：

1. **代码可以运行**
   ```bash
   python your_script.py
   ```

2. **检查格式**（可选）
   ```bash
   # 使用black格式化代码
   black your_file.py

   # 使用flake8检查代码风格
   flake8 your_file.py
   ```

3. **运行示例**
   确保现有的示例脚本仍然可以正常运行

## 📚 文档贡献

### 改进教程

- 每周的README.md可以添加更多说明
- 添加更多的使用示例
- 创建新的Jupyter Notebook教程

### 添加资源链接

如果发现优质的学习资源：

1. 在[docs/学习资源快速索引.md](docs/学习资源快速索引.md)中添加
2. 提供资源链接、语言、难度、简介
3. 说明推荐理由

**格式示例：**
```markdown
### 新资源标题

- 平台：YouTube / B站 / CSDN / 知乎
- 链接：[URL]
- 语言：中文 / 英文
- 难度：⭐⭐⭐ (1-5星)
- 内容概要：简要说明
- 适合人群：初学者 / 进阶学习者 / 研究人员
- 推荐理由：为什么推荐这个资源
```

## 🎯 贡献方向

以下是一些我们特别欢迎的贡献方向：

### 高优先级
- [ ] 更多的代码示例和教程
- [ ] Jupyter Notebook交互式教程
- [ ] 单元测试（pytest）
- [ ] 英文版文档
- [ ] 视频教程（录屏讲解）

### 中优先级
- [ ] 更多的特征提取方法（小波变换、非线性特征等）
- [ ] 深度学习模型（LSTM、CNN等）
- [ ] 实时系统的完整实现
- [ ] GUI图形界面
- [ ] 更多的数据增强方法

### 低优先级
- [ ] Docker容器化
- [ ] Web应用（Flask/Django）
- [ ] 移动端应用
- [ ] 硬件集成（Arduino、树莓派）

## 💡 贡献建议

### 对于初学者

1. **从简单开始**
   - 修正文档中的错误
   - 改进注释
   - 添加使用示例

2. **选择感兴趣的领域**
   - 如果擅长数据可视化，改进图表
   - 如果喜欢写作，完善文档
   - 如果熟悉机器学习，添加新模型

### 对于进阶贡献者

1. **代码质量提升**
   - 重构代码以提高可读性
   - 优化性能
   - 添加错误处理

2. **功能扩展**
   - 实现新的算法
   - 添加新的工具
   - 开发教学示例

### 对于研究人员

1. **学术内容**
   - 添加最新的研究方法
   - 引用相关论文
   - 实现经典算法

2. **数据集和基准**
   - 添加公开数据集的使用示例
   - 提供性能基准测试
   - 对比不同方法

## 📋 提交检查清单

在提交PR之前，请确保：

- [ ] 代码遵循项目的风格规范
- [ ] 添加了必要的注释和文档字符串
- [ ] 代码可以正常运行
- [ ] 更新了相关的README（如果需要）
- [ ] 没有包含个人敏感信息
- [ ] 提交信息清晰明确
- [ ] 如果是大的更改，先创建Issue讨论

## 🎨 提交信息规范

使用语义化的提交信息：

```
<type>: <subject>

<body>

<footer>
```

**Type类型：**
- `feat`: 新功能
- `fix`: Bug修复
- `docs`: 文档改进
- `style`: 代码格式（不影响功能）
- `refactor`: 重构
- `test`: 添加测试
- `chore`: 构建/工具相关

**示例：**
```
feat: 添加小波变换特征提取方法

- 实现了离散小波变换
- 添加了小波包分解
- 更新了features.py文档

Closes #42
```

## 🙏 致谢

感谢每一位贡献者！您的贡献让这个项目变得更好。

贡献者列表将显示在项目README中。

## 📮 联系方式

有任何问题？

- 创建GitHub Issue
- 发送邮件到：[项目邮箱]
- 加入讨论组：[Discord/QQ群]

---

**再次感谢您的贡献！** 🎉
