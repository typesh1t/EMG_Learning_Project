# EMG信号特征分析详解

## 第一部分：时域特征分析

### 1.1 时域分析的意义

时域分析是直接在时间轴上观察和分析EMG信号的方法。它提供了信号幅度、能量、形态等直观信息，是EMG分析的基础。

#### 1.1.1 时域分析的优势
- 计算简单快速
- 物理意义明确
- 实时性好
- 适合在线处理

#### 1.1.2 时域分析的局限
- 难以反映频率信息
- 对噪声敏感
- 无法揭示信号的频率结构

### 1.2 常用时域特征

#### 1.2.1 平均绝对值（MAV - Mean Absolute Value）

**定义：**
```
MAV = (1/N) × Σ|x[i]|
```
其中N是采样点数，x[i]是第i个采样值。

**物理意义：**
- 表示信号的平均强度
- 与肌肉收缩力量正相关
- 单位：mV或μV

**特点：**
- 对信号幅度变化敏感
- 简单高效
- 广泛用于肌力评估

**计算实现：**
```python
def calculate_mav(signal):
    """
    计算平均绝对值

    参数:
        signal: 一维数组，EMG信号
    返回:
        mav: 平均绝对值
    """
    return np.mean(np.abs(signal))
```

**典型值范围：**
- 静息期：0.01-0.05 mV
- 轻度收缩：0.05-0.2 mV
- 中度收缩：0.2-0.5 mV
- 强力收缩：0.5-2.0 mV

**应用：**
- 肌肉疲劳监测（MAV随疲劳增加）
- 力量估计（MAV与力量相关）
- 激活检测（设置阈值判断）

#### 1.2.2 均方根值（RMS - Root Mean Square）

**定义：**
```
RMS = sqrt((1/N) × Σx[i]²)
```

**物理意义：**
- 表示信号的"有效值"
- 与信号功率直接相关
- 更适合描述信号能量

**特点：**
- 统计意义强
- 对异常值更敏感
- 常用于力量估计

**与MAV的区别：**
```
对于正弦信号：
MAV = (2/π) × A ≈ 0.637A
RMS = A/√2 ≈ 0.707A

对于EMG信号（近似高斯）：
RMS ≈ 1.11 × MAV
```

**计算实现：**
```python
def calculate_rms(signal):
    """
    计算均方根值

    参数:
        signal: 一维数组，EMG信号
    返回:
        rms: 均方根值
    """
    return np.sqrt(np.mean(signal ** 2))
```

**应用：**
- 肌力评估（RMS与力量线性相关性好）
- EMG归一化（通常归一化到MVC的RMS）
- 信号质量评估

**RMS-力量关系模型：**
```
简化线性模型（低强度）：
Force = k × RMS + b

非线性模型（全范围）：
Force = a × RMS^n
其中n通常在1.5-2之间
```

#### 1.2.3 方差（VAR - Variance）

**定义：**
```
VAR = (1/N) × Σ(x[i] - mean)²
```

**物理意义：**
- 信号偏离均值的程度
- 反映信号的变化幅度
- 与信号功率相关

**特点：**
- 与RMS相关：VAR ≈ RMS² （当均值≈0时）
- 反映信号的活跃程度
- 单位：mV²

**计算实现：**
```python
def calculate_variance(signal):
    """
    计算方差

    参数:
        signal: 一维数组，EMG信号
    返回:
        var: 方差
    """
    return np.var(signal)
```

**应用：**
- 信号分割（静息vs激活）
- 信号质量评估
- 与其他特征组合使用

#### 1.2.4 波形长度（WL - Waveform Length）

**定义：**
```
WL = Σ|x[i+1] - x[i]|
```

**物理意义：**
- 信号波形的总长度
- 反映信号的复杂度和频率
- 单位：mV

**特点：**
- 对信号频率和幅度都敏感
- 值越大表示信号变化越剧烈
- 计算简单

**计算实现：**
```python
def calculate_waveform_length(signal):
    """
    计算波形长度

    参数:
        signal: 一维数组，EMG信号
    返回:
        wl: 波形长度
    """
    return np.sum(np.abs(np.diff(signal)))
```

**影响因素：**
- 信号频率：高频信号WL大
- 信号幅度：幅度大WL也大
- 采样率：采样率越高WL越大

**归一化WL：**
```python
# 归一化到单位时间
WL_normalized = WL / duration_seconds

# 或归一化到采样点数
WL_per_sample = WL / N
```

**应用：**
- 肌肉激活复杂度评估
- 与频域特征配合使用
- 运动单位募集模式分析

#### 1.2.5 过零率（ZC - Zero Crossing）

**定义：**
信号穿过零点的次数。为了避免噪声影响，通常设置阈值：

```
ZC = count{|x[i]| > threshold and sign(x[i]) ≠ sign(x[i+1])}
```

**物理意义：**
- 反映信号的频率信息（时域表示）
- 过零次数多表示频率高
- 是频域特征在时域的粗略体现

**特点：**
- 计算非常快
- 提供粗略的频率信息
- 对噪声敏感，需要阈值

**阈值选择：**
```python
# 方法1：固定阈值（根据经验）
threshold = 0.01  # mV

# 方法2：自适应阈值（基于信号标准差）
threshold = 0.05 * np.std(signal)

# 方法3：基于噪声水平
threshold = 3 * noise_std
```

**计算实现：**
```python
def calculate_zero_crossing(signal, threshold=0.01):
    """
    计算过零率

    参数:
        signal: 一维数组，EMG信号
        threshold: 阈值，避免噪声影响
    返回:
        zc: 过零次数
        zcr: 过零率（Hz）
    """
    zc = 0
    for i in range(len(signal) - 1):
        cond1 = signal[i] > threshold and signal[i+1] < -threshold
        cond2 = signal[i] < -threshold and signal[i+1] > threshold
        if cond1 or cond2:
            zc += 1

    # 如果知道采样率，可以计算过零率（Hz）
    # zcr = zc / (len(signal) / sampling_rate)

    return zc
```

**与频率的关系：**
对于正弦信号 x(t) = A sin(2πft)：
```
ZC = 2f × duration
```

对于EMG信号：
```
ZC ≈ 2 × f_mean × duration
其中f_mean是信号的近似平均频率
```

**应用：**
- 快速频率估计
- 肌肉疲劳检测（ZC随疲劳减少）
- 实时处理（计算量小）

#### 1.2.6 斜率符号变化（SSC - Slope Sign Change）

**定义：**
信号斜率符号改变的次数，即波形转折点的数量。

```
SSC = count{(x[i] - x[i-1]) × (x[i] - x[i+1]) > threshold}
```

**物理意义：**
- 反映信号的频率信息
- 转折点多表示信号变化快
- 与信号的"锯齿"程度相关

**特点：**
- 对高频成分敏感
- 需要阈值减少噪声影响
- 与ZC互补

**阈值设置：**
```python
# 通常使用信号标准差的一小部分
threshold = 0.01 * np.std(signal)
```

**计算实现：**
```python
def calculate_slope_sign_change(signal, threshold=0.01):
    """
    计算斜率符号变化次数

    参数:
        signal: 一维数组，EMG信号
        threshold: 阈值
    返回:
        ssc: 斜率符号变化次数
    """
    ssc = 0
    for i in range(1, len(signal) - 1):
        slope_product = (signal[i] - signal[i-1]) * (signal[i] - signal[i+1])
        if slope_product > threshold:
            ssc += 1
    return ssc
```

**与ZC的区别：**
- ZC：信号穿过零线
- SSC：信号改变上升/下降方向
- SSC通常大于ZC
- 二者都反映频率，但SSC更敏感

**应用：**
- 频率信息提取
- 与ZC结合提高分类精度
- 运动模式识别

#### 1.2.7 积分EMG（IEMG - Integrated EMG）

**定义：**
```
IEMG = Σ|x[i]|
```

**物理意义：**
- 信号绝对值的累积
- 反映总体肌肉活动量
- 与MAV相关：IEMG = N × MAV

**特点：**
- 随时间窗口长度线性增长
- 适合分析整体活动水平
- 需要固定时间窗口比较

**计算实现：**
```python
def calculate_iemg(signal):
    """
    计算积分EMG

    参数:
        signal: 一维数组，EMG信号
    返回:
        iemg: 积分EMG
    """
    return np.sum(np.abs(signal))
```

**归一化IEMG：**
```python
# 归一化到单位时间（秒）
IEMG_per_second = IEMG / duration_seconds

# 或转换为MAV
MAV = IEMG / N
```

**应用：**
- 疲劳评估（累积肌肉活动）
- 工作负荷评估
- 能量消耗估计

#### 1.2.8 威尔逊幅度（WAMP - Willison Amplitude）

**定义：**
相邻样本差值超过阈值的次数：

```
WAMP = count{|x[i+1] - x[i]| > threshold}
```

**物理意义：**
- 信号突变的频繁程度
- 反映肌肉激活的动态特性
- 对高频成分敏感

**阈值选择：**
```python
# 典型值：信号范围的一小部分
threshold = 0.05  # 对于归一化到[-1,1]的信号
# 或基于统计
threshold = 0.1 * (np.max(signal) - np.min(signal))
```

**计算实现：**
```python
def calculate_wamp(signal, threshold=0.05):
    """
    计算威尔逊幅度

    参数:
        signal: 一维数组，EMG信号
        threshold: 阈值
    返回:
        wamp: 威尔逊幅度
    """
    diff = np.abs(np.diff(signal))
    return np.sum(diff > threshold)
```

**应用：**
- 肌肉激活模式分析
- 运动强度评估
- 手势识别

#### 1.2.9 对数检波器（LOG）

**定义：**
```
LOG = exp((1/N) × Σlog(|x[i]|))
```

**物理意义：**
- 基于指数平均的幅度估计
- 对小幅度信号更敏感
- 模拟肌肉力量感知

**特点：**
- 非线性特征
- 对低幅度信号敏感
- 与人体力量感知相关

**计算实现：**
```python
def calculate_log_detector(signal, epsilon=1e-10):
    """
    计算对数检波器

    参数:
        signal: 一维数组，EMG信号
        epsilon: 小值，防止log(0)
    返回:
        log_det: 对数检波器值
    """
    # 添加小值防止log(0)
    abs_signal = np.abs(signal) + epsilon
    return np.exp(np.mean(np.log(abs_signal)))
```

**应用：**
- 力量估计
- 与其他特征组合提高识别率

#### 1.2.10 差值绝对标准差（DASDV）

**定义：**
```
DASDV = sqrt((1/(N-1)) × Σ(x[i+1] - x[i])²)
```

**物理意义：**
- 相邻样本差值的标准差
- 反映信号的变化率
- 结合了频率和幅度信息

**计算实现：**
```python
def calculate_dasdv(signal):
    """
    计算差值绝对标准差

    参数:
        signal: 一维数组，EMG信号
    返回:
        dasdv: DASDV值
    """
    diff = np.diff(signal)
    return np.sqrt(np.mean(diff ** 2))
```

**应用：**
- 信号复杂度评估
- 与其他特征组合

### 1.3 时域特征的选择与组合

#### 1.3.1 特征相关性

不同时域特征之间存在相关性：

**高度相关（r > 0.9）：**
- MAV与RMS
- MAV与IEMG
- VAR与RMS²

**中度相关（0.5 < r < 0.9）：**
- WL与MAV
- ZC与SSC
- WAMP与WL

**低相关（r < 0.5）：**
- MAV与ZC
- RMS与SSC

**建议：**
- 选择互补特征（低相关）
- 避免冗余特征（高相关）
- 实验验证最佳组合

#### 1.3.2 常用特征组合

**基础组合（4个特征）：**
```python
features = [MAV, RMS, WL, ZC]
```
- 覆盖幅度和频率信息
- 计算简单
- 适合实时应用

**标准组合（6个特征）：**
```python
features = [MAV, RMS, VAR, WL, ZC, SSC]
```
- Hudgins特征集（1993）
- 广泛用于手势识别
- 性能与复杂度平衡

**扩展组合（8-10个特征）：**
```python
features = [MAV, RMS, VAR, WL, ZC, SSC, IEMG, WAMP, LOG, DASDV]
```
- 更全面的信息
- 可能存在冗余
- 需要特征选择

#### 1.3.3 特征选择方法

**方法1：基于相关性**
```python
from sklearn.feature_selection import SelectKBest, f_classif

# 选择top-k个特征
selector = SelectKBest(f_classif, k=6)
X_selected = selector.fit_transform(X, y)
```

**方法2：基于重要性（随机森林）**
```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(X, y)
importances = rf.feature_importances_
# 选择重要性高的特征
```

**方法3：递归特征消除**
```python
from sklearn.feature_selection import RFE

rfe = RFE(estimator=rf, n_features_to_select=6)
X_selected = rfe.fit_transform(X, y)
```

---

## 第二部分：频域特征分析

### 2.1 傅里叶变换基础

#### 2.1.1 为什么需要频域分析

**时域的局限性：**
- 难以直接观察频率成分
- 无法区分不同频率的信号叠加
- 难以分析周期性和震荡特性

**频域的优势：**
- 直观显示频率分布
- 揭示信号的频率结构
- 便于滤波器设计
- 疲劳分析的重要工具

#### 2.1.2 离散傅里叶变换（DFT）

**定义：**
```
X[k] = Σ(n=0 to N-1) x[n] × e^(-j2πkn/N)
```

其中：
- x[n]：时域信号
- X[k]：频域复数表示
- k：频率索引
- N：采样点数

**快速傅里叶变换（FFT）：**
- DFT的快速算法
- 复杂度从O(N²)降到O(N log N)
- Python中使用scipy.fft或numpy.fft

**实现：**
```python
from scipy.fft import fft, fftfreq

def compute_fft(signal, fs=1000):
    """
    计算FFT

    参数:
        signal: 时域信号
        fs: 采样率
    返回:
        frequencies: 频率轴
        magnitude: 幅度谱
        power: 功率谱
    """
    N = len(signal)

    # 计算FFT
    yf = fft(signal)

    # 频率轴（只取正频率部分）
    xf = fftfreq(N, 1/fs)[:N//2]

    # 幅度谱
    magnitude = 2.0/N * np.abs(yf[:N//2])

    # 功率谱
    power = magnitude ** 2

    return xf, magnitude, power
```

**理解频率分辨率：**
```
Δf = fs / N

例如：
fs = 1000 Hz
N = 1000 点
Δf = 1 Hz

如果需要更高频率分辨率：
- 增加采样点数N
- 或降低采样率（不推荐，会丢失高频信息）
```

### 2.2 功率谱分析

#### 2.2.1 功率谱密度（PSD）

**定义：**
功率谱密度描述信号功率在各个频率上的分布。

**计算方法：**

**方法1：周期图法**
```python
from scipy.signal import periodogram

def compute_psd_periodogram(signal, fs=1000):
    """
    使用周期图法计算PSD

    参数:
        signal: 时域信号
        fs: 采样率
    返回:
        frequencies: 频率
        psd: 功率谱密度
    """
    frequencies, psd = periodogram(signal, fs)
    return frequencies, psd
```

**方法2：Welch法（推荐）**
```python
from scipy.signal import welch

def compute_psd_welch(signal, fs=1000, nperseg=256):
    """
    使用Welch法计算PSD

    参数:
        signal: 时域信号
        fs: 采样率
        nperseg: 每段长度
    返回:
        frequencies: 频率
        psd: 功率谱密度
    """
    frequencies, psd = welch(signal, fs, nperseg=nperseg)
    return frequencies, psd
```

**Welch法的优势：**
- 将信号分段
- 每段计算周期图
- 取平均降低方差
- 结果更平滑稳定

#### 2.2.2 EMG频谱的典型形态

**正常EMG频谱：**
```
功率
 │
 │    ╱‾‾‾╲
 │   ╱     ╲___
 │  ╱           ╲___
 │ ╱                 ╲____
 └─────────────────────────> 频率(Hz)
   0   50  100 150 200  500
       ↑        ↑
     峰值    能量中心
```

特点：
- 20-150 Hz能量最集中
- 峰值频率通常在80-120 Hz
- 高频端逐渐衰减

**疲劳时的频谱变化：**
```
功率     未疲劳
 │       ╱‾╲
 │      ╱   ╲___
 │     ╱         ╲___
 │  疲劳╱              ╲___
 │   ╱                     ╲___
 └──────────────────────────────> 频率
    0    50   100  150  200  500
         ←─频谱左移
```

现象：
- 峰值频率左移（降低）
- 低频能量增加
- 高频能量减少
- MDF和MNF下降

### 2.3 常用频域特征

#### 2.3.1 平均频率（MNF - Mean Frequency）

**定义：**
```
MNF = Σ(f[i] × P[i]) / Σ(P[i])
```
其中f[i]是频率，P[i]是该频率的功率。

**物理意义：**
- 功率谱的"重心"
- 反映频谱的整体分布
- 单位：Hz

**特点：**
- 对频谱形状敏感
- 疲劳时MNF下降
- 用于疲劳检测

**计算实现：**
```python
def calculate_mean_frequency(signal, fs=1000):
    """
    计算平均频率

    参数:
        signal: 时域信号
        fs: 采样率
    返回:
        mnf: 平均频率(Hz)
    """
    # 计算功率谱
    freqs, psd = welch(signal, fs)

    # 限制在EMG有效范围
    mask = (freqs >= 20) & (freqs <= 500)
    freqs = freqs[mask]
    psd = psd[mask]

    # 计算MNF
    if np.sum(psd) > 0:
        mnf = np.sum(freqs * psd) / np.sum(psd)
    else:
        mnf = 0

    return mnf
```

**典型值：**
- 新鲜肌肉：100-120 Hz
- 疲劳肌肉：70-90 Hz
- 下降幅度：20-40%

**影响因素：**
- 肌肉疲劳（主要）
- 肌纤维类型
- 收缩强度
- 电极位置

#### 2.3.2 中值频率（MDF - Median Frequency）

**定义：**
将功率谱平分为两个相等面积的频率点。

```
Σ(0到MDF) P[i] = Σ(MDF到fmax) P[i] = 总功率/2
```

**物理意义：**
- 频谱的"中位数"
- 50%功率以下，50%功率以上
- 单位：Hz

**与MNF的区别：**
- MDF对异常值不敏感（鲁棒性好）
- MNF计算简单
- 二者趋势相同，数值接近

**计算实现：**
```python
def calculate_median_frequency(signal, fs=1000):
    """
    计算中值频率

    参数:
        signal: 时域信号
        fs: 采样率
    返回:
        mdf: 中值频率(Hz)
    """
    # 计算功率谱
    freqs, psd = welch(signal, fs)

    # 限制范围
    mask = (freqs >= 20) & (freqs <= 500)
    freqs = freqs[mask]
    psd = psd[mask]

    # 累积功率
    cumsum = np.cumsum(psd)
    total_power = cumsum[-1]

    # 找到50%功率点
    half_power = total_power / 2
    idx = np.where(cumsum >= half_power)[0]

    if len(idx) > 0:
        mdf = freqs[idx[0]]
    else:
        mdf = 0

    return mdf
```

**应用：**
- 疲劳监测（MDF下降表示疲劳）
- 康复评估（跟踪恢复进度）
- 运动强度控制

#### 2.3.3 峰值频率（Peak Frequency）

**定义：**
功率谱中功率最大的频率点。

```
f_peak = arg max P[f]
```

**物理意义：**
- 信号能量最集中的频率
- 反映主要的激活频率
- 单位：Hz

**计算实现：**
```python
def calculate_peak_frequency(signal, fs=1000):
    """
    计算峰值频率

    参数:
        signal: 时域信号
        fs: 采样率
    返回:
        peak_freq: 峰值频率(Hz)
        peak_power: 峰值功率
    """
    freqs, psd = welch(signal, fs)

    # 限制范围
    mask = (freqs >= 20) & (freqs <= 500)
    freqs = freqs[mask]
    psd = psd[mask]

    # 找峰值
    peak_idx = np.argmax(psd)
    peak_freq = freqs[peak_idx]
    peak_power = psd[peak_idx]

    return peak_freq, peak_power
```

**特点：**
- 直观但不稳定
- 对噪声敏感
- 可能有多个峰值

**应用：**
- 主导频率识别
- 配合MNF和MDF使用
- 频谱形态分析

#### 2.3.4 频谱矩（Spectral Moments）

**定义：**
频谱的各阶矩，类似统计学中的矩。

```
SM_n = Σ(f[i]^n × P[i]) / Σ(P[i])
```

**常用阶数：**
- SM0：总功率
- SM1：平均频率（= MNF）
- SM2：频率方差（反映频谱展宽）
- SM3：频谱偏度

**计算实现：**
```python
def calculate_spectral_moments(signal, fs=1000, max_order=3):
    """
    计算频谱矩

    参数:
        signal: 时域信号
        fs: 采样率
        max_order: 最大阶数
    返回:
        moments: 各阶矩的字典
    """
    freqs, psd = welch(signal, fs)
    mask = (freqs >= 20) & (freqs <= 500)
    freqs = freqs[mask]
    psd = psd[mask]

    moments = {}
    total_power = np.sum(psd)

    if total_power > 0:
        for n in range(max_order + 1):
            moments[f'SM{n}'] = np.sum((freqs ** n) * psd) / total_power
    else:
        moments = {f'SM{n}': 0 for n in range(max_order + 1)}

    return moments
```

**应用：**
- SM2：评估频谱集中度
- 高阶矩：捕捉频谱细节
- 与其他特征组合

#### 2.3.5 频率比（Frequency Ratio）

**定义：**
不同频段功率的比值。

```
Ratio = P_high / P_low
```

**常见划分：**
```python
# 方法1：简单划分
low_band: 20-100 Hz
high_band: 100-500 Hz

# 方法2：三段划分
low: 20-80 Hz
mid: 80-150 Hz
high: 150-500 Hz

# 方法3：自定义
根据应用需求定义
```

**计算实现：**
```python
def calculate_frequency_ratio(signal, fs=1000,
                             low_band=(20, 100),
                             high_band=(100, 500)):
    """
    计算频率比

    参数:
        signal: 时域信号
        fs: 采样率
        low_band: 低频段范围(Hz)
        high_band: 高频段范围(Hz)
    返回:
        ratio: 高频/低频功率比
    """
    freqs, psd = welch(signal, fs)

    # 低频段功率
    low_mask = (freqs >= low_band[0]) & (freqs < low_band[1])
    low_power = np.sum(psd[low_mask])

    # 高频段功率
    high_mask = (freqs >= high_band[0]) & (freqs < high_band[1])
    high_power = np.sum(psd[high_mask])

    # 计算比值
    if low_power > 0:
        ratio = high_power / low_power
    else:
        ratio = 0

    return ratio
```

**意义：**
- 疲劳时ratio下降（高频减少）
- 不同肌肉ratio不同
- 归一化效果

**应用：**
- 疲劳评估
- 肌肉分类
- 归一化特征

#### 2.3.6 总功率（Total Power）

**定义：**
整个频段的功率总和。

```
P_total = Σ P[i]
```

**计算实现：**
```python
def calculate_total_power(signal, fs=1000, freq_range=(20, 500)):
    """
    计算总功率

    参数:
        signal: 时域信号
        fs: 采样率
        freq_range: 频率范围
    返回:
        total_power: 总功率
    """
    freqs, psd = welch(signal, fs)

    mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    total_power = np.sum(psd[mask])

    return total_power
```

**与RMS的关系：**
```
P_total ≈ RMS²（Parseval定理）
时域能量 = 频域能量
```

**应用：**
- 信号能量评估
- 与时域特征互补验证
- 归一化其他频域特征

### 2.4 疲劳分析

#### 2.4.1 疲劳的频域表现

**生理机制：**
1. 肌肉疲劳时：
   - 乳酸堆积
   - 代谢物增加
   - 肌纤维传导速度下降

2. EMG变化：
   - 频谱向低频移动
   - MDF和MNF下降
   - 高频成分减少

**量化指标：**
```
疲劳指数 = (MDF_initial - MDF_fatigue) / MDF_initial × 100%

典型下降：20-40%
```

#### 2.4.2 疲劳检测实现

```python
def fatigue_analysis(signal_initial, signal_fatigue, fs=1000):
    """
    疲劳分析

    参数:
        signal_initial: 初始（未疲劳）信号
        signal_fatigue: 疲劳信号
        fs: 采样率
    返回:
        result: 疲劳分析结果字典
    """
    # 计算初始状态特征
    mnf_init = calculate_mean_frequency(signal_initial, fs)
    mdf_init = calculate_median_frequency(signal_initial, fs)

    # 计算疲劳状态特征
    mnf_fatigue = calculate_mean_frequency(signal_fatigue, fs)
    mdf_fatigue = calculate_median_frequency(signal_fatigue, fs)

    # 计算变化率
    mnf_change = (mnf_init - mnf_fatigue) / mnf_init * 100
    mdf_change = (mdf_init - mdf_fatigue) / mdf_init * 100

    result = {
        'MNF_initial': mnf_init,
        'MNF_fatigue': mnf_fatigue,
        'MNF_change_%': mnf_change,
        'MDF_initial': mdf_init,
        'MDF_fatigue': mdf_fatigue,
        'MDF_change_%': mdf_change,
        'is_fatigued': mdf_change > 15  # 阈值：15%
    }

    return result
```

---

## 第三部分：时频分析

### 3.1 短时傅里叶变换（STFT）

**目的：**
同时观察信号的时间和频率变化。

**原理：**
```
STFT(t, f) = Σ x[n] × w[n-t] × e^(-j2πfn)
```

其中w[n]是窗函数。

**实现：**
```python
from scipy.signal import spectrogram

def compute_spectrogram(signal, fs=1000, nperseg=256):
    """
    计算时频谱图

    参数:
        signal: 时域信号
        fs: 采样率
        nperseg: 每段长度
    返回:
        t: 时间轴
        f: 频率轴
        Sxx: 时频谱
    """
    f, t, Sxx = spectrogram(signal, fs, nperseg=nperseg)
    return t, f, Sxx
```

**可视化：**
```python
def plot_spectrogram(signal, fs=1000):
    """
    绘制时频谱图
    """
    t, f, Sxx = compute_spectrogram(signal, fs)

    plt.figure(figsize=(12, 6))
    plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
    plt.ylabel('频率 (Hz)')
    plt.xlabel('时间 (秒)')
    plt.title('EMG时频谱图')
    plt.colorbar(label='功率/频率 (dB/Hz)')
    plt.ylim(0, 300)
    plt.show()
```

**应用：**
- 观察疲劳过程中频谱变化
- 分析瞬态事件
- 非平稳信号分析

---

## 总结

### 关键要点

**时域特征：**
- 计算简单快速
- 适合实时应用
- MAV和RMS最常用
- 组合使用效果好

**频域特征：**
- 揭示频率结构
- 疲劳分析必备
- MDF和MNF是核心
- 计算相对复杂

**特征选择：**
- 时域+频域组合
- 考虑相关性
- 实验验证
- 平衡性能和复杂度

### 实践建议

1. **初学者**：先掌握MAV、RMS、MNF、MDF
2. **手势识别**：时域特征为主，6-8个即可
3. **疲劳分析**：频域特征必需，追踪MDF变化
4. **研究项目**：全面提取，再进行特征选择

下一章将学习如何使用Python实现这些特征提取，并应用于实际数据。
