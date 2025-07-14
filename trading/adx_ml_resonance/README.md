# ADX-ML自适应SuperTrend共振扫描器

## 概述

这个扫描器结合了ADX（平均方向指数）和机器学习自适应SuperTrend指标，通过两者的共振来识别高质量的买卖信号。

## 核心理念

**共振信号** = ADX强趋势 + ML SuperTrend方向变化

- **ADX**: 衡量趋势强度，过滤掉震荡市场中的假信号
- **ML自适应SuperTrend**: 基于机器学习的动态SuperTrend，能适应不同的市场波动情况
- **共振**: 两个指标同时发出信号，增加信号的可靠性

## 策略类型

### 1. AdxMlResonanceStrategy (基础共振策略)
- **信号条件**: ADX > 阈值 + ML SuperTrend方向转换
- **特点**: 简单直接，信号较多
- **适用**: 活跃交易者

### 2. AdvancedAdxMlResonanceStrategy (高级共振策略)  
- **信号条件**: ADX > 阈值 + DI交叉 + ML SuperTrend方向转换
- **特点**: 三重确认，信号质量更高
- **适用**: 寻求高精度信号的交易者

### 3. ConservativeResonanceStrategy (保守共振策略)
- **信号条件**: 更高的ADX阈值 + 价格趋势确认 + 只做多
- **特点**: 风险控制严格，只捕捉最明确的机会
- **适用**: 保守投资者

## 文件结构

```
trading/adx_ml_resonance/
├── indicators.py          # 共振指标计算
├── strategies.py          # 三种共振策略
├── scan.py               # 股票扫描器
└── README.md             # 说明文档
```

## 使用方法

### 1. 运行单策略扫描

```python
# 修改 scan.py 中的配置
USE_MULTI_STRATEGY = False
SINGLE_STRATEGY = AdxMlResonanceStrategy

# 运行扫描
python trading/adx_ml_resonance/scan.py
```

### 2. 运行多策略扫描

```python
# 修改 scan.py 中的配置  
USE_MULTI_STRATEGY = True

# 运行扫描
python trading/adx_ml_resonance/scan.py
```

### 3. 自定义参数

可以在strategies.py中调整以下参数：

```python
# ADX参数
adx_length = 14        # ADX计算周期
adx_threshold = 25     # ADX强趋势阈值

# ML SuperTrend参数
ml_atr_len = 10        # ATR周期
ml_fact = 3.0          # SuperTrend倍数
training_data_period = 100  # ML训练周期

# 风控参数
stop_loss_pct = 0.05   # 止损百分比
take_profit_pct = 0.15 # 止盈百分比
max_holding_days = 10  # 最大持仓天数
```

## 输出结果

扫描器会在`results/`目录生成以下文件：

### 单策略模式:
- `adx_ml_买入信号.ini` - 买入信号股票
- `adx_ml_卖出信号.ini` - 卖出信号股票

### 多策略模式:
```
results/
├── 基础共振/
│   ├── 基础共振_买入信号.ini
│   └── 基础共振_卖出信号.ini
├── 高级共振/
│   ├── 高级共振_买入信号.ini  
│   └── 高级共振_卖出信号.ini
├── 保守共振/
│   ├── 保守共振_买入信号.ini
│   └── 保守共振_卖出信号.ini
├── 合并_买入信号.ini    # 所有策略买入信号合集
└── 合并_卖出信号.ini    # 所有策略卖出信号合集
```

## 信号质量

### 高级共振策略 > 基础共振策略 > 保守共振策略

- **高级共振**: 信号最精确，但数量较少
- **基础共振**: 平衡信号质量和数量
- **保守共振**: 信号数量最少，但风险最低

## 注意事项

1. **数据要求**: 每只股票需要至少200个交易日的数据
2. **计算资源**: 使用多核并行处理，计算较耗时
3. **参数调优**: 可根据不同市场环境调整参数
4. **风险管理**: 所有策略都包含止损和止盈设置

## 技术指标说明

### ADX (Average Directional Index)
- **范围**: 0-100
- **强趋势**: ADX > 25
- **弱趋势**: ADX < 20  
- **作用**: 过滤震荡市场，确保在趋势中交易

### ML自适应SuperTrend
- **特点**: 基于K-means聚类动态调整ATR参数
- **优势**: 能适应不同波动率环境
- **信号**: 方向从-1变为1(看涨)，从1变为-1(看跌)

### DI交叉 (仅高级策略使用)
- **+DI上穿-DI**: 看涨信号
- **-DI上穿+DI**: 看跌信号
- **作用**: 提供额外的方向确认

## 回测建议

使用backtesting.py框架对策略进行回测：

```python
from backtesting import Backtest
from trading.adx_ml_resonance.strategies import AdxMlResonanceStrategy

bt = Backtest(data, AdxMlResonanceStrategy)
stats = bt.run()
print(stats)
bt.plot()
``` 