# Reverse RSI策略

基于Pine Script的"Reverse RSI Signals"指标的Python实现，包含完整的回测和扫描功能。

## 策略概述

Reverse RSI策略结合了以下几个核心组件：

1. **RSI反向价格计算**: 根据RSI水平（30/70）反向计算对应的价格水平
2. **SuperTrend趋势确认**: 使用SuperTrend指标确认趋势方向
3. **RSI发散检测**: 检测价格与RSI之间的看涨/看跌发散
4. **风险管理**: 包含止损、止盈和最大持仓时间限制

## 文件结构

```
reverse_rsi/
├── indicators.py           # 核心指标实现
├── strategies.py           # 策略类实现
├── reverse_rsi_backtest.py # 参数优化回测脚本
├── scan.py                 # 股票扫描脚本
├── test_strategy.py        # 策略测试脚本
├── reverse_rsi.pine        # 原始Pine Script代码
└── README.md               # 本文件
```

## 策略信号

### 买入信号
- 价格向上突破超卖价格水平(RSI 30对应价格) + SuperTrend看涨
- RSI看涨发散 + SuperTrend看涨  
- SuperTrend从看跌转为看涨

### 卖出信号
- 价格向下突破超买价格水平(RSI 70对应价格) + SuperTrend看跌
- RSI看跌发散 + SuperTrend看跌
- SuperTrend从看涨转为看跌

## 使用方法

### 1. 策略测试

首先运行测试脚本确保策略正常工作：

```bash
cd trading/reverse_rsi
python test_strategy.py
```

### 2. 参数优化回测

运行完整的参数优化回测：

```bash
python reverse_rsi_backtest.py
```

回测结果将保存在 `../results/` 目录中，包含：
- 最优参数组合
- 详细的回测统计
- 性能图表

### 3. 股票扫描

扫描所有股票寻找当前的买入/卖出信号：

```bash
python scan.py
```

扫描结果将保存为同花顺格式的INI文件：
- `results/Reverse_RSI_买入信号.ini`
- `results/Reverse_RSI_卖出信号.ini`

## 策略参数

### 核心参数
- `rsi_length`: RSI计算周期 (默认: 14)
- `smooth_bands`: 是否平滑价格带 (默认: True)
- `st_factor`: SuperTrend因子 (默认: 2.4)
- `st_atr_len`: SuperTrend ATR周期 (默认: 10)
- `div_lookback`: 发散检测回看周期 (默认: 3)

### 信号控制参数
- `use_price_breakout`: 是否使用价格突破信号 (默认: True)
- `use_divergence`: 是否使用发散信号 (默认: True)
- `use_trend_filter`: 是否使用趋势过滤 (默认: True)

### 风险管理参数
- `stop_loss_pct`: 止损百分比 (默认: 0.05 = 5%)
- `take_profit_pct`: 止盈百分比 (默认: 0.12 = 12%)
- `max_holding_days`: 最大持仓天数 (默认: 10)

## 策略变体

### ReverseRsiStrategy
完整的双向策略，可以做多也可以做空。

### ReverseRsiLongOnlyStrategy  
纯多头策略，只做多不做空，适合A股市场。

## 数据要求

- 数据格式: CSV文件，包含Date, open, high, low, close, vol列
- 数据目录: `../tushare_data/daily/`
- 最小数据量: 每个股票至少100个交易日的数据

## 注意事项

1. **数据质量**: 确保股票数据完整且格式正确
2. **参数调优**: 不同市场环境可能需要调整参数
3. **风险控制**: 建议根据实际情况调整止损止盈参数
4. **回测偏差**: 回测结果仅供参考，实盘交易需要考虑滑点、手续费等因素

## 依赖库

```bash
pip install backtesting pandas pandas-ta tqdm scikit-learn numpy
```

## 性能优化

- 扫描脚本使用多进程并行处理，充分利用CPU资源
- 指标计算使用向量化操作，提高计算效率
- 支持大规模股票池的快速扫描

## 技术指标说明

### RSI反向价格计算
根据目标RSI水平反向计算对应的价格，这是该策略的核心创新点。

### SuperTrend
基于ATR的趋势跟踪指标，用于确认趋势方向和过滤信号。

### 发散检测
通过比较价格和RSI的高低点来识别潜在的趋势反转信号。 