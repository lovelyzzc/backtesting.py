import pandas as pd
import numpy as np
from trading.adx.indicators import adx_indicator
from trading.ml_adaptive_super_trend.indicators import ml_adaptive_super_trend


def adx_ml_resonance_indicator(high: pd.Series, low: pd.Series, close: pd.Series, 
                               adx_length: int = 10, adx_threshold: float = 20,
                               ml_atr_len: int = 10, ml_fact: float = 3.0,
                               training_data_period: int = 100,
                               highvol: float = 0.75, midvol: float = 0.5, lowvol: float = 0.25):
    """
    计算ADX和ML自适应SuperTrend的共振信号
    
    参数:
    - high, low, close: 价格序列
    - adx_length: ADX计算周期
    - adx_threshold: ADX阈值，用于判断趋势强度
    - ml_atr_len: ML SuperTrend的ATR周期
    - ml_fact: ML SuperTrend的倍数因子
    - training_data_period: ML算法的训练数据周期
    - highvol, midvol, lowvol: 波动率聚类的分界点
    
    返回:
    - adx: ADX值
    - di_plus: +DI值
    - di_minus: -DI值
    - ml_st: ML自适应SuperTrend值
    - ml_direction: ML SuperTrend方向 (1=上涨, -1=下跌)
    - resonance_signal: 共振信号 (1=买入, -1=卖出, 0=无信号)
    """
    
    # 转换为pandas Series
    if not isinstance(high, pd.Series):
        high = pd.Series(high)
    if not isinstance(low, pd.Series):
        low = pd.Series(low)
    if not isinstance(close, pd.Series):
        close = pd.Series(close)
    
    # 计算ADX指标
    adx, di_plus, di_minus = adx_indicator(high, low, close, adx_length)
    
    # 计算ML自适应SuperTrend
    ml_st, ml_direction = ml_adaptive_super_trend(
        high, low, close, 
        atr_len=ml_atr_len, 
        fact=ml_fact,
        training_data_period=training_data_period,
        highvol=highvol, 
        midvol=midvol, 
        lowvol=lowvol
    )
    
    # 转换为numpy数组以确保一致性
    adx = np.array(adx)
    di_plus = np.array(di_plus)
    di_minus = np.array(di_minus)
    ml_st = np.array(ml_st)
    ml_direction = np.array(ml_direction)
    
    # 计算共振信号
    resonance_signal = np.zeros(len(close))
    
    for i in range(1, len(close)):
        # ADX趋势强度条件
        is_adx_strong = adx[i] > adx_threshold
        
        # DI交叉条件
        di_plus_cross_up = di_plus[i] > di_minus[i] and di_plus[i-1] <= di_minus[i-1]
        di_minus_cross_up = di_minus[i] > di_plus[i] and di_minus[i-1] <= di_plus[i-1]
        
        # ML SuperTrend方向变化
        ml_bullish_signal = ml_direction[i] == 1 and ml_direction[i-1] == -1
        ml_bearish_signal = ml_direction[i] == -1 and ml_direction[i-1] == 1
        
        # 共振买入信号：ADX强势 + DI+上穿DI- + ML转为看涨
        if is_adx_strong and di_plus_cross_up and ml_bullish_signal:
            resonance_signal[i] = 1
        
        # 共振卖出信号：ADX强势 + DI-上穿DI+ + ML转为看跌  
        elif is_adx_strong and di_minus_cross_up and ml_bearish_signal:
            resonance_signal[i] = -1
        
        # 其他情况：保持前一个状态或无信号
        else:
            resonance_signal[i] = 0
    
    return adx, di_plus, di_minus, ml_st, ml_direction, resonance_signal


def simple_resonance_indicator(high: pd.Series, low: pd.Series, close: pd.Series,
                               adx_length: int = 10, adx_threshold: float = 20,
                               ml_atr_len: int = 10, ml_fact: float = 3.0,
                               training_data_period: int = 100):
    """
    简化版共振指标，只需要ADX强势和ML SuperTrend方向一致
    
    返回:
    - buy_signal: 买入信号 (布尔数组)
    - sell_signal: 卖出信号 (布尔数组)
    - adx: ADX值
    - ml_direction: ML SuperTrend方向
    """
    
    # 转换为pandas Series
    if not isinstance(high, pd.Series):
        high = pd.Series(high)
    if not isinstance(low, pd.Series):
        low = pd.Series(low)
    if not isinstance(close, pd.Series):
        close = pd.Series(close)
    
    # 计算ADX指标
    adx, di_plus, di_minus = adx_indicator(high, low, close, adx_length)
    
    # 计算ML自适应SuperTrend
    ml_st, ml_direction = ml_adaptive_super_trend(
        high, low, close, 
        atr_len=ml_atr_len, 
        fact=ml_fact,
        training_data_period=training_data_period
    )
    
    # 转换为numpy数组
    adx = np.array(adx)
    ml_direction = np.array(ml_direction)
    
    # 计算信号
    buy_signal = np.zeros(len(close), dtype=bool)
    sell_signal = np.zeros(len(close), dtype=bool)
    
    for i in range(1, len(close)):
        # ADX强势条件
        is_adx_strong = adx[i] > adx_threshold
        
        # ML SuperTrend方向变化
        ml_bullish_reversal = ml_direction[i] == 1 and ml_direction[i-1] == -1
        ml_bearish_reversal = ml_direction[i] == -1 and ml_direction[i-1] == 1
        
        # 共振买入信号：ADX强势 + ML转为看涨
        if is_adx_strong and ml_bullish_reversal:
            buy_signal[i] = True
        
        # 共振卖出信号：ADX强势 + ML转为看跌
        elif is_adx_strong and ml_bearish_reversal:
            sell_signal[i] = True
    
    return buy_signal, sell_signal, adx, ml_direction 