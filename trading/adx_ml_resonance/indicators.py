import pandas as pd
import numpy as np
from trading.adx.indicators import adx_indicator
from trading.ml_adaptive_super_trend.indicators import ml_adaptive_super_trend


def adx_ml_resonance_indicator(high: pd.Series, low: pd.Series, close: pd.Series, 
                               adx_length: int = 10, adx_threshold: float = 20,
                               ml_atr_len: int = 10, ml_fact: float = 3.0,
                               training_data_period: int = 100,
                               highvol: float = 0.75, midvol: float = 0.5, lowvol: float = 0.25,
                               di_threshold: float = 0):
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
    - di_threshold: DI差值阈值，0表示不使用DI条件，>0表示启用增强版逻辑
    
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
        
        # ML SuperTrend方向变化
        ml_bullish_signal = ml_direction[i] == 1 and ml_direction[i-1] == -1
        ml_bearish_signal = ml_direction[i] == -1 and ml_direction[i-1] == 1
        
        # 基础共振条件：ADX强势 + ML方向转换
        basic_bullish = is_adx_strong and ml_bullish_signal
        basic_bearish = is_adx_strong and ml_bearish_signal
        
        if di_threshold > 0:
            # 增强版逻辑：基础共振 + DI差值条件
            di_diff_bullish = di_plus[i] - di_minus[i] > di_threshold  # DI+占优
            di_diff_bearish = di_minus[i] - di_plus[i] > di_threshold  # DI-占优
            
            # 增强版买入信号：基础共振 + DI+占优
            if basic_bullish and di_diff_bullish:
                resonance_signal[i] = 1
            
            # 增强版卖出信号：基础共振 + DI-占优
            elif basic_bearish and di_diff_bearish:
                resonance_signal[i] = -1
            
            else:
                resonance_signal[i] = 0
        
        else:
            # 标准版逻辑：只需要基础共振
            if basic_bullish:
                resonance_signal[i] = 1
            elif basic_bearish:
                resonance_signal[i] = -1
            else:
                resonance_signal[i] = 0
    
    return adx, di_plus, di_minus, ml_st, ml_direction, resonance_signal


def simple_resonance_indicator(high: pd.Series, low: pd.Series, close: pd.Series,
                               adx_length: int = 10, adx_threshold: float = 20,
                               ml_atr_len: int = 10, ml_fact: float = 3.0,
                               training_data_period: int = 100,
                               highvol: float = 0.75, midvol: float = 0.5, lowvol: float = 0.25):
    """
    简化版共振指标，只需要ADX强势和ML SuperTrend方向一致
    现在使用完整的K-means聚类算法来计算ML自适应SuperTrend
    
    参数:
    - high, low, close: 价格序列
    - adx_length: ADX计算周期
    - adx_threshold: ADX阈值，用于判断趋势强度
    - ml_atr_len: ML SuperTrend的ATR周期
    - ml_fact: ML SuperTrend的倍数因子
    - training_data_period: ML算法的训练数据周期
    - highvol, midvol, lowvol: 波动率聚类的分界点
    
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
    
    # 计算ML自适应SuperTrend（使用完整的K-means聚类算法）
    ml_st, ml_direction = ml_adaptive_super_trend(
        high, low, close, 
        atr_len=ml_atr_len, 
        fact=ml_fact,
        training_data_period=training_data_period,
        highvol=highvol, 
        midvol=midvol, 
        lowvol=lowvol
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