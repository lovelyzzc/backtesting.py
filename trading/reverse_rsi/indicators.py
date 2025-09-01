# -*- coding: utf-8 -*-
"""
Reverse RSI指标实现

基于Pine Script的Reverse RSI Signals指标的Python实现。
包含RSI反向价格计算、SuperTrend趋势判断和RSI发散检测。
"""

import numpy as np
import pandas as pd
import pandas_ta as ta
from typing import Tuple


def rma(series: np.ndarray, length: int) -> np.ndarray:
    """
    计算RMA (Running Moving Average)，等同于Wilder's Smoothing
    """
    alpha = 1.0 / length
    result = np.full_like(series, np.nan)
    
    # 找到第一个非NaN值
    first_valid = np.where(~np.isnan(series))[0]
    if len(first_valid) == 0:
        return result
    
    start_idx = first_valid[0]
    result[start_idx] = series[start_idx]
    
    for i in range(start_idx + 1, len(series)):
        if not np.isnan(series[i]):
            result[i] = alpha * series[i] + (1 - alpha) * result[i-1]
    
    return result


def price_for_rsi(level: float, up_prev: float, dn_prev: float, length: int, src_prev: float) -> float:
    """
    根据RSI水平反向计算价格
    这是Pine Script中f_price_for_rsi函数的Python实现
    """
    lv = max(1e-6, min(level, 100 - 1e-6))
    RS = lv / (100.0 - lv)
    A = up_prev * (length - 1)
    B = dn_prev * (length - 1)
    c_pos = RS * B - A
    c_neg = B - A / RS
    c = c_pos if c_pos >= 0 else c_neg
    return src_prev + c


def calc_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, length: int) -> np.ndarray:
    """
    计算ATR (Average True Range)
    """
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # 第一个值使用high-low
    
    return rma(tr, length)


def supertrend(high: np.ndarray, low: np.ndarray, close: np.ndarray, 
               factor: float, atr_period: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算SuperTrend指标
    返回: (supertrend_values, trend_direction)
    """
    hl2 = (high + low) / 2
    atr = calc_atr(high, low, close, atr_period)
    
    upper_band = hl2 + factor * atr
    lower_band = hl2 - factor * atr
    
    # 初始化数组
    final_upper = np.full_like(upper_band, np.nan)
    final_lower = np.full_like(lower_band, np.nan)
    supertrend = np.full_like(close, np.nan)
    direction = np.full_like(close, np.nan)
    
    for i in range(len(close)):
        if i == 0:
            final_upper[i] = upper_band[i]
            final_lower[i] = lower_band[i]
            direction[i] = 1
        else:
            # 计算final bands
            final_upper[i] = upper_band[i] if (upper_band[i] < final_upper[i-1] or close[i-1] > final_upper[i-1]) else final_upper[i-1]
            final_lower[i] = lower_band[i] if (lower_band[i] > final_lower[i-1] or close[i-1] < final_lower[i-1]) else final_lower[i-1]
            
            # 计算趋势方向
            if supertrend[i-1] == final_upper[i-1]:
                direction[i] = -1 if close[i] > final_upper[i] else 1
            else:
                direction[i] = 1 if close[i] < final_lower[i] else -1
        
        # 设置supertrend值
        supertrend[i] = final_lower[i] if direction[i] == -1 else final_upper[i]
    
    return supertrend, direction


def find_pivots(series: np.ndarray, left_bars: int, right_bars: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    寻找数据序列中的高点和低点
    返回: (pivot_highs, pivot_lows) - 布尔数组
    """
    pivot_highs = np.zeros(len(series), dtype=bool)
    pivot_lows = np.zeros(len(series), dtype=bool)
    
    for i in range(left_bars, len(series) - right_bars):
        # 检查是否为高点
        is_high = True
        for j in range(i - left_bars, i + right_bars + 1):
            if j != i and series[j] >= series[i]:
                is_high = False
                break
        pivot_highs[i] = is_high
        
        # 检查是否为低点
        is_low = True
        for j in range(i - left_bars, i + right_bars + 1):
            if j != i and series[j] <= series[i]:
                is_low = False
                break
        pivot_lows[i] = is_low
    
    return pivot_highs, pivot_lows


def reverse_rsi_indicator(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray,
                         rsi_length: int = 14, smooth_bands: bool = True, 
                         st_factor: float = 2.4, st_atr_len: int = 10,
                         div_lookback: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Reverse RSI指标主函数
    
    返回:
    - ob_price: 超买价格水平 (RSI 70对应的价格)
    - os_price: 超卖价格水平 (RSI 30对应的价格)  
    - mid_price: 中性价格水平 (RSI 50对应的价格)
    - st_value: SuperTrend值
    - st_direction: SuperTrend方向 (1=看涨, -1=看跌)
    - bull_divergence: 看涨发散信号
    - bear_divergence: 看跌发散信号
    """
    # 计算价格变化和RSI组件
    chg = np.diff(close, prepend=close[0])
    up_now = np.maximum(chg, 0)
    dn_now = np.maximum(-chg, 0)
    
    up = rma(up_now, rsi_length)
    dn = rma(dn_now, rsi_length)
    
    # 初始化价格水平数组
    ob_price_raw = np.full_like(close, np.nan)
    os_price_raw = np.full_like(close, np.nan)
    mid_price = np.full_like(close, np.nan)
    
    # 计算反向RSI价格水平
    for i in range(1, len(close)):
        if not (np.isnan(up[i-1]) or np.isnan(dn[i-1]) or np.isnan(close[i-1])):
            ob_price_raw[i] = price_for_rsi(70.0, up[i-1], dn[i-1], rsi_length, close[i-1])
            os_price_raw[i] = price_for_rsi(30.0, up[i-1], dn[i-1], rsi_length, close[i-1])
            mid_price[i] = price_for_rsi(50.0, up[i-1], dn[i-1], rsi_length, close[i-1])
    
    # 平滑价格带（如果启用）
    if smooth_bands:
        ema_len = 14
        ob_price = pd.Series(ob_price_raw).ewm(span=ema_len).mean().values
        os_price = pd.Series(os_price_raw).ewm(span=ema_len).mean().values
    else:
        ob_price = ob_price_raw
        os_price = os_price_raw
    
    # 计算SuperTrend（基于实际收盘价）
    st_value, st_direction = supertrend(high, low, close, st_factor, st_atr_len)
    
    # 计算RSI用于发散检测
    rsi = pd.Series(close).rolling(window=rsi_length).apply(
        lambda x: 100 - (100 / (1 + (x.diff().clip(lower=0).mean() / 
                                   (-x.diff().clip(upper=0).mean() + 1e-10))))
    ).values
    
    # 寻找RSI和价格的枢轴点
    rsi_pivot_highs, rsi_pivot_lows = find_pivots(rsi, div_lookback, 1)
    price_pivot_highs, price_pivot_lows = find_pivots(high, div_lookback, 1)
    price_low_pivots, _ = find_pivots(-low, div_lookback, 1)  # 寻找价格低点
    
    # 检测发散
    bull_divergence = np.zeros(len(close), dtype=bool)
    bear_divergence = np.zeros(len(close), dtype=bool)
    
    for i in range(div_lookback + 5, len(close)):
        # 看涨发散：价格创新低但RSI创新高
        if rsi_pivot_lows[i]:
            # 寻找之前的RSI低点
            prev_rsi_lows = np.where(rsi_pivot_lows[:i])[0]
            if len(prev_rsi_lows) > 0:
                prev_idx = prev_rsi_lows[-1]
                if (i - prev_idx >= 5 and i - prev_idx <= 60 and  # 时间范围检查
                    rsi[i] > rsi[prev_idx] and  # RSI创新高
                    low[i] < low[prev_idx]):    # 价格创新低
                    bull_divergence[i] = True
        
        # 看跌发散：价格创新高但RSI创新低  
        if rsi_pivot_highs[i]:
            # 寻找之前的RSI高点
            prev_rsi_highs = np.where(rsi_pivot_highs[:i])[0]
            if len(prev_rsi_highs) > 0:
                prev_idx = prev_rsi_highs[-1]
                if (i - prev_idx >= 5 and i - prev_idx <= 60 and  # 时间范围检查
                    rsi[i] < rsi[prev_idx] and  # RSI创新低
                    high[i] > high[prev_idx]):  # 价格创新高
                    bear_divergence[i] = True
    
    return ob_price, os_price, mid_price, st_value, st_direction, bull_divergence, bear_divergence 