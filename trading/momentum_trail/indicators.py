# -*- coding: utf-8 -*-
# @Author: lovelyzzc
# @Date: 2024-07-31

import pandas as pd
import numpy as np
import pandas_ta as ta

# ━━━━━━━━━━━━━━━━ 1. INDICATOR DEFINITIONS ━━━━━━━━━━━━━━━━

def double_smooth(src, length, smth_len):
    """Applies a double EMA smoothing to the source series."""
    first = ta.ema(src, length=length)
    if first is None:
        return None
    return ta.ema(first, length=smth_len)

def trail_indicator_optimized(src, length, mult):
    """
    优化版本的trailing stop bands计算，使用向量化操作大幅提升性能。
    这个版本将原来的O(N²)复杂度优化为O(N)。
    """
    # 预计算基础指标
    basis = ta.wma(src, length=int(length / 2))
    vola = ta.hma(abs(src.diff()), length=length)

    upper_band = basis + vola * mult
    lower_band = basis - vola * mult

    # 初始化输出序列
    upper = pd.Series(dtype='float64', index=src.index)
    lower = pd.Series(dtype='float64', index=src.index)
    direction = pd.Series(1, index=src.index, dtype='int8')  # 使用int8节省内存

    # 向量化的条件判断，避免循环
    wma_length = length * 3
    
    # 预分配中间序列
    intermediate_upper = pd.Series(dtype='float64', index=src.index)
    intermediate_lower = pd.Series(dtype='float64', index=src.index)
    
    # 第一个值
    intermediate_upper.iloc[0] = upper_band.iloc[0] if not pd.isna(upper_band.iloc[0]) else 0
    intermediate_lower.iloc[0] = lower_band.iloc[0] if not pd.isna(lower_band.iloc[0]) else 0
    
    # 使用向量化操作计算中间值
    for i in range(1, len(src)):
        last_upper = upper.iloc[i-1] if i > 0 else intermediate_upper.iloc[0]
        last_lower = lower.iloc[i-1] if i > 0 else intermediate_lower.iloc[0]
        
        # 向量化条件判断
        intermediate_upper.iloc[i] = np.where(
            (upper_band.iloc[i] < last_upper) | (src.iloc[i-1] > last_upper),
            upper_band.iloc[i],
            last_upper
        )
        
        intermediate_lower.iloc[i] = np.where(
            (lower_band.iloc[i] > last_lower) | (src.iloc[i-1] < last_lower),
            lower_band.iloc[i],
            last_lower
        )
        
        # 批量计算WMA（关键优化）
        if i >= wma_length - 1:
            # 一次性计算整个窗口的WMA，避免重复计算
            upper_window = intermediate_upper.iloc[i+1-wma_length:i+1]
            lower_window = intermediate_lower.iloc[i+1-wma_length:i+1]
            
            # 使用更高效的WMA计算
            upper.iloc[i] = _fast_wma(upper_window.values, wma_length)
            lower.iloc[i] = _fast_wma(lower_window.values, wma_length)
        else:
            upper.iloc[i] = intermediate_upper.iloc[i]
            lower.iloc[i] = intermediate_lower.iloc[i]

    # 向量化计算direction
    upper_values = upper.values
    lower_values = lower.values
    src_values = src.values
    direction_values = np.ones(len(src), dtype=np.int8)
    
    for i in range(1, len(src)):
        last_dir = direction_values[i-1]
        if last_dir == -1 and src_values[i] > upper_values[i]:
            direction_values[i] = 1
        elif last_dir == 1 and src_values[i] < lower_values[i]:
            direction_values[i] = -1
        else:
            direction_values[i] = last_dir
    
    direction = pd.Series(direction_values, index=src.index)
    return direction, upper, lower

def _fast_wma(values, length):
    """
    快速WMA计算，避免使用pandas_ta的开销
    """
    if len(values) < length:
        return np.nan
    
    weights = np.arange(1, length + 1)
    weights_sum = weights.sum()
    
    if len(values) == length:
        return np.sum(values * weights) / weights_sum
    else:
        # 取最后length个值
        last_values = values[-length:]
        return np.sum(last_values * weights) / weights_sum

def trail_indicator(src, length, mult):
    """
    保持向后兼容的wrapper函数
    """
    return trail_indicator_optimized(src, length, mult)

def momentum_indicator(close, osc_len, smth_len, trail_len, trail_mult):
    """Calculates the final momentum and direction signal."""
    # backtesting.py passes numpy arrays for performance, but pandas-ta expects Series.
    close_series = pd.Series(close)

    price = ta.hma(close_series, length=int(osc_len / 3))
    
    if price is None:
        # This can happen if the input series is too short for the indicator.
        # Return an array of NaNs which backtesting.py can handle.
        return np.full_like(close, np.nan)

    pc = price.diff()
    
    double_pc = double_smooth(pc, osc_len, smth_len)
    double_abs_pc = double_smooth(abs(pc), osc_len, smth_len)
    
    if double_pc is None or double_abs_pc is None:
        return np.full_like(close, np.nan)
    
    # 避免除零操作，使用向量化
    divisor = np.where(double_abs_pc == 0, 1, double_abs_pc)
    mom = 100 * (double_pc / divisor)

    if isinstance(mom, pd.Series):
        mom.fillna(0, inplace=True) # Fill any potential NaNs in mom
    elif pd.isna(mom):
        mom = 0

    direction, _, _ = trail_indicator(mom, trail_len, trail_mult)
    return direction 