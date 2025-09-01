# -*- coding: utf-8 -*-
"""
简化测试002299.SZ的SuperTrend计算
"""
import pandas as pd
import numpy as np

def simple_supertrend(high, low, close, factor=2.4, atr_period=10):
    """简化的SuperTrend计算"""
    # 计算ATR
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # 简单移动平均计算ATR
    atr = pd.Series(tr).rolling(window=atr_period).mean().values
    
    # 计算基础带
    hl2 = (high + low) / 2
    upper_band = hl2 + factor * atr
    lower_band = hl2 - factor * atr
    
    # 计算SuperTrend
    supertrend = np.full_like(close, np.nan)
    direction = np.full_like(close, np.nan)
    
    for i in range(len(close)):
        if i == 0:
            direction[i] = 1  # 初始为看涨
        else:
            # 简化的方向判断
            if close[i] > upper_band[i-1]:
                direction[i] = -1  # 看跌
            elif close[i] < lower_band[i-1]:
                direction[i] = 1   # 看涨
            else:
                direction[i] = direction[i-1]  # 保持之前方向
        
        supertrend[i] = lower_band[i] if direction[i] == -1 else upper_band[i]
    
    return supertrend, direction

def test_002299():
    """测试002299.SZ"""
    # 加载数据
    filepath = "/home/lovelyzzc/backtesting.py/trading/tushare_data/daily/002299.SZ.csv"
    df = pd.read_csv(filepath, index_col='Date', parse_dates=True)
    df = df.sort_index()
    
    # 只使用2024年的数据
    df_2024 = df.loc['2024-01-01':'2024-12-31']
    
    print(f"数据范围: {df_2024.index[0]} 到 {df_2024.index[-1]}")
    print(f"总共 {len(df_2024)} 天数据")
    
    # 计算SuperTrend
    st_value, st_direction = simple_supertrend(
        df_2024['High'].values,
        df_2024['Low'].values,
        df_2024['Close'].values
    )
    
    # 显示最后10天
    n_days = 10
    print(f"\n=== 002299.SZ 最后{n_days}天数据 ===")
    print("日期\t\t收盘价\tST方向\t变化")
    print("-" * 60)
    
    for i in range(-n_days, 0):
        date = df_2024.index[i]
        close_price = df_2024['Close'].iloc[i]
        curr_direction = st_direction[i]
        
        # 检查方向变化
        change = ""
        if i > -n_days:
            prev_direction = st_direction[i-1]
            if curr_direction != prev_direction:
                if curr_direction == 1 and prev_direction == -1:
                    change = "🔵 转涨"
                elif curr_direction == -1 and prev_direction == 1:
                    change = "🔴 转跌"
        
        direction_str = "看涨" if curr_direction == 1 else "看跌"
        print(f"{date.strftime('%Y-%m-%d')}\t{close_price:.2f}\t{direction_str}\t{change}")
    
    # 检查最后的趋势变化
    if len(st_direction) >= 2:
        last_dir = st_direction[-1]
        prev_dir = st_direction[-2]
        
        print(f"\n最后两天方向: {prev_dir} -> {last_dir}")
        
        if last_dir == -1 and prev_dir == 1:
            print("✅ 检测到趋势转跌")
        elif last_dir == 1 and prev_dir == -1:
            print("✅ 检测到趋势转涨")
        else:
            print("❌ 无趋势转换")

if __name__ == "__main__":
    test_002299() 