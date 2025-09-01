# -*- coding: utf-8 -*-
"""
专门调试002299.SZ的SuperTrend计算
"""
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Setup Python Path
script_path = os.path.abspath(__file__)
project_root = os.path.dirname(script_path)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from trading.reverse_rsi.indicators import reverse_rsi_indicator

def debug_002299():
    """调试002299.SZ的SuperTrend计算"""
    
    # 加载数据
    filepath = "/home/lovelyzzc/backtesting.py/trading/tushare_data/daily/002299.SZ.csv"
    df = pd.read_csv(filepath, index_col='Date', parse_dates=True)
    df = df.sort_index()
    
    # 重命名列
    df.rename(columns={
        'Open': 'Open', 'High': 'High', 'Low': 'Low', 
        'Close': 'Close', 'Volume': 'Volume'
    }, inplace=True)
    
    # 过滤到今天为止的数据
    today = datetime.now().strftime('%Y-%m-%d')
    df_filtered = df.loc['2024-01-01':today]
    
    print(f"数据范围: {df_filtered.index[0]} 到 {df_filtered.index[-1]}")
    print(f"总共 {len(df_filtered)} 天数据")
    
    # 检查是否有未来日期
    future_dates = df_filtered[df_filtered.index > pd.Timestamp(today)]
    if len(future_dates) > 0:
        print(f"\n⚠️ 发现未来日期数据:")
        print(future_dates.tail())
        # 重新过滤，只保留今天及之前的数据
        df_filtered = df_filtered[df_filtered.index <= pd.Timestamp(today)]
        print(f"\n过滤后数据范围: {df_filtered.index[0]} 到 {df_filtered.index[-1]}")
    
    # 计算指标
    (ob_price, os_price, mid_price, 
     st_value, st_direction, 
     bull_divergence, bear_divergence) = reverse_rsi_indicator(
        df_filtered['High'].values,
        df_filtered['Low'].values, 
        df_filtered['Close'].values,
        df_filtered['Volume'].values,
        rsi_length=14,
        smooth_bands=True,
        st_factor=2.4,
        st_atr_len=10,
        div_lookback=3
    )
    
    # 分析最后10天的SuperTrend变化
    n_days = min(10, len(df_filtered))
    print(f"\n=== 002299.SZ 最后{n_days}天SuperTrend分析 ===")
    print("日期\t\t收盘价\tST方向\tST值\t\t变化")
    print("-" * 80)
    
    for i in range(-n_days, 0):
        date = df_filtered.index[i]
        close_price = df_filtered['Close'].iloc[i]
        
        if len(st_direction) > abs(i):
            curr_direction = st_direction[i]
            curr_st_value = st_value[i]
            
            # 检查方向变化
            change = ""
            if i > -n_days and len(st_direction) > abs(i-1):
                prev_direction = st_direction[i-1]
                if curr_direction != prev_direction:
                    if curr_direction == 1 and prev_direction == -1:
                        change = "🔵 转为看涨"
                    elif curr_direction == -1 and prev_direction == 1:
                        change = "🔴 转为看跌"
            
            direction_str = "看涨" if curr_direction == 1 else "看跌" if curr_direction == -1 else "N/A"
            
            print(f"{date.strftime('%Y-%m-%d')}\t{close_price:.2f}\t{direction_str}\t{curr_st_value:.2f}\t\t{change}")
        else:
            print(f"{date.strftime('%Y-%m-%d')}\t{close_price:.2f}\t无数据")
    
    # 检查最后一天的具体信号
    if len(st_direction) >= 2:
        last_direction = st_direction[-1]
        prev_direction = st_direction[-2]
        
        print(f"\n=== 信号分析 ===")
        print(f"前一天SuperTrend方向: {'看涨' if prev_direction == 1 else '看跌'} ({prev_direction})")
        print(f"最后一天SuperTrend方向: {'看涨' if last_direction == 1 else '看跌'} ({last_direction})")
        
        if last_direction == -1 and prev_direction == 1:
            print("✅ 检测到趋势转跌信号 (从看涨转为看跌)")
        elif last_direction == 1 and prev_direction == -1:
            print("✅ 检测到趋势转涨信号 (从看跌转为看涨)")
        else:
            print("❌ 没有趋势转换信号")
    
    # 检查价格突破信号
    if len(ob_price) >= 2 and len(os_price) >= 2:
        last_close = df_filtered['Close'].iloc[-1]
        prev_close = df_filtered['Close'].iloc[-2]
        
        print(f"\n=== 价格突破分析 ===")
        print(f"当前收盘价: {last_close:.2f}")
        print(f"当前超买价格: {ob_price[-1]:.2f}")
        print(f"当前超卖价格: {os_price[-1]:.2f}")
        
        # 检查突破超买
        if (last_close < ob_price[-1] and prev_close >= ob_price[-2]):
            print("✅ 检测到突破超买信号 (价格向下突破超买线)")
        
        # 检查突破超卖  
        if (last_close > os_price[-1] and prev_close <= os_price[-2]):
            print("✅ 检测到突破超卖信号 (价格向上突破超卖线)")

if __name__ == "__main__":
    debug_002299() 