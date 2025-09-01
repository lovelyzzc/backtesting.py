# -*- coding: utf-8 -*-
"""
调试Reverse RSI信号检测
验证SuperTrend方向判断是否正确
"""
import os
import sys
import pandas as pd
import numpy as np
from backtesting import Backtest
from datetime import datetime

# Setup Python Path
script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from trading.reverse_rsi.strategies import ReverseRsiLongOnlyStrategy

def debug_single_stock(stock_symbol, data_dir, start_date, end_date):
    """调试单个股票的信号"""
    filepath = os.path.join(data_dir, f"{stock_symbol}.csv")
    
    if not os.path.exists(filepath):
        print(f"文件不存在: {filepath}")
        return
    
    # 加载数据
    df = pd.read_csv(filepath, index_col='Date', parse_dates=True)
    df = df.sort_index()
    
    # 重命名列
    df.rename(columns={
        'open': 'Open', 'high': 'High', 'low': 'Low', 
        'close': 'Close', 'vol': 'Volume'
    }, inplace=True)
    
    # 过滤数据
    df_filtered = df.loc[start_date:end_date]
    
    if len(df_filtered) < 100:
        print(f"数据不足: {len(df_filtered)} 行")
        return
    
    # 运行回测
    bt = Backtest(df_filtered, ReverseRsiLongOnlyStrategy, cash=1000000, commission=0.0)
    stats = bt.run()
    
    strategy_instance = stats._strategy
    
    # 获取最后10天的数据进行分析
    n_days = min(10, len(strategy_instance.data.Close))
    
    print(f"\n=== {stock_symbol} 最后{n_days}天数据分析 ===")
    print("日期\t\t收盘价\tST方向\tST值\t\t信号")
    print("-" * 80)
    
    dates = df_filtered.index[-n_days:]
    
    for i in range(-n_days, 0):
        date = dates[i + n_days]
        close_price = strategy_instance.data.Close[i]
        st_direction = strategy_instance.st_direction[i] if len(strategy_instance.st_direction) > abs(i) else np.nan
        st_value = strategy_instance.st_value[i] if len(strategy_instance.st_value) > abs(i) else np.nan
        
        # 检查信号
        signal = ""
        if i == -1:  # 最后一天
            # 检查趋势转换
            if len(strategy_instance.st_direction) >= 2:
                prev_direction = strategy_instance.st_direction[i-1]
                curr_direction = strategy_instance.st_direction[i]
                
                if curr_direction == 1 and prev_direction == -1:
                    signal = "🔵 趋势转涨"
                elif curr_direction == -1 and prev_direction == 1:
                    signal = "🔴 趋势转跌"
            
            # 检查价格突破
            if (hasattr(strategy_instance, 'os_price') and 
                len(strategy_instance.os_price) >= 2):
                if (close_price > strategy_instance.os_price[i] and 
                    strategy_instance.data.Close[i-1] <= strategy_instance.os_price[i-1]):
                    signal += " 🟢 突破超卖"
            
            if (hasattr(strategy_instance, 'ob_price') and 
                len(strategy_instance.ob_price) >= 2):
                if (close_price < strategy_instance.ob_price[i] and
                    strategy_instance.data.Close[i-1] >= strategy_instance.ob_price[i-1]):
                    signal += " 🔴 突破超买"
        
        direction_str = "看涨" if st_direction == 1 else "看跌" if st_direction == -1 else "N/A"
        
        print(f"{date.strftime('%Y-%m-%d')}\t{close_price:.2f}\t{direction_str}\t{st_value:.2f}\t\t{signal}")
    
    print(f"\n当前SuperTrend方向: {'看涨' if strategy_instance.st_direction[-1] == 1 else '看跌' if strategy_instance.st_direction[-1] == -1 else 'N/A'}")

if __name__ == "__main__":
    # 测试几个之前报告有"趋势转涨"信号的股票
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'tushare_data', 'daily')
    
    test_stocks = ['300101.SZ', '002299.SZ', '603861.SH', '002217.SZ']
    
    today = datetime.now().strftime('%Y-%m-%d')
    
    for stock in test_stocks:
        debug_single_stock(stock, data_dir, '2024-01-01', today)
        print("\n" + "="*100 + "\n") 