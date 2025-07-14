# -*- coding: utf-8 -*-
import os
import sys
import pandas as pd
import numpy as np
import traceback
from backtesting import Backtest

# --- Setup Python Path to allow imports from parent directory ---
script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from trading.adx.strategies import AdxStrategy
from trading.adx.indicators import adx_indicator

def test_specific_failed_file():
    """测试一个具体失败的文件"""
    # 选择一个失败的文件进行测试
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'tushare_data', 'daily')
    failed_file = '002095.SZ.csv'  # 从错误信息中选择一个
    csv_file = os.path.join(data_dir, failed_file)
    
    print(f"测试失败文件: {csv_file}")
    
    try:
        # 读取数据
        df = pd.read_csv(csv_file, index_col='Date', parse_dates=True)
        print(f"原始数据形状: {df.shape}")
        print("前5行数据:")
        print(df.head())
        print("数据统计:")
        print(df.describe())
        
        # 检查数据质量
        print(f"\n数据质量检查:")
        print(f"有无穷大值: {np.isinf(df).sum().sum()}")
        print(f"有负值: {(df < 0).sum().sum()}")
        print(f"有零值: {(df == 0).sum().sum()}")
        
        # 测试ADX指标计算
        print(f"\n测试ADX指标计算:")
        high_series = pd.Series(df['High'])
        low_series = pd.Series(df['Low'])
        close_series = pd.Series(df['Close'])
        
        for length in [10, 14, 15, 20]:
            try:
                print(f"\n--- 测试长度 {length} ---")
                adx, di_plus, di_minus = adx_indicator(
                    high_series, low_series, close_series, length
                )
                print(f"ADX计算成功, 长度: {len(adx)}")
                print(f"前5个值: {adx[:5]}")
                print(f"后5个值: {adx[-5:]}")
            except Exception as e:
                print(f"ADX计算失败 (长度{length}): {e}")
                traceback.print_exc()
        
        # 测试backtesting框架
        print(f"\n测试backtesting框架:")
        try:
            # 重命名列
            df = df.rename(columns={
                'Open': 'Open', 'High': 'High', 'Low': 'Low', 
                'Close': 'Close', 'Volume': 'Volume'
            })
            
            bt = Backtest(df, AdxStrategy, cash=10000, commission=.001)
            print("Backtest对象创建成功")
            
            # 尝试运行回测
            stats = bt.run(di_length=15, adx_threshold=20)
            print("回测运行成功!")
            print(f"最终资产: {stats['End']}")
            
        except Exception as e:
            print(f"回测失败: {e}")
            traceback.print_exc()
            
    except Exception as e:
        print(f"读取文件失败: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_specific_failed_file() 