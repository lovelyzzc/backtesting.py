# -*- coding: utf-8 -*-
"""
Reverse RSI策略测试脚本

用于快速测试策略是否正常工作的简单脚本
"""

import os
import sys
import pandas as pd
from backtesting import Backtest

# --- Setup Python Path to allow imports from parent directory ---
script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from trading.reverse_rsi.strategies import ReverseRsiLongOnlyStrategy

def test_strategy():
    """测试策略是否正常工作"""
    
    # 寻找测试数据文件
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'tushare_data', 'daily')
    
    if not os.path.exists(data_dir):
        print(f"错误: 数据目录 {data_dir} 不存在")
        return
    
    # 获取第一个CSV文件进行测试
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    if not csv_files:
        print(f"错误: 在 {data_dir} 中没有找到CSV文件")
        return
    
    test_file = csv_files[0]
    print(f"使用测试文件: {test_file}")
    
    # 加载数据
    filepath = os.path.join(data_dir, test_file)
    df = pd.read_csv(filepath, index_col='Date', parse_dates=True)
    df = df.sort_index()
    
    # 重命名列
    df.rename(columns={
        'open': 'Open', 'high': 'High', 'low': 'Low', 
        'close': 'Close', 'vol': 'Volume'
    }, inplace=True)
    
    # 过滤数据（最近2年）
    df_filtered = df.loc['2023-01-01':'2025-01-15']
    
    if len(df_filtered) < 100:
        print("警告: 数据量不足，可能影响测试结果")
    
    print(f"数据范围: {df_filtered.index[0]} 到 {df_filtered.index[-1]}")
    print(f"数据点数: {len(df_filtered)}")
    
    # 运行回测
    try:
        bt = Backtest(df_filtered, ReverseRsiLongOnlyStrategy, cash=100000, commission=0.001)
        stats = bt.run()
        
        print("\n=== 回测结果 ===")
        print(f"起始资金: ¥{stats['Start']:,.0f}")
        print(f"结束资金: ¥{stats['End']:,.0f}")
        print(f"总收益: {stats['Return [%]']:.2f}%")
        print(f"年化收益: {stats['Return (Ann.) [%]']:.2f}%")
        print(f"最大回撤: {stats['Max. Drawdown [%]']:.2f}%")
        print(f"夏普比率: {stats['Sharpe Ratio']:.2f}")
        print(f"交易次数: {stats['# Trades']}")
        print(f"胜率: {stats['Win Rate [%]']:.2f}%")
        
        # 检查策略实例
        strategy_instance = stats._strategy
        print(f"\n=== 策略指标检查 ===")
        print(f"ob_price长度: {len(strategy_instance.ob_price) if hasattr(strategy_instance, 'ob_price') else 'N/A'}")
        print(f"os_price长度: {len(strategy_instance.os_price) if hasattr(strategy_instance, 'os_price') else 'N/A'}")
        print(f"st_direction长度: {len(strategy_instance.st_direction) if hasattr(strategy_instance, 'st_direction') else 'N/A'}")
        
        # 检查最后几个值
        if hasattr(strategy_instance, 'ob_price') and len(strategy_instance.ob_price) > 0:
            print(f"最后ob_price: {strategy_instance.ob_price[-1]:.2f}")
        if hasattr(strategy_instance, 'os_price') and len(strategy_instance.os_price) > 0:
            print(f"最后os_price: {strategy_instance.os_price[-1]:.2f}")
        if hasattr(strategy_instance, 'st_direction') and len(strategy_instance.st_direction) > 0:
            print(f"最后st_direction: {strategy_instance.st_direction[-1]}")
        
        print(f"最后收盘价: {df_filtered['Close'][-1]:.2f}")
        
        print("\n✅ 策略测试成功！")
        
    except Exception as e:
        print(f"\n❌ 策略测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_strategy() 