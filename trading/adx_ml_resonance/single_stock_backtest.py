# -*- coding: utf-8 -*-
"""
ADX ML共振策略单股票回测示例

这是一个简单的单股票回测示例，用于快速测试ADX ML共振策略的效果。
适合用于策略验证和参数调试。

使用方法：
python single_stock_backtest.py
"""

import os
import sys
import pandas as pd

# --- Setup Python Path to allow imports from parent directory ---
script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backtesting import Backtest
from trading.adx_ml_resonance.strategies import AdxMlResonanceStrategy

def load_stock_data(filepath, start_date=None, end_date=None):
    """加载股票数据"""
    try:
        data = pd.read_csv(filepath, index_col='Date', parse_dates=True)
        data = data.sort_index()
        
        # 过滤日期范围
        if start_date:
            data = data[data.index >= pd.to_datetime(start_date)]
        if end_date:
            data = data[data.index <= pd.to_datetime(end_date)]
            
        return data
    except Exception as e:
        print(f"加载数据失败: {e}")
        return None

def run_single_backtest(stock_file, start_date='2023-01-01', end_date='2025-01-15'):
    """运行单个股票的回测"""
    
    # 构建文件路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'tushare_data', 'daily')
    filepath = os.path.join(data_dir, stock_file)
    
    if not os.path.exists(filepath):
        print(f"错误: 文件 {filepath} 不存在")
        return None
    
    # 加载数据
    print(f"正在加载数据: {stock_file}")
    data = load_stock_data(filepath, start_date, end_date)
    
    if data is None or len(data) < 200:
        print(f"错误: 数据不足，需要至少200个交易日")
        return None
    
    print(f"数据范围: {data.index[0]} 到 {data.index[-1]} ({len(data)} 个交易日)")
    
    # 创建回测实例
    bt = Backtest(
        data, 
        AdxMlResonanceStrategy,
        cash=100000,           # 初始资金10万
        commission=0.001,      # 手续费0.1%
        exclusive_orders=True  # 避免同时开多空仓位
    )
    
    # 运行回测（使用默认参数）
    print("正在运行回测...")
    try:
        stats = bt.run()
        return bt, stats
    except Exception as e:
        print(f"回测失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def print_backtest_results(stats):
    """打印回测结果"""
    if stats is None:
        return
    
    print("\n" + "="*50)
    print("回测结果")
    print("="*50)
    
    # 主要指标
    key_metrics = {
        '期初资金': f"{stats['Start']:,.0f}",
        '期末资金': f"{stats['End']:,.0f}",
        '总收益率': f"{stats['Return [%]']:.2f}%",
        '年化收益率': f"{stats['Return (Ann.) [%]']:.2f}%",
        '买入持有收益率': f"{stats['Buy & Hold Return [%]']:.2f}%",
        '最大回撤': f"{stats['Max. Drawdown [%]']:.2f}%",
        '夏普比率': f"{stats['Sharpe Ratio']:.3f}",
        '盈利因子': f"{stats['Profit Factor']:.3f}",
        '胜率': f"{stats['Win Rate [%]']:.2f}%",
        '交易次数': f"{int(stats['# Trades'])}",
        '平均交易收益': f"{stats['Avg. Trade [%]']:.2f}%",
        '最佳交易': f"{stats['Best Trade [%]']:.2f}%",
        '最差交易': f"{stats['Worst Trade [%]']:.2f}%",
    }
    
    for name, value in key_metrics.items():
        print(f"{name:15}: {value}")
    
    print("="*50)

def main():
    """主函数"""
    print("ADX ML共振策略单股票回测示例")
    print("="*50)
    
    # 可以修改这些参数
    stock_file = '689009.SH.csv'      # 要测试的股票文件
    start_date = '2023-01-01'         # 回测开始日期
    end_date = '2025-01-15'           # 回测结束日期
    
    # 运行回测
    bt, stats = run_single_backtest(stock_file, start_date, end_date)
    
    if bt is not None and stats is not None:
        # 打印结果
        print_backtest_results(stats)
        
        # 询问是否要显示图表
        try:
            show_plot = input("\n是否显示回测图表？(y/n): ").lower().strip()
            if show_plot in ['y', 'yes', '是']:
                print("正在生成图表...")
                bt.plot(open_browser=True)
        except KeyboardInterrupt:
            print("\n程序退出")
    else:
        print("回测失败")

if __name__ == "__main__":
    main() 