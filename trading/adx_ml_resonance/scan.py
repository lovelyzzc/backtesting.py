# -*- coding: utf-8 -*-
import os
import sys
import pandas as pd
from backtesting import Backtest
from multiprocessing import Pool, cpu_count
from functools import partial
from datetime import datetime

# --- Setup Python Path to allow imports from parent directory ---
script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from trading.adx_ml_resonance.strategies import AdxMlResonanceStrategy

def analyze_stock(filename, data_dir, start_date, end_date):
    """
    分析单个股票文件，检测最后一天的买入或卖出信号
    这是为并行处理设计的工作函数
    """
    filepath = os.path.join(data_dir, filename)
    stock_symbol = os.path.splitext(filename)[0]
    
    try:
        # 从Tushare CSV加载数据
        df = pd.read_csv(filepath, index_col='Date', parse_dates=True)
        df = df.sort_index()

        # backtesting.py需要列名: 'Open', 'High', 'Low', 'Close', 'Volume'
        df.rename(columns={
            'open': 'Open', 'high': 'High', 'low': 'Low', 
            'close': 'Close', 'vol': 'Volume'
        }, inplace=True)

        # 过滤相关时期的数据
        df_filtered = df.loc[start_date:end_date]

        # 策略需要足够的数据来生成指标
        if len(df_filtered) < 200:  # 共振策略需要更多数据
            return None

        # 运行回测，设置足够的初始资金以避免价格超出警告
        bt = Backtest(df_filtered, AdxMlResonanceStrategy, cash=1000000, commission=0.0)
        stats = bt.run()

        # 检查最后一天的信号
        strategy_instance = stats._strategy
        
        # 检查高级共振信号
        if hasattr(strategy_instance, 'resonance_signal'):
            if len(strategy_instance.resonance_signal) > 0:
                if strategy_instance.resonance_signal[-1] == 1:
                    print(f"✅ 高级共振+成交量买入信号: {stock_symbol}")
                    return (stock_symbol, 'buy')
                elif strategy_instance.resonance_signal[-1] == -1:
                    # print(f"❌ 高级共振+成交量卖出信号: {stock_symbol}")
                    return (stock_symbol, 'sell')

    except Exception as e:
        # 静默忽略个别股票的错误，避免中断整个扫描
        # 更强大的解决方案可以将这些错误记录到单独的文件中
        print(f"⚠️ 分析 {stock_symbol} 时出错: {str(e)}")
        return None
    
    return None

def find_resonance_signals(data_dir, start_date='2020-01-01', end_date='2025-07-08'):
    """
    并行扫描目录中的所有CSV文件，寻找最后一天有买入或卖出信号的股票
    """
    files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    n_cores = cpu_count()
    print(f"发现 {len(files)} 个CSV文件。开始使用 {n_cores} 个CPU核心分析...")

    # 使用functools.partial创建一个预填充了部分参数的新函数
    # 这是在pool.map中向工作函数传递额外参数的简洁方式
    process_func = partial(analyze_stock, 
                           data_dir=data_dir, 
                           start_date=start_date, 
                           end_date=end_date)

    # 进程池管理一组工作进程
    with Pool(n_cores) as p:
        # map将process_func应用到'files'列表中的每个项目
        # 处理过程在工作进程之间分布
        results = p.map(process_func, files)
    
    # 过滤掉None值并分离信号
    valid_results = [res for res in results if res is not None]
    buy_signal_stocks = [stock for stock, sig_type in valid_results if sig_type == 'buy']
    sell_signal_stocks = [stock for stock, sig_type in valid_results if sig_type == 'sell']
    
    return buy_signal_stocks, sell_signal_stocks

def save_signals_to_ini(filepath, stock_list, block_name):
    """将股票代码列表保存到同花顺的INI文件中"""
    if stock_list:
        print(f"\n--- 发现 {block_name} 信号 ---")
        
        formatted_stocks = []
        for stock_symbol in stock_list:
            try:
                code = stock_symbol.split('.')[0]
                formatted_stocks.append(code)
                print(f"- {stock_symbol} -> {code}")
            except IndexError:
                print(f"  - 警告: 跳过格式错误的股票代码 {stock_symbol}")
                continue
        
        with open(filepath, 'w', encoding='gbk') as f:
            f.write("[自选股设置]\n")
            f.write(f"股票代码={','.join(formatted_stocks)}\n")
            f.write(f"板块名称={block_name}\n")

        print(f"\n同花顺导入文件已保存到: {filepath}")
    else:
        print(f"\n--- 未发现 {block_name} 信号 ---")
        # 如果没有找到股票，创建一个空的INI文件
        with open(filepath, 'w', encoding='gbk') as f:
            f.write("[自选股设置]\n")
            f.write("股票代码=\n")
            f.write(f"板块名称={block_name}\n")
        print(f"\n已创建空结果文件: {filepath}")

if __name__ == "__main__":
    # 定义数据目录 (请确保路径正确)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'tushare_data', 'daily')

    # 运行扫描
    today = datetime.now().strftime('%Y-%m-%d')
    print("开始ADX-ML高级共振+成交量策略股票扫描...")
    
    buy_stocks, sell_stocks = find_resonance_signals(
        data_dir=data_dir,
        start_date='2024-01-01',
        end_date=today
    )

    # 打印并保存结果
    results_dir = os.path.join(script_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # 保存买入信号
    buy_output_filepath = os.path.join(results_dir, '高级共振+成交量_买入信号.ini')
    save_signals_to_ini(buy_output_filepath, buy_stocks, "高级共振+成交量买入")

    # 保存卖出信号
    sell_output_filepath = os.path.join(results_dir, '高级共振+成交量_卖出信号.ini')
    save_signals_to_ini(sell_output_filepath, sell_stocks, "高级共振+成交量卖出")
    
    print(f"\n🎉 扫描完成！结果已保存到 results 目录") 