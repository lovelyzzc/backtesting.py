# -*- coding: utf-8 -*-
import os
import sys
import pandas as pd
from backtesting import Backtest
from multiprocessing import Pool, cpu_count
from functools import partial

# --- Setup Python Path to allow imports from parent directory ---
script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from trading.ml_adaptive_super_trend.strategies import MlAdaptiveSuperTrendStrategy

def analyze_stock(filename, data_dir, strategy_class, start_date, end_date):
    """
    Analyzes a single stock file to find a buy signal on the last day.
    This is a worker function designed for parallel processing.
    """
    filepath = os.path.join(data_dir, filename)
    stock_symbol = os.path.splitext(filename)[0]
    
    try:
        # Load data from Tushare CSV
        df = pd.read_csv(filepath, index_col='Date', parse_dates=True)
        df = df.sort_index()

        # backtesting.py requires columns: 'Open', 'High', 'Low', 'Close', 'Volume'
        df.rename(columns={
            'open': 'Open', 'high': 'High', 'low': 'Low', 
            'close': 'Close', 'vol': 'Volume'
        }, inplace=True)

        # Filter data for the relevant period
        df_filtered = df.loc[start_date:end_date]

        # The strategy requires a certain amount of data to generate indicators
        if len(df_filtered) < 150:
            return None

        # Run the backtest. We don't need cash or commission for signal detection.
        bt = Backtest(df_filtered, strategy_class)
        stats = bt.run()

        # Check for a buy signal on the very last day
        strategy_instance = stats._strategy
        if len(strategy_instance.direction) > 1:
            if strategy_instance.direction[-2] == -1 and strategy_instance.direction[-1] == 1:
                print(f"✅ Buy signal detected for {stock_symbol}")
                return stock_symbol
    except Exception:
        # Silently ignore errors for individual stocks to not break the whole scan.
        # A more robust solution could log these errors to a separate file.
        return None
    
    return None

def find_buy_signals(data_dir, strategy_class, start_date='2020-01-01', end_date='2025-07-08'):
    """
    Scans through all CSV files in a directory in parallel to find stocks 
    with a buy signal on the last day.
    """
    files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    n_cores = cpu_count()
    print(f"Found {len(files)} CSV files. Starting analysis using {n_cores} CPU cores...")

    # Use functools.partial to create a new function with some arguments pre-filled.
    # This is a clean way to pass extra arguments to the worker function in pool.map.
    process_func = partial(analyze_stock, 
                           data_dir=data_dir, 
                           strategy_class=strategy_class, 
                           start_date=start_date, 
                           end_date=end_date)

    # A Process Pool manages a set of worker processes.
    with Pool(n_cores) as p:
        # map applies the process_func to every item in the 'files' list.
        # The processing is distributed among the worker processes.
        results = p.map(process_func, files)
    
    # Filter out None values from the results to get the final list of stocks.
    buy_signal_stocks = [stock for stock in results if stock]
    
    return buy_signal_stocks

if __name__ == "__main__":
    # 1. 定义要使用的策略
    STRATEGY_TO_SCAN = MlAdaptiveSuperTrendStrategy

    # 2. 定义数据目录 (请确保路径正确)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'tushare_data', 'daily')

    # 3. 运行扫描
    print("Starting stock scan for buy signals...")
    flagged_stocks = find_buy_signals(
        data_dir=data_dir,
        strategy_class=STRATEGY_TO_SCAN,
        start_date='2024-01-01',
        end_date='2025-07-10' # Use a recent date for live scanning
    )

    # 4. 打印并保存为同花顺INI格式
    results_dir = os.path.join(script_dir, '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    output_filepath = os.path.join(results_dir, 'buy_signals.ini')

    if flagged_stocks:
        print("\n--- Scan Complete ---")
        print("Stocks with a buy signal found:")
        
        # --- 格式化为同花顺代码 (仅保留代码部分) ---
        formatted_stocks = []
        for stock_symbol in flagged_stocks:
            try:
                # 只保留点号（.）之前的部分作为股票代码
                code = stock_symbol.split('.')[0]
                formatted_stocks.append(code)
                print(f"- {stock_symbol} -> {code}")
            except IndexError:
                print(f"  - Warning: Skipping malformed stock symbol {stock_symbol}")
                continue
        
        # --- 写入新的INI文件 ---
        with open(output_filepath, 'w', encoding='gbk') as f:
            f.write("[自选股设置]\n")
            f.write(f"股票代码={','.join(formatted_stocks)}\n")
            f.write("板块名称=AI扫描结果\n")

        print(f"\nTongHuaShun import file saved to: {output_filepath}")
    else:
        print("\n--- Scan Complete ---")
        print("No stocks with a buy signal were found based on the criteria.")
        # 如果没有符合条件的股票，创建一个内容为空的INI文件
        with open(output_filepath, 'w', encoding='gbk') as f:
            f.write("[自选股设置]\n")
            f.write("股票代码=\n")
            f.write("板块名称=AI扫描结果\n")
        print(f"\nAn empty result file was created: {output_filepath}") 