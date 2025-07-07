# -*- coding: utf-8 -*-
# @Author: lovelyzzc
# @Date: 2024-07-31
#
# backtesting.py momentum_trail_strategy
#
# Before running, please ensure you have the required libraries installed:
# pip install backtesting pandas pandas-ta tqdm
#
# This script is a Python implementation of the "Momentum Trail Strategy".
# It iterates through all stock data files in the 'stock_data_cleaned'
# directory and runs a backtest for each.

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='pandas_ta')

import os
import sys
import pandas as pd
from backtesting import Backtest
from tqdm import tqdm
import itertools
import concurrent.futures
from functools import partial
from datetime import datetime

# --- Setup Python Path to allow imports from parent directory ---
# This makes the script runnable from anywhere.
script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(script_path))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from trading.strategies import MomentumTrailStrategy


# ━━━━━━━━━━━━━━━━ 1. BACKTEST EXECUTION ━━━━━━━━━━━━━━━━

def run_backtest_for_file(filepath, strategy_class, **strategy_params):
    """Loads a single CSV and runs the backtest with given parameters."""
    try:
        data = pd.read_csv(filepath, index_col='Date', parse_dates=True)
    except Exception as e:
        print(f"Could not read {filepath}: {e}")
        return None

    # --- Data Cleaning and Preparation ---
    # Ensure correct column names for backtesting.py: Open, High, Low, Close, Volume
    # The library is case-sensitive.
    data = data.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    })

    # Ensure required columns are present
    required_cols = ['Open', 'High', 'Low', 'Close']
    if not all(col in data.columns for col in required_cols):
        print(f"Skipping {filepath}: Missing one of the required columns (Open, High, Low, Close).")
        return None
        
    # Drop rows with missing values that might affect calculations
    data.dropna(subset=required_cols, inplace=True)
    
    if len(data) < 50: # Need enough data for indicator calculation
        #print(f"Skipping {filepath}: Not enough data rows.")
        return None

    bt = Backtest(data, strategy_class,
                  cash=10000,
                  commission=.00075, # 0.075%
                  exclusive_orders=True) # Stop-and-reverse behavior

    stats = bt.run(**strategy_params)
    return stats

def process_file_wrapper(filename, data_dir, strategy_class, params):
    """
    Wrapper function to run backtest for a single file.
    This makes it easier to use with ProcessPoolExecutor.
    """
    filepath = os.path.join(data_dir, filename)
    stats = run_backtest_for_file(filepath, strategy_class, **params)
    if stats is not None:
        stats['Stock'] = filename.replace('.csv', '')
        return stats
    print(f"Skipping {filepath}: No stats.")
    return None

if __name__ == "__main__":
    # The strategy to be tested is now easily switchable
    STRATEGY_TO_TEST = MomentumTrailStrategy

    # Construct the path to the data directory relative to the script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'stock_data', 'stock_data_cleaned')

    if not os.path.isdir(data_dir):
        print(f"错误: 数据目录 '{data_dir}' 未找到.")
        exit()

    # Get all CSV files from the directory
    all_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

    if not all_files:
        print(f"在 '{data_dir}' 中未找到CSV文件.")
        exit()

    # --- 2. 定义参数优化的网格 ---
    # 在这里定义你想要测试的参数范围
    param_grid = {
        'osc_len': [15, 21, 30],
        'smth_len': [15, 21, 30],
        'trail_len': [3, 5, 7],
        'trail_mult': [10.0, 12.0, 15.0]
    }

    # 生成所有参数组合
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(f"--- 开始对 {len(all_files)} 支股票使用 {len(param_combinations)} 组参数进行优化回测 ({STRATEGY_TO_TEST.__name__}) ---")

    optimization_results = []

    # --- 3. 遍历每一种参数组合 ---
    for params in tqdm(param_combinations, desc="参数优化进度"):
        all_stats_for_params = []
        
        # --- 使用多进程并行处理文件回测 ---
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # 使用 functools.partial 预先填充 data_dir 和 params 参数
            task = partial(process_file_wrapper, data_dir=data_dir, strategy_class=STRATEGY_TO_TEST, params=params)
            
            # 将任务提交到进程池，并使用 tqdm 显示进度
            futures = [executor.submit(task, filename) for filename in all_files]
            
            progress_desc = f"回测(osc_len={params['osc_len']}, smth_len={params['smth_len']})"
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(all_files), desc=progress_desc, leave=False):
                try:
                    result = future.result()
                    if result is not None:
                        all_stats_for_params.append(result)
                except Exception as exc:
                    print(f"一个回测任务产生错误: {exc}")

        # --- 4. 汇总当前参数组合的表现 ---
        if all_stats_for_params:
            results_df = pd.DataFrame(all_stats_for_params)
            
            # 计算这组参数在所有回测股票上的平均表现
            # 我们关心一些关键指标的均值，例如夏普比率、回报率等
            summary = {
                'params': params,
                'Avg. Sharpe Ratio': results_df['Sharpe Ratio'].mean(),
                'Avg. Return [%]': results_df['Return [%]'].mean(),
                'Avg. Win Rate [%]': results_df['Win Rate [%]'].mean(),
                'Total Trades': results_df['# Trades'].sum(),
                'Num Stocks Tested': len(results_df)
            }
            optimization_results.append(summary)

    # --- 5. 分析优化结果 ---
    if not optimization_results:
        print("\n--- 优化过程未产生任何有效结果 ---")
    else:
        # 将优化汇总结果转换为DataFrame
        optimization_summary_df = pd.DataFrame(optimization_results)
        
        print("\n\n--- 参数优化结果汇总 ---")
        with pd.option_context('display.max_rows', None, 'display.width', 1000):
            print(optimization_summary_df.sort_values('Avg. Sharpe Ratio', ascending=False))

        # 找到夏普比率最高的最佳参数
        best_params_row = optimization_summary_df.loc[optimization_summary_df['Avg. Sharpe Ratio'].idxmax()]

        print("\n\n--- 最佳参数组合 (基于最高平均夏普比率) ---")
        print(best_params_row)

        # Optionally, save the results to a CSV file
        try:
            # 创建 results 目录 (如果不存在)
            results_dir = os.path.join(script_dir, 'results')
            os.makedirs(results_dir, exist_ok=True)

            # 生成带时间戳的文件名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_filename = f"{STRATEGY_TO_TEST.__name__}_optimization_results_{timestamp}.csv"
            full_path = os.path.join(results_dir, results_filename)
            
            optimization_summary_df.to_csv(full_path, index=False)
            print(f"\n优化结果已保存至 '{full_path}'")
        except Exception as e:
            print(f"\n保存优化结果失败: {e}")