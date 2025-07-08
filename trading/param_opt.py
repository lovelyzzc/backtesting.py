# -*- coding: utf-8 -*-
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
import traceback  # 添加traceback导入

# --- Setup Python Path to allow imports from parent directory ---
# This makes the script runnable from anywhere.
script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(script_path))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def run_backtest_for_file(filepath, strategy_class, start_date=None, end_date=None, **strategy_params):
    """Loads a single CSV and runs the backtest with given parameters."""
    try:
        data = pd.read_csv(filepath, index_col='Date', parse_dates=True)
    except Exception as e:
        print(f"Could not read {filepath}: {e}")
        return None

    # --- Date Range Filtering ---
    if start_date:
        data = data[data.index >= pd.to_datetime(start_date)]
    if end_date:
        data = data[data.index <= pd.to_datetime(end_date)]
    if data.empty:
        return None

    # --- Data Cleaning and Preparation ---
    data = data.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    })

    required_cols = ['Open', 'High', 'Low', 'Close']
    if not all(col in data.columns for col in required_cols):
        print(f"Skipping {filepath}: Missing one of the required columns (Open, High, Low, Close).")
        return None
        
    data.dropna(subset=required_cols, inplace=True)
    
    if len(data) < 50:
        return None

    bt = Backtest(data, strategy_class,
                  cash=100000,
                  commission=.00075,
                  exclusive_orders=True)

    stats = bt.run(**strategy_params)
    return stats

def process_file_wrapper(filename, data_dir, strategy_class, params, start_date=None, end_date=None):
    """Wrapper function to run backtest for a single file."""
    filepath = os.path.join(data_dir, filename)
    stats = run_backtest_for_file(filepath, strategy_class, start_date=start_date, end_date=end_date, **params)
    if stats is not None:
        stats['Stock'] = filename.replace('.csv', '')
        return stats
    return None

def _setup_results_dir(base_dir, strategy_name):
    """为优化运行创建并返回一个带时间戳的唯一目录。"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_results_dir = os.path.join(base_dir, f"{strategy_name}_opt_{timestamp}")
    try:
        os.makedirs(run_results_dir, exist_ok=True)
        print(f"结果将保存在: '{run_results_dir}'")
        return run_results_dir
    except Exception as e:
        print(f"创建结果目录失败: {e}")
        return None

def _get_param_combinations(param_grid):
    """从参数网格生成所有参数组合。"""
    keys, values = zip(*param_grid.items())
    return [dict(zip(keys, v)) for v in itertools.product(*values)]

def _run_backtests_for_param_set(params, all_files, data_dir, strategy_class, start_date=None, end_date=None):
    """为一组给定的参数在所有文件上并行运行回测。"""
    all_stats = []
    task = partial(process_file_wrapper, data_dir=data_dir, strategy_class=strategy_class, params=params, start_date=start_date, end_date=end_date)
    param_str = ', '.join([f'{k}={v}' for k, v in params.items()])
    progress_desc = f"回测({param_str})"

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(task, filename) for filename in all_files]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(all_files), desc=progress_desc, leave=False):
            try:
                result = future.result()
                if result is not None:
                    all_stats.append(result)
            except Exception as exc:
                print(f"一个回测任务产生错误: {exc}")
                print("完整错误堆栈:")
                traceback.print_exc()
    return all_stats

def _process_and_save_results(optimization_results, run_results_dir):
    """处理、排序、打印并保存最终的优化摘要。"""
    if not optimization_results:
        print("\n--- 优化过程未产生任何有效结果 ---")
        return

    df = pd.DataFrame(optimization_results)

    # Reorder columns for better readability
    cols_to_front = [
        'params', 'Num Stocks Tested', 'Total Trades',
        'Sharpe Ratio', 'Return [%]', 'Max. Drawdown [%]'
    ]
    cols = list(df.columns)
    existing_cols_to_front = [col for col in cols_to_front if col in cols]
    other_cols = [col for col in cols if col not in existing_cols_to_front]
    df = df[existing_cols_to_front + other_cols]

    # Sort by Sharpe Ratio
    sort_col = 'Sharpe Ratio'
    if sort_col in df.columns:
        df = df.sort_values(by=[sort_col], ascending=False).reset_index(drop=True)

    print("\n\n--- 参数优化结果汇总 ---")
    with pd.option_context('display.max_rows', None, 'display.width', 1000):
        print(df)

    if df.empty:
        print("\n--- 没有找到最佳参数 ---")
        return

    best_params_row = df.iloc[0]
    print("\n\n--- 最佳参数组合 (基于最高平均夏普比率) ---")
    print(best_params_row)

    try:
        summary_filename = "optimization_summary_report.csv"
        full_path_summary = os.path.join(run_results_dir, summary_filename)
        df.to_csv(full_path_summary, index=False)
        print(f"\n优化摘要报告已保存至 '{full_path_summary}'")

        best_params_filename = "best_parameters_summary.txt"
        full_path_best_params = os.path.join(run_results_dir, best_params_filename)
        with open(full_path_best_params, 'w', encoding='utf-8') as f:
            f.write("--- 最佳参数组合 (基于最高平均夏普比率) ---\n\n")
            f.write(best_params_row.to_string())
        print(f"最佳参数摘要已保存至 '{full_path_best_params}'")
        
        print(f"\n优化运行完成。结果保存在目录中: '{run_results_dir}'")

    except Exception as e:
        print(f"\n保存优化结果失败: {e}")

def run_parameter_optimization(strategy_class, param_grid, data_dir, results_dir, start_date=None, end_date=None):
    """
    对给定的策略、参数网格和数据目录运行参数优化。
    将详细和摘要结果保存到一个唯一的、带时间戳的目录中。
    """
    if not os.path.isdir(data_dir):
        print(f"错误: 数据目录 '{data_dir}' 未找到.")
        return

    run_results_dir = _setup_results_dir(results_dir, strategy_class.__name__)
    if not run_results_dir:
        return

    all_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    if not all_files:
        print(f"在 '{data_dir}' 中未找到CSV文件.")
        return

    param_combinations = _get_param_combinations(param_grid)

    print(f"--- 开始对 {len(all_files)} 支股票使用 {len(param_combinations)} 组参数进行优化回测 ({strategy_class.__name__}) ---")

    optimization_results = []
    for params in tqdm(param_combinations, desc="参数优化进度"):
        all_stats_for_params = _run_backtests_for_param_set(params, all_files, data_dir, strategy_class, start_date=start_date, end_date=end_date)

        if all_stats_for_params:
            results_df = pd.DataFrame(all_stats_for_params)
            mean_series = results_df.select_dtypes(include='number').mean()
            mean_stats = mean_series.to_dict()
            
            summary = {
                'params': str(params),
                **mean_stats,
                'Total Trades': results_df['# Trades'].sum(),
                'Num Stocks Tested': len(results_df)
            }
            if '# Trades' in summary:
                del summary['# Trades']
            
            optimization_results.append(summary)
            
    _process_and_save_results(optimization_results, run_results_dir) 