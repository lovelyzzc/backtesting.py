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

# --- Setup Python Path to allow imports from parent directory ---
# This makes the script runnable from anywhere.
script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(script_path))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def run_backtest_for_file(filepath, strategy_class, **strategy_params):
    """Loads a single CSV and runs the backtest with given parameters."""
    try:
        data = pd.read_csv(filepath, index_col='Date', parse_dates=True)
    except Exception as e:
        print(f"Could not read {filepath}: {e}")
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

def process_file_wrapper(filename, data_dir, strategy_class, params):
    """Wrapper function to run backtest for a single file."""
    filepath = os.path.join(data_dir, filename)
    stats = run_backtest_for_file(filepath, strategy_class, **params)
    if stats is not None:
        stats['Stock'] = filename.replace('.csv', '')
        return stats
    return None

def run_parameter_optimization(strategy_class, param_grid, data_dir, results_dir):
    """
    对给定的策略、参数网格和数据目录运行参数优化。
    将详细和摘要结果保存到一个唯一的、带时间戳的目录中。
    """
    if not os.path.isdir(data_dir):
        print(f"错误: 数据目录 '{data_dir}' 未找到.")
        return

    # --- 为本次优化运行创建一个唯一的目录 ---
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_results_dir = os.path.join(results_dir, f"{strategy_class.__name__}_opt_{timestamp}")
    try:
        os.makedirs(run_results_dir, exist_ok=True)
        print(f"结果将保存在: '{run_results_dir}'")
    except Exception as e:
        print(f"创建结果目录失败: {e}")
        return

    all_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    if not all_files:
        print(f"在 '{data_dir}' 中未找到CSV文件.")
        return

    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(f"--- 开始对 {len(all_files)} 支股票使用 {len(param_combinations)} 组参数进行优化回测 ({strategy_class.__name__}) ---")

    optimization_results = []

    for params in tqdm(param_combinations, desc="参数优化进度"):
        all_stats_for_params = []
        
        with concurrent.futures.ProcessPoolExecutor() as executor:
            task = partial(process_file_wrapper, data_dir=data_dir, strategy_class=strategy_class, params=params)
            
            futures = [executor.submit(task, filename) for filename in all_files]
            
            param_str = ', '.join([f'{k}={v}' for k, v in params.items()])
            progress_desc = f"回测({param_str})"
            
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(all_files), desc=progress_desc, leave=False):
                try:
                    result = future.result()
                    if result is not None:
                        all_stats_for_params.append(result)
                except Exception as exc:
                    print(f"一个回测任务产生错误: {exc}")

        if all_stats_for_params:
            results_df = pd.DataFrame(all_stats_for_params)

            # --- 创建更详细的性能摘要，展示指标分布 ---
            summary = {
                'params': str(params),
                'Num Stocks Tested': len(results_df),
                'Total Trades': results_df['# Trades'].sum()
            }

            # 计算关键指标的描述性统计
            key_metrics = [
                'Sharpe Ratio', 'Return [%]', 'Max. Drawdown [%]',
                'Win Rate [%]', 'Profit Factor', 'SQN'
            ]
            # 确保只对结果中存在的列进行操作
            existing_metrics = [m for m in key_metrics if m in results_df.columns]
            
            if existing_metrics:
                stats_desc = results_df[existing_metrics].describe(percentiles=[.25, .5, .75])
                
                # 将分位数数据平铺到摘要中
                for metric in existing_metrics:
                    for percentile_label in ['25%', '50%', '75%']:
                        # 将 Pandas 的百分比标签转换为更清晰的列名
                        col_name = f'{metric} ({percentile_label})'
                        summary[col_name] = stats_desc.loc[percentile_label, metric]

            optimization_results.append(summary)

    if not optimization_results:
        print("\n--- 优化过程未产生任何有效结果 ---")
        return

    # 创建并保存主要的优化摘要报告
    optimization_summary_df = pd.DataFrame(optimization_results)

    # 将参数和关键统计数据移到前面，方便查看
    if not optimization_summary_df.empty:
        # 基于新的摘要结构更新列排序
        cols_to_front = [
            'params', 'Num Stocks Tested', 'Total Trades',
            'Sharpe Ratio (50%)', 'Return [%] (50%)', 'Max. Drawdown [%] (50%)'
        ]
        
        cols = list(optimization_summary_df.columns)
        existing_cols_to_front = [col for col in cols_to_front if col in cols]
        other_cols = [col for col in cols if col not in existing_cols_to_front]

        optimization_summary_df = optimization_summary_df[existing_cols_to_front + other_cols]
        
        # 使用中位数夏普比率进行排序
        sort_col = 'Sharpe Ratio (50%)'
        if sort_col in optimization_summary_df.columns:
            optimization_summary_df = optimization_summary_df.sort_values(by=sort_col, ascending=False)

    print("\n\n--- 参数优化结果汇总 ---")
    with pd.option_context('display.max_rows', None, 'display.width', 1000):
        print(optimization_summary_df)

    best_params_row = optimization_summary_df.iloc[0]

    print("\n\n--- 最佳参数组合 (基于最高平均夏普比率) ---")
    print(best_params_row)

    try:
        # 2. 保存优化摘要报告 (优化报告)
        summary_filename = "optimization_summary_report.csv"
        full_path_summary = os.path.join(run_results_dir, summary_filename)
        optimization_summary_df.to_csv(full_path_summary, index=False)
        print(f"\n优化摘要报告已保存至 '{full_path_summary}'")

        # 3. 保存最佳参数摘要 (总报告的一部分)
        best_params_filename = "best_parameters_summary.txt"
        full_path_best_params = os.path.join(run_results_dir, best_params_filename)
        with open(full_path_best_params, 'w', encoding='utf-8') as f:
            f.write("--- 最佳参数组合 (基于最高平均夏普比率) ---\n\n")
            f.write(best_params_row.to_string())
        print(f"最佳参数摘要已保存至 '{full_path_best_params}'")
        
        print(f"\n优化运行完成。结果保存在目录中: '{run_results_dir}'")

    except Exception as e:
        print(f"\n保存优化结果失败: {e}") 