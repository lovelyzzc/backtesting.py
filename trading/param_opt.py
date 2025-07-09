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
import numpy as np
from typing import Optional, Dict, Any, List

# --- Setup Python Path to allow imports from parent directory ---
# This makes the script runnable from anywhere.
script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(script_path))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 全局缓存，避免重复读取相同文件
_data_cache = {}

def _load_and_cache_data(filepath: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    带缓存的数据加载函数，避免重复读取相同文件
    """
    cache_key = (filepath, start_date, end_date)
    
    if cache_key in _data_cache:
        return _data_cache[cache_key].copy()
    
    try:
        data = pd.read_csv(filepath, index_col='Date', parse_dates=True)
    except Exception as e:
        print(f"Could not read {filepath}: {e}")
        return None

    # 确保data是DataFrame类型
    if not isinstance(data, pd.DataFrame):
        return None

    # --- Date Range Filtering ---
    if start_date:
        data = data[data.index >= pd.to_datetime(start_date)]
    if end_date:
        data = data[data.index <= pd.to_datetime(end_date)]
    if len(data) == 0:  # 修复: 使用len(data) == 0替代data.empty
        return None

    # --- Data Cleaning and Preparation ---
    # 使用向量化操作进行列名重命名
    rename_map = {
        'open': 'Open',
        'high': 'High', 
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }
    data = data.rename(columns=rename_map)

    required_cols = ['Open', 'High', 'Low', 'Close']
    if not all(col in data.columns for col in required_cols):
        print(f"Skipping {filepath}: Missing one of the required columns (Open, High, Low, Close).")
        return None
        
    # 使用更高效的数据清理
    data = data.dropna(subset=required_cols)
    
    if len(data) < 50:
        return None
    
    # 缓存清理后的数据
    _data_cache[cache_key] = data.copy()
    return data

def run_backtest_for_file(filepath, strategy_class, start_date=None, end_date=None, **strategy_params):
    """Loads a single CSV and runs the backtest with given parameters."""
    data = _load_and_cache_data(filepath, start_date, end_date)
    if data is None:
        return None

    # 使用更高效的Backtest配置
    bt = Backtest(data, strategy_class,
                  cash=100000,
                  commission=.00075,
                  exclusive_orders=True,
                  trade_on_close=True)  # 添加这个参数可以提升性能

    try:
        stats = bt.run(**strategy_params)
        return stats
    except Exception as e:
        print(f"回测失败 {filepath}: {e}")
        return None

def process_file_wrapper(filename, data_dir, strategy_class, params, start_date=None, end_date=None):
    """Wrapper function to run backtest for a single file."""
    filepath = os.path.join(data_dir, filename)
    stats = run_backtest_for_file(filepath, strategy_class, start_date=start_date, end_date=end_date, **params)
    # 明确检查None
    if stats is not None:
        # 只提取数值型的统计数据，避免复杂对象的传输
        stats_dict = {}
        for key, value in stats.items():
            if isinstance(value, (int, float, np.number)) and not pd.isna(value):
                stats_dict[key] = float(value)
        stats_dict['Stock'] = filename.replace('.csv', '')
        return stats_dict
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

    # 使用更高效的并行处理策略
    max_workers = min(os.cpu_count() or 4, len(all_files))  # 动态调整worker数量
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 批量提交任务，减少overhead
        batch_size = max(1, len(all_files) // (max_workers * 2))
        file_batches = [all_files[i:i + batch_size] for i in range(0, len(all_files), batch_size)]
        
        futures = []
        for batch in file_batches:
            for filename in batch:
                futures.append(executor.submit(task, filename))
        
        # 使用更高效的进度条
        for future in tqdm(concurrent.futures.as_completed(futures), 
                          total=len(futures), 
                          desc=progress_desc, 
                          leave=False,
                          mininterval=0.5):  # 减少更新频率
            try:
                result = future.result(timeout=30)  # 添加超时
                if result is not None:
                    all_stats.append(result)
            except concurrent.futures.TimeoutError:
                print(f"回测任务超时")
            except Exception as exc:
                print(f"一个回测任务产生错误: {exc}")
                
    return all_stats

def _process_and_save_results(optimization_results, run_results_dir):
    """处理、排序、打印并保存最终的优化摘要。"""
    if not optimization_results:
        print("\n--- 优化过程未产生任何有效结果 ---")
        return

    # 使用更高效的DataFrame创建
    df = pd.DataFrame(optimization_results)

    # 向量化的列重排序
    cols_to_front = [
        'params', 'Num Stocks Tested', 'Total Trades',
        'Sharpe Ratio', 'Return [%]', 'Max. Drawdown [%]'
    ]
    cols = list(df.columns)
    existing_cols_to_front = [col for col in cols_to_front if col in cols]
    other_cols = [col for col in cols if col not in existing_cols_to_front]
    df = df[existing_cols_to_front + other_cols]

    # 更高效的排序
    sort_col = 'Sharpe Ratio'
    if sort_col in df.columns:
        df = df.sort_values(sort_col, ascending=False, na_position='last').reset_index(drop=True)

    print("\n\n--- 参数优化结果汇总 ---")
    # 限制显示的行数，提升输出性能
    display_rows = min(20, len(df))
    with pd.option_context('display.max_rows', display_rows, 'display.width', 1000, 'display.precision', 4):
        print(df.head(display_rows))

    if df.empty:
        print("\n--- 没有找到最佳参数 ---")
        return

    best_params_row = df.iloc[0]
    best_sharpe = best_params_row.get('Sharpe Ratio', 'N/A')
    print(f"\n\n--- 最佳参数组合 (基于最高平均夏普比率): {best_sharpe} ---")
    print(best_params_row)

    try:
        summary_filename = f"optimization_summary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        full_path_summary = os.path.join(run_results_dir, summary_filename)
        # 使用更高效的CSV写入
        df.to_csv(full_path_summary, index=False, float_format='%.4f')
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

    # 预过滤文件，避免后续重复检查
    all_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    if not all_files:
        print(f"在 '{data_dir}' 中未找到CSV文件.")
        return

    param_combinations = _get_param_combinations(param_grid)
    
    # 估算总体进度
    total_combinations = len(param_combinations)
    total_files = len(all_files)
    estimated_total_tasks = total_combinations * total_files
    
    print(f"--- 开始对 {total_files} 支股票使用 {total_combinations} 组参数进行优化回测 ({strategy_class.__name__}) ---")
    print(f"预计总任务数: {estimated_total_tasks}")

    optimization_results = []
    
    # 使用更精确的进度跟踪
    start_time = datetime.now()
    
    for i, params in enumerate(tqdm(param_combinations, desc="参数优化进度")):
        all_stats_for_params = _run_backtests_for_param_set(params, all_files, data_dir, strategy_class, start_date=start_date, end_date=end_date)

        if all_stats_for_params:
            # 使用向量化操作进行统计计算
            results_df = pd.DataFrame(all_stats_for_params)
            numeric_cols = results_df.select_dtypes(include=[np.number])
            mean_stats = numeric_cols.mean()
            
            # 确保mean_stats是Series类型
            if isinstance(mean_stats, pd.Series):
                mean_stats_dict = mean_stats.to_dict()
            else:
                mean_stats_dict = {}
            
            # 安全地获取交易次数
            total_trades = 0
            if '# Trades' in numeric_cols.columns:
                total_trades = numeric_cols['# Trades'].sum()
            
            summary = {
                'params': str(params),
                **mean_stats_dict,
                'Total Trades': total_trades,
                'Num Stocks Tested': len(results_df)
            }
            
            # 清理重复的交易计数
            if '# Trades' in summary:
                del summary['# Trades']
            
            optimization_results.append(summary)
            
        # 每完成25%输出进度信息
        if (i + 1) % max(1, total_combinations // 4) == 0:
            elapsed = datetime.now() - start_time
            progress = (i + 1) / total_combinations
            estimated_total = elapsed / progress
            remaining = estimated_total - elapsed
            print(f"进度: {progress:.1%}, 已用时: {elapsed}, 预计剩余: {remaining}")
            
    _process_and_save_results(optimization_results, run_results_dir) 