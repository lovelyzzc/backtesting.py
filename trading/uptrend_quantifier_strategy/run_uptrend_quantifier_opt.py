# -*- coding: utf-8 -*-
"""
针对UptrendQuantifierStrategyOptimizedFixed策略的参数优化脚本
保存每组参数的汇总统计结果（类似optimization_summary_report.csv）
"""
import os
import sys
import numpy as np

# --- Setup Python Path to allow imports from parent directory ---
script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from trading.uptrend_quantifier_strategy.uptrend_quantifier_strategy import UptrendQuantifierStrategy
from trading.param_opt import run_parameter_optimization

if __name__ == '__main__':
    # --- 1. Define Directories ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir,  '..', 'tushare_data', 'daily')
    results_dir = os.path.join(script_dir,  '..', 'results', 'uptrend_quantifier_optimized')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # --- 2. Define Parameter Grid for Optimization ---
    param_grid = {
        'len_short': range(10, 31, 5),        # 测试短期EMA长度: [10, 15, 20, 25, 30]
        'len_mid': range(40, 71, 10),         # 测试中期EMA长度: [40, 50, 60, 70]
        'len_long': range(150, 251, 25),      # 测试长期EMA长度: [150, 175, 200, 225, 250]
        'adx_len': range(12, 19, 2),          # 测试ADX长度: [12, 14, 16, 18]
        'adx_threshold': range(20, 36, 5),    # 测试ADX阈值: [20, 25, 30, 35]
        'stop_loss_pct': [0.03, 0.05, 0.07, 0.10],  # 测试止损百分比
    }
    
    # 计算参数组合数量
    total_combinations = 1
    for key, values in param_grid.items():
        total_combinations *= len(values)
    
    print("--- 优化后策略的参数网格 ---")
    for key, value in param_grid.items():
        # Convert range/numpy.arange to list for clean printing
        try:
            param_list = list(value)
        except TypeError:
            param_list = value
        print(f"'{key}': {param_list}")
    print(f"总参数组合数: {total_combinations}")
    print("-" * 35)

    # --- 3. Define Date Range ---
    start_date = '2021-01-01'
    end_date = '2025-01-01'
    
    # --- 4. Run the Summary Optimization ---
    run_parameter_optimization(
        strategy_class=UptrendQuantifierStrategy,
        param_grid=param_grid,
        data_dir=data_dir,
        results_dir=results_dir,
        start_date=start_date,
        end_date=end_date,
    )

    print(f"\n参数汇总优化过程完成。检查 '{results_dir}' 文件夹获取汇总报告。") 