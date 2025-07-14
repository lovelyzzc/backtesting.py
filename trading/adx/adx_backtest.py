# -*- coding: utf-8 -*-
import os
import sys

# --- Setup Python Path to allow imports from parent directory ---
script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from trading.adx.strategies import AdxStrategy
from trading.param_opt import run_parameter_optimization

if __name__ == "__main__":
    # 1. 定义要测试的策略
    STRATEGY_TO_TEST = AdxStrategy

    # 2. 定义参数优化的网格
    # AdxStrategy 参数: di_length, adx_threshold
    param_grid = {
        'di_length': range(10, 31, 5),      # e.g., 10, 15, 20, 25, 30
        'adx_threshold': [20, 25, 30],      # e.g., 20, 25, 30
        'di_diff_threshold': range(5, 26, 5), # e.g., 5, 10, 15, 20, 25
    }

    # 3. 定义数据和结果目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 注意: 请确保您的CSV数据文件存放在这个目录中
    data_dir = os.path.join(script_dir, '..', 'tushare_data', 'daily')
    results_dir = os.path.join(script_dir, '..', 'results')

    # 4. 运行参数优化
    run_parameter_optimization(
        strategy_class=STRATEGY_TO_TEST,
        param_grid=param_grid,
        data_dir=data_dir,
        results_dir=results_dir,
        start_date='2023-01-01',
        end_date='2025-07-08'
    ) 