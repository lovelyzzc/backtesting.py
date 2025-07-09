# -*- coding: utf-8 -*-
"""
This is the main script to run the parameter optimization for the RangeFilterStrategy.
It uses the `run_parameter_optimization` function from the `param_opt` module
to test a grid of parameters and find the best combination based on Sharpe Ratio.
"""
import os
import sys
import numpy as np

# --- Setup Python Path to allow imports from parent directory ---
script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from trading.smc_lite.range_filter_strategy import RangeFilterStrategy
from trading.param_opt import run_parameter_optimization

if __name__ == '__main__':
    # --- 1. Define Directories ---
    # This assumes the script is run from the 'trading' directory's parent,
    # or that the 'trading' directory is in the Python path.
    # 3. 定义数据和结果目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'tushare_data', 'daily')
    results_dir = os.path.join(script_dir, '..', 'results')

    # --- 2. Define Parameter Grid for Optimization ---
    # This grid defines the range of values to test for each parameter.
    # The keys must match the parameter names in the Strategy class.
    # The target is to find the best combination of these parameters.
    param_grid = {
        # Test SMA/ATR lengths from 10 to 100 with a step of 10
        'n_len': range(5, 60, 5),
        # Test ATR multipliers from 1.5 to 4.0 with a step of 0.5
        'atr_multiplier': np.arange(1.0, 4.1, 0.5), 
    }
    print("--- Parameter Grid for Optimization ---")
    for key, value in param_grid.items():
        print(f"'{key}': {list(value)}")
    print("-" * 35)

    # --- 3. Define Date Range (Optional) ---
    # You can specify a start and end date for the backtest period.
    # This is useful for in-sample / out-of-sample testing.
    # Set to None to use all available data.
    start_date = '2021-01-01'
    end_date = '2025-07-08'
    
    # --- 4. Run the Optimization ---
    # The script will test all combinations of parameters on all CSV files in the data directory.
    # It will then rank the results by the mean Sharpe Ratio across all tested stocks
    # and save detailed reports in the results directory.
    run_parameter_optimization(
        strategy_class=RangeFilterStrategy,
        param_grid=param_grid,
        data_dir=data_dir,
        results_dir=results_dir,
        start_date=start_date,
        end_date=end_date
    )

    print(f"\nOptimization process finished. Check the '{results_dir}' folder for detailed reports.") 