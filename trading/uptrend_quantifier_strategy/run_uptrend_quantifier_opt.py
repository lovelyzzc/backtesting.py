# -*- coding: utf-8 -*-
"""
This is the main script to run the parameter optimization for the UptrendQuantifierStrategy.
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

from trading.uptrend_quantifier_strategy.uptrend_quantifier_strategy import UptrendQuantifierStrategy
from trading.param_opt import run_parameter_optimization

if __name__ == '__main__':
    # --- 1. Define Directories ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir,  '..', 'tushare_data', 'daily')
    results_dir = os.path.join(script_dir,  '..', 'results', 'uptrend_quantifier')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # --- 2. Define Parameter Grid for Optimization ---
    param_grid = {
        'len_short': range(5, 21, 5),        # Test short EMA lengths
        'len_mid': range(30, 61, 10),          # Test mid EMA lengths
        'len_long': range(160, 201, 20),       # Test long EMA lengths
        'adx_len': range(12, 17, 1),           # Test ADX lengths
        'adx_threshold': range(21, 31, 2),     # Test ADX strength threshold
    }
    print("--- Parameter Grid for Optimization ---")
    for key, value in param_grid.items():
        # Convert range/numpy.arange to list for clean printing
        try:
            param_list = list(value)
        except TypeError:
            param_list = value
        print(f"'{key}': {param_list}")
    print("-" * 35)

    # --- 3. Define Date Range (Optional) ---
    start_date = '2021-01-01'
    end_date = '2025-07-08'
    
    # --- 4. Run the Optimization ---
    run_parameter_optimization(
        strategy_class=UptrendQuantifierStrategy,
        param_grid=param_grid,
        data_dir=data_dir,
        results_dir=results_dir,
        start_date=start_date,
        end_date=end_date
    )

    print(f"\nOptimization process finished. Check the '{results_dir}' folder for detailed reports.") 