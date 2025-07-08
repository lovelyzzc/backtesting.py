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
script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from trading.momentum_trail.strategies import MomentumTrailStrategy
from trading.param_opt import run_parameter_optimization


if __name__ == "__main__":
    # 1. 定义要测试的策略
    STRATEGY_TO_TEST = MomentumTrailStrategy

    # 2. 定义参数优化的网格
    param_grid = {
        'osc_len': [15, 21, 30],
        'smth_len': [15, 21, 30],
        'trail_len': [3, 5, 7],
        'trail_mult': [10.0, 12.0, 15.0]
    }

    # 3. 定义数据和结果目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'tushare_data', 'daily')
    results_dir = os.path.join(script_dir, '..', 'results')

    # 4. 运行参数优化
    run_parameter_optimization(
        strategy_class=STRATEGY_TO_TEST,
        param_grid=param_grid,
        data_dir=data_dir,
        results_dir=results_dir,
        start_date='2024-01-01',
        end_date='2025-07-08'
    )