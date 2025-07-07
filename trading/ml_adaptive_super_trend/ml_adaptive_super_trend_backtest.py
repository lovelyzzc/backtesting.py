# -*- coding: utf-8 -*-
import os
import sys

# --- Setup Python Path to allow imports from parent directory ---
script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from trading.ml_adaptive_super_trend.strategies import MlAdaptiveSuperTrendStrategy
from trading.param_opt import run_parameter_optimization

if __name__ == "__main__":
    # 1. 定义要测试的策略
    STRATEGY_TO_TEST = MlAdaptiveSuperTrendStrategy

    # 2. 定义参数优化的网格
    # MlAdaptiveSuperTrendStrategy 参数:
    # atr_len, fact, training_data_period, highvol, midvol, lowvol
    param_grid = {
        'atr_len': [10, 20],
        'fact': [2.0, 3.0],
        'training_data_period': [100, 200]
        # 'highvol', 'midvol', 'lowvol' are kept as default for this example
    }

    # 3. 定义数据和结果目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 注意: 请确保您的CSV数据文件存放在这个目录中
    data_dir = os.path.join(script_dir, 'data') 
    results_dir = os.path.join(script_dir, 'results')

    # 4. 运行参数优化
    run_parameter_optimization(
        strategy_class=STRATEGY_TO_TEST,
        param_grid=param_grid,
        data_dir=data_dir,
        results_dir=results_dir
    ) 