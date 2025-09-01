# -*- coding: utf-8 -*-
"""
Reverse RSI策略回测脚本

使用backtesting.py框架对Reverse RSI策略进行参数优化和回测。
该策略基于RSI反向价格水平、SuperTrend趋势确认和RSI发散信号。

运行前请确保已安装必要的库：
pip install backtesting pandas pandas-ta tqdm scikit-learn

使用方法：
python reverse_rsi_backtest.py
"""

import os
import sys

# --- Setup Python Path to allow imports from parent directory ---
script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from trading.reverse_rsi.strategies import ReverseRsiStrategy, ReverseRsiLongOnlyStrategy
from trading.param_opt import run_parameter_optimization

if __name__ == "__main__":
    # 1. 定义要测试的策略
    STRATEGY_TO_TEST = ReverseRsiLongOnlyStrategy  # 可以改为 ReverseRsiStrategy 测试双向策略

    # 2. 定义参数优化的网格
    # ReverseRsiStrategy 主要参数:
    # - rsi_length: RSI计算周期
    # - smooth_bands: 是否平滑价格带
    # - st_factor: SuperTrend因子
    # - st_atr_len: SuperTrend ATR周期
    # - div_lookback: 发散检测回看周期
    # - use_price_breakout: 是否使用价格突破信号
    # - use_divergence: 是否使用发散信号
    # - use_trend_filter: 是否使用趋势过滤
    # - stop_loss_pct: 止损百分比
    # - take_profit_pct: 止盈百分比
    # - max_holding_days: 最大持仓天数
    param_grid = {
        'rsi_length': [14],                    # RSI计算周期
        'smooth_bands': [True],                # 是否平滑价格带
        'st_factor': [2.0, 2.4, 3.0],        # SuperTrend因子
        'st_atr_len': [10],                   # SuperTrend ATR周期
        'div_lookback': [3],                  # 发散检测回看周期
        'use_price_breakout': [True],         # 是否使用价格突破信号
        'use_divergence': [True, False],      # 是否使用发散信号
        'use_trend_filter': [True],           # 是否使用趋势过滤
        'stop_loss_pct': [0.04, 0.05, 0.06], # 止损百分比
        'take_profit_pct': [0.08, 0.10, 0.12, 0.15], # 止盈百分比
        'max_holding_days': [8, 10, 15],      # 最大持仓天数
    }

    # 3. 定义数据和结果目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'tushare_data', 'daily')
    results_dir = os.path.join(script_dir, '..', 'results')

    # 确保目录存在
    if not os.path.exists(data_dir):
        print(f"错误: 数据目录 {data_dir} 不存在")
        sys.exit(1)

    # 4. 显示配置信息
    print("=" * 60)
    print("Reverse RSI策略回测配置")
    print("=" * 60)
    print(f"策略: {STRATEGY_TO_TEST.__name__}")
    print(f"数据目录: {data_dir}")
    print(f"结果目录: {results_dir}")
    print("参数网格:")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")
    print("=" * 60)

    # 5. 运行参数优化
    try:
        run_parameter_optimization(
            strategy_class=STRATEGY_TO_TEST,
            param_grid=param_grid,
            data_dir=data_dir,
            results_dir=results_dir,
            start_date='2023-01-01',  # 回测开始日期
            end_date='2025-01-15'     # 回测结束日期
        )
        print("\n✅ 回测完成！请查看results目录中的结果文件。")
        
    except Exception as e:
        print(f"\n❌ 回测过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 