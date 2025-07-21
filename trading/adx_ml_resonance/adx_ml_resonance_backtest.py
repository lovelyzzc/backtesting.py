# -*- coding: utf-8 -*-
"""
ADX ML共振策略回测脚本

使用backtesting.py框架对ADX ML共振策略进行参数优化和回测。
该策略结合了ADX（趋向指标）和ML自适应SuperTrend的共振信号。

运行前请确保已安装必要的库：
pip install backtesting pandas pandas-ta tqdm scikit-learn

使用方法：
python adx_ml_resonance_backtest.py
"""

import os
import sys

# --- Setup Python Path to allow imports from parent directory ---
script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from trading.adx_ml_resonance.strategies import AdxMlResonanceStrategy
from trading.param_opt import run_parameter_optimization

if __name__ == "__main__":
    # 1. 定义要测试的策略
    STRATEGY_TO_TEST = AdxMlResonanceStrategy

    # 2. 定义参数优化的网格
    # AdxMlResonanceStrategy 主要参数:
    # - adx_length: ADX计算周期
    # - adx_threshold: ADX强度阈值  
    # - ml_atr_len: ML SuperTrend的ATR周期
    # - ml_fact: ML SuperTrend的倍数因子
    # - di_threshold: DI差值阈值
    # - volume_ma_period: 成交量移动平均周期
    # - stop_loss_pct: 止损百分比
    # - take_profit_pct: 止盈百分比
    param_grid = {
        'adx_length': [10],                    # ADX计算周期
        'adx_threshold': [25],                 # ADX强度阈值
        'ml_atr_len': [10],                     # ML SuperTrend ATR周期
        'ml_fact': [3.0],                   # ML SuperTrend倍数
        'di_threshold': [5],                     # DI差值阈值
        'volume_ma_period': [10],               # 成交量过滤周期
        'stop_loss_pct': [0.05],          # 止损百分比
        'take_profit_pct': [0.08, 0.10, 0.15, 0.20, 0.25],        # 止盈百分比
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
    print("ADX ML共振策略回测配置")
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