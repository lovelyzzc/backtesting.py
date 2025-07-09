# -*- coding: utf-8 -*-
"""
使用 MultiBacktest 进行 RangeFilterStrategy 的参数优化。
这个版本利用 MultiBacktest 的并行处理能力，可以同时在多个股票上进行优化。
"""
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# --- Setup Python Path ---
script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from trading.smc_lite.range_filter_strategy import RangeFilterStrategy
from backtesting.lib import MultiBacktest

def load_data_files(data_dir, start_date=None, end_date=None, max_files=None):
    """
    加载数据目录中的所有CSV文件
    
    Args:
        data_dir: 数据目录路径
        start_date: 开始日期
        end_date: 结束日期 
        max_files: 最大文件数量（用于测试）
    
    Returns:
        list: 包含所有数据DataFrame的列表
        list: 对应的股票代码列表
    """
    data_files = list(Path(data_dir).glob("*.csv"))
    if max_files:
        data_files = data_files[:max_files]
    
    dataframes = []
    stock_codes = []
    
    print(f"Loading {len(data_files)} data files...")
    
    for file_path in data_files:
        try:
            df = pd.read_csv(file_path)
            
            # 数据预处理
            if 'trade_date' in df.columns:
                df['Date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
            elif 'date' in df.columns:
                df['Date'] = pd.to_datetime(df['date'])
            
            df.set_index('Date', inplace=True)
            df.sort_index(inplace=True)
            
            # 确保列名符合要求
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            column_mapping = {
                'open': 'Open', 'high': 'High', 'low': 'Low', 
                'close': 'Close', 'vol': 'Volume', 'volume': 'Volume'
            }
            
            for old_col, new_col in column_mapping.items():
                if old_col in df.columns and new_col not in df.columns:
                    df[new_col] = df[old_col]
            
            # 检查必需的列
            if not all(col in df.columns for col in required_columns):
                print(f"Skipping {file_path.name}: Missing required columns")
                continue
            
            # 日期过滤
            if start_date:
                df = df[df.index >= start_date]
            if end_date:
                df = df[df.index <= end_date]
            
            # 数据质量检查
            if len(df) < 100:  # 至少需要100个数据点
                print(f"Skipping {file_path.name}: Insufficient data ({len(df)} rows)")
                continue
            
            # 去除缺失值
            df = df.dropna()
            
            if len(df) >= 100:
                dataframes.append(df[required_columns])
                stock_codes.append(file_path.stem)
                
        except Exception as e:
            print(f"Error loading {file_path.name}: {e}")
            continue
    
    print(f"Successfully loaded {len(dataframes)} datasets")
    return dataframes, stock_codes

def run_multibacktest_optimization(
    strategy_class,
    param_grid,
    data_dir,
    results_dir,
    start_date=None,
    end_date=None,
    max_files=None,
    optimization_metric='Sharpe Ratio'
):
    """
    使用 MultiBacktest 进行参数优化
    
    Args:
        strategy_class: 策略类
        param_grid: 参数网格
        data_dir: 数据目录
        results_dir: 结果目录
        start_date: 开始日期
        end_date: 结束日期
        max_files: 最大文件数量
        optimization_metric: 优化指标
    """
    # 创建结果目录
    os.makedirs(results_dir, exist_ok=True)
    
    # 加载数据
    dataframes, stock_codes = load_data_files(
        data_dir, start_date, end_date, max_files
    )
    
    if not dataframes:
        print("No valid data files found!")
        return
    
    print(f"Running optimization on {len(dataframes)} datasets...")
    print(f"Optimization metric: {optimization_metric}")
    
    # 创建 MultiBacktest 实例
    multi_bt = MultiBacktest(dataframes, strategy_class)
    
    # 执行参数优化
    print("Starting parameter optimization...")
    try:
        heatmap_results = multi_bt.optimize(
            **param_grid,
            maximize=optimization_metric,
            constraint=lambda p: p.n_len >= 5  # 确保参数合理性
        )
        
        print(f"Optimization completed! Results shape: {heatmap_results.shape}")
        
        # 处理结果
        process_optimization_results(
            heatmap_results, stock_codes, results_dir, optimization_metric
        )
        
    except Exception as e:
        print(f"Optimization failed: {e}")
        import traceback
        traceback.print_exc()

def process_optimization_results(heatmap_results, stock_codes, results_dir, metric):
    """
    处理优化结果并生成报告
    
    Args:
        heatmap_results: MultiBacktest.optimize() 的结果
        stock_codes: 股票代码列表
        results_dir: 结果目录
        metric: 优化指标
    """
    print("Processing optimization results...")
    
    # 计算跨所有股票的平均表现
    mean_results = heatmap_results.mean(axis=1)
    best_params_idx = mean_results.idxmax()
    best_performance = mean_results.max()
    
    print(f"Best average {metric}: {best_performance:.4f}")
    print(f"Best parameters: {best_params_idx}")
    
    # 保存详细结果
    results_file = os.path.join(results_dir, "multibacktest_optimization_results.csv")
    
    # 创建结果DataFrame
    results_df = pd.DataFrame({
        'Parameters': heatmap_results.index,
        'Mean_Performance': mean_results.values,
        **{f'Stock_{i}_{code}': heatmap_results.iloc[:, i].values 
           for i, code in enumerate(stock_codes[:min(10, len(stock_codes))])}  # 只保存前10个股票的详细结果
    })
    
    # 按平均表现排序
    results_df = results_df.sort_values('Mean_Performance', ascending=False)
    results_df.to_csv(results_file, index=False)
    
    # 保存最佳参数
    best_params_file = os.path.join(results_dir, "best_parameters.txt")
    with open(best_params_file, 'w', encoding='utf-8') as f:
        f.write(f"=== MultiBacktest 参数优化结果 ===\n")
        f.write(f"优化指标: {metric}\n")
        f.write(f"测试股票数量: {len(stock_codes)}\n")
        f.write(f"最佳平均表现: {best_performance:.4f}\n")
        f.write(f"最佳参数组合: {best_params_idx}\n\n")
        
        f.write("=== 前10个最佳参数组合 ===\n")
        for i, (params, score) in enumerate(results_df.head(10)[['Parameters', 'Mean_Performance']].values):
            f.write(f"{i+1}. {params} -> {score:.4f}\n")
    
    # 生成可视化结果（如果可能的话）
    try:
        from backtesting.lib import plot_heatmaps
        plot_file = os.path.join(results_dir, "optimization_heatmap.html")
        plot_heatmaps(mean_results, filename=plot_file, open_browser=False)
        print(f"Heatmap saved to: {plot_file}")
    except Exception as e:
        print(f"Could not generate heatmap: {e}")
    
    print(f"Results saved to: {results_dir}")
    print(f"Detailed results: {results_file}")
    print(f"Best parameters: {best_params_file}")

if __name__ == '__main__':
    # --- 配置参数 ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'tushare_data', 'daily')
    results_dir = os.path.join(script_dir, '..', 'results', 'multibacktest')
    
    # 参数网格（与原来相同）
    param_grid = {
        'n_len': range(5, 60, 5),
        'atr_multiplier': np.arange(1.0, 4.1, 0.5),
    }
    
    print("=== MultiBacktest 参数优化 ===")
    print("参数网格:")
    for key, value in param_grid.items():
        print(f"  {key}: {list(value)}")
    
    # 日期范围
    start_date = '2021-01-01'
    end_date = '2025-07-08'
    
    # 运行优化（可以设置 max_files 进行测试）
    run_multibacktest_optimization(
        strategy_class=RangeFilterStrategy,
        param_grid=param_grid,
        data_dir=data_dir,
        results_dir=results_dir,
        start_date=start_date,
        end_date=end_date,
        max_files=None,  # 设置为 5 可以先测试
        optimization_metric='Sharpe Ratio'
    )
    
    print("\n=== 优化完成 ===") 