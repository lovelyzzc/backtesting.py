# -*- coding: utf-8 -*-
"""
详细Stats指标参数优化示例
================================================================================
功能特性：
📊 记录每个参数组合的完整backtesting统计指标
📈 生成详细的性能分析报告  
💾 保存所有中间结果供后续分析
🎯 多维度参数优化和评估
📋 自动生成统计摘要和最佳参数推荐
================================================================================
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# 设置路径
script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from trading.uptrend_quantifier_strategy.uptrend_quantifier_strategy import UptrendQuantifierStrategy
from trading.uptrend_quantifier_strategy.advanced_param_optimizer import AdvancedParameterOptimizer


def run_detailed_stats_optimization():
    """
    运行详细统计指标的参数优化
    """
    print("🚀 启动详细Stats指标参数优化")
    print("=" * 60)
    
    # --- 1. 设置路径 ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'tushare_data', 'daily')
    results_dir = os.path.join(script_dir, '..', 'results', 'detailed_stats_optimization')
    
    # 确保目录存在
    os.makedirs(results_dir, exist_ok=True)
    
    # --- 2. 检查数据文件 ---
    if not os.path.exists(data_dir):
        print(f"❌ 数据目录不存在: {data_dir}")
        return
    
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    print(f"📁 发现 {len(csv_files)} 个数据文件")
    
    if len(csv_files) == 0:
        print("❌ 没有找到CSV数据文件")
        return
    
    # --- 3. 创建高级优化器 ---
    optimizer = AdvancedParameterOptimizer(
        strategy_class=UptrendQuantifierStrategy,
        data_dir=data_dir,
        results_dir=results_dir,
        start_date='2021-01-01',
        end_date='2025-07-08',
        max_workers=4  # 可以根据需要调整
    )
    
    # --- 4. 定义参数范围 ---
    param_ranges = {
        'len_short': range(5, 21, 5),        # 短期EMA: 5, 10, 15, 20
        'len_mid': range(30, 61, 10),        # 中期EMA: 30, 40, 50, 60
        'len_long': range(160, 201, 20),     # 长期EMA: 160, 180, 200
        'adx_len': range(12, 17, 1),         # ADX长度: 12, 13, 14, 15, 16
        'adx_threshold': range(21, 31, 2),   # ADX阈值: 21, 23, 25, 27, 29
    }
    
    print("\n🎯 参数优化范围:")
    total_combinations = 1
    for param, values in param_ranges.items():
        value_list = list(values)
        total_combinations *= len(value_list)
        print(f"  {param}: {value_list} ({len(value_list)} 个值)")
    
    print(f"\n💡 理论总组合数: {total_combinations:,}")
    print("🧠 将使用智能优化算法大幅减少实际测试数量\n")
    
    # --- 5. 运行优化 ---
    start_time = datetime.now()
    print(f"⏰ 开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        results = optimizer.optimize(
            param_ranges=param_ranges,
            optimization_method='hybrid',  # 混合优化：网格+贝叶斯+精细搜索
            n_initial=15,      # 初始网格搜索点数
            n_bayesian=25,     # 贝叶斯优化迭代数
            n_refined=10       # 精细搜索点数
        )
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        print(f"\n✅ 优化完成！")
        print(f"⏱️  总耗时: {duration.total_seconds():.2f} 秒")
        
        # --- 6. 结果分析 ---
        print("\n" + "=" * 60)
        print("📊 优化结果分析")
        print("=" * 60)
        
        if len(results) > 0:
            print(f"📈 测试的参数组合数: {len(results)}")
            print(f"🎯 实际减少比例: {(1 - len(results)/total_combinations)*100:.1f}%")
            
            # 显示前5个最佳结果
            print("\n🏆 前5名最佳参数组合:")
            print("-" * 40)
            top_5 = results.head(5)
            
            param_cols = list(param_ranges.keys())
            display_cols = ['score'] + param_cols
            
            for i, (idx, row) in enumerate(top_5.iterrows(), 1):
                print(f"\n第{i}名 (得分: {row['score']:.4f}):")
                for param in param_cols:
                    print(f"  {param}: {row[param]}")
        else:
            print("❌ 没有获得有效结果")
            
        # --- 7. 文件输出说明 ---
        print("\n" + "=" * 60)
        print("📁 生成的文件说明")
        print("=" * 60)
        print("以下文件已保存到结果目录:")
        print(f"📂 {optimizer.run_dir}")
        print()
        print("🎯 核心结果文件:")
        print("  📄 detailed_results.csv - 优化过程的汇总结果")
        print("  📄 best_parameters.txt - 最佳参数组合")
        print()
        print("📊 详细Stats文件 (新增功能):")
        print("  📈 parameter_detailed_stats.csv - 每个参数组合的完整stats指标")
        print("  📊 stats_summary_analysis.txt - 详细统计分析摘要")
        print()
        print("📋 性能监控文件:")
        print("  ⚡ performance_stats.json - 优化性能统计")
        print()
        print("💡 parameter_detailed_stats.csv 包含的指标:")
        print("  - Sharpe Ratio, Return [%], Max. Drawdown [%]")
        print("  - Volatility (Ann.) [%], Sortino Ratio, Calmar Ratio") 
        print("  - # Trades, Win Rate [%], Profit Factor")
        print("  - Best/Worst/Avg Trade [%], Expectancy [%]")
        print("  - Trade Duration统计, SQN等")
        print("  - 以及每个参数组合的具体设置")
        
        return results
        
    except Exception as e:
        print(f"❌ 优化过程出错: {e}")
        import traceback
        traceback.print_exc()
        return None


def analyze_detailed_results(results_dir: str):
    """
    分析详细结果的辅助函数
    """
    detail_file = os.path.join(results_dir, 'parameter_detailed_stats.csv')
    
    if not os.path.exists(detail_file):
        print(f"❌ 详细结果文件不存在: {detail_file}")
        return
    
    df = pd.read_csv(detail_file)
    
    print("\n🔍 详细结果分析")
    print("=" * 40)
    print(f"📊 总记录数: {len(df)}")
    print(f"🏢 涉及股票数: {df['Stock'].nunique()}")
    print(f"🎯 参数组合数: {df['param_hash'].nunique()}")
    
    # 关键指标的分布
    key_metrics = ['Sharpe Ratio', 'Return [%]', 'Max. Drawdown [%]', 'Win Rate [%]']
    
    print("\n📈 关键指标分布:")
    for metric in key_metrics:
        if metric in df.columns:
            values = df[metric].dropna()
            print(f"\n{metric}:")
            print(f"  均值: {values.mean():.4f}")
            print(f"  中位数: {values.median():.4f}")
            print(f"  最优: {values.max():.4f}")
            print(f"  最差: {values.min():.4f}")


if __name__ == "__main__":
    print("🎯 详细Stats指标参数优化系统")
    print("功能: 记录每个参数组合的完整backtesting统计指标")
    print()
    
    # 运行优化
    results = run_detailed_stats_optimization()
    
    if results is not None and len(results) > 0:
        print("\n" + "="*60)
        print("🎉 优化成功完成!")
        print("💡 请查看生成的 parameter_detailed_stats.csv 文件")
        print("   该文件包含每个参数组合的详细统计指标")
        print("="*60)
    else:
        print("\n❌ 优化未能完成，请检查错误信息") 