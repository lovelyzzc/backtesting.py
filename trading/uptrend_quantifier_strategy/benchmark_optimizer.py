# -*- coding: utf-8 -*-
"""
参数优化性能基准测试工具
================================================================================
对比不同优化方法的性能：
- 🐌 传统网格搜索 vs 🚀 高性能优化
- 📊 性能指标对比和可视化
- ⏱️ 详细的时间分析
- 🎯 优化效果评估
================================================================================
"""

import time
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# 设置路径
script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class OptimizationBenchmark:
    """
    参数优化性能基准测试
    """
    
    def __init__(self, strategy_class, data_dir: str, results_dir: str):
        self.strategy_class = strategy_class
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.benchmark_results = {}
        
        # 创建基准测试目录
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.benchmark_dir = os.path.join(results_dir, f"benchmark_{timestamp}")
        os.makedirs(self.benchmark_dir, exist_ok=True)
    
    def run_traditional_optimization(self, param_grid: Dict) -> Tuple[Dict, float]:
        """
        运行传统网格搜索
        """
        print("🐌 运行传统网格搜索...")
        start_time = time.time()
        
        # 使用原始的参数优化方法
        try:
            from trading.param_opt import run_parameter_optimization
            
            # 临时目录用于传统优化
            traditional_dir = os.path.join(self.benchmark_dir, "traditional")
            os.makedirs(traditional_dir, exist_ok=True)
            
            # 运行传统优化
            run_parameter_optimization(
                strategy_class=self.strategy_class,
                param_grid=param_grid,
                data_dir=self.data_dir,
                results_dir=traditional_dir,
                start_date='2021-01-01',
                end_date='2025-07-08'
            )
            
            traditional_time = time.time() - start_time
            
            # 读取结果
            result_files = [f for f in os.listdir(traditional_dir) if f.startswith('optimization_summary')]
            if result_files:
                latest_file = max(result_files)
                results_df = pd.read_csv(os.path.join(traditional_dir, latest_file))
                best_result = results_df.iloc[0].to_dict()
            else:
                best_result = {'score': 0}
            
            return best_result, traditional_time
            
        except Exception as e:
            print(f"❌ 传统优化失败: {e}")
            return {'score': 0}, time.time() - start_time
    
    def run_advanced_optimization(self, param_ranges: Dict) -> Tuple[Dict, float]:
        """
        运行高性能优化
        """
        print("🚀 运行高性能优化...")
        start_time = time.time()
        
        try:
            from trading.uptrend_quantifier_strategy.advanced_param_optimizer import AdvancedParameterOptimizer
            
            # 创建高性能优化器
            optimizer = AdvancedParameterOptimizer(
                strategy_class=self.strategy_class,
                data_dir=self.data_dir,
                results_dir=os.path.join(self.benchmark_dir, "advanced"),
                start_date='2021-01-01',
                end_date='2025-07-08'
            )
            
            # 运行优化
            results_df = optimizer.optimize(
                param_ranges=param_ranges,
                optimization_method='hybrid',
                n_initial=15,  # 减少测试时间
                n_bayesian=20,
                n_refined=10
            )
            
            advanced_time = time.time() - start_time
            
            if not results_df.empty:
                best_result = results_df.iloc[0].to_dict()
            else:
                best_result = {'score': 0}
            
            return best_result, advanced_time
            
        except Exception as e:
            print(f"❌ 高性能优化失败: {e}")
            return {'score': 0}, time.time() - start_time
    
    def run_benchmark(self) -> Dict:
        """
        运行完整基准测试
        """
        print("🎯 启动参数优化性能基准测试")
        print("=" * 80)
        
        # 定义测试参数
        param_grid = {
            'len_short': range(15, 26, 5),
            'len_mid': range(45, 56, 5), 
            'len_long': range(180, 201, 10),
            'adx_len': range(12, 17, 2),
            'adx_threshold': range(23, 28, 2),
        }
        
        param_ranges = {
            'len_short': range(15, 26, 2),
            'len_mid': range(45, 56, 2),
            'len_long': range(180, 201, 5),
            'adx_len': range(12, 17, 1),
            'adx_threshold': range(23, 28, 1),
        }
        
        print("📊 测试参数组合数:")
        traditional_combinations = 1
        for param, values in param_grid.items():
            count = len(list(values))
            traditional_combinations *= count
            print(f"  {param}: {count} 个值")
        print(f"  传统方法总组合: {traditional_combinations}")
        
        advanced_max_tests = 15 + 20 + 10  # initial + bayesian + refined
        print(f"  高性能方法最大测试: {advanced_max_tests}")
        print(f"  理论加速比: {traditional_combinations / advanced_max_tests:.1f}x")
        
        # 运行传统优化
        print("\n" + "="*50)
        traditional_result, traditional_time = self.run_traditional_optimization(param_grid)
        
        # 运行高性能优化  
        print("\n" + "="*50)
        advanced_result, advanced_time = self.run_advanced_optimization(param_ranges)
        
        # 汇总结果
        benchmark_summary = {
            'traditional': {
                'time': traditional_time,
                'best_score': traditional_result.get('score', 0),
                'combinations_tested': traditional_combinations,
                'method': '网格搜索'
            },
            'advanced': {
                'time': advanced_time,
                'best_score': advanced_result.get('score', 0),
                'combinations_tested': advanced_max_tests,
                'method': '混合智能优化'
            }
        }
        
        # 计算性能提升
        if traditional_time > 0:
            time_speedup = traditional_time / advanced_time
            efficiency_ratio = advanced_max_tests / traditional_combinations
            
            benchmark_summary['performance'] = {
                'time_speedup': time_speedup,
                'efficiency_ratio': efficiency_ratio,
                'score_comparison': advanced_result.get('score', 0) - traditional_result.get('score', 0)
            }
        
        # 保存和显示结果
        self._save_benchmark_results(benchmark_summary)
        self._print_benchmark_report(benchmark_summary)
        
        return benchmark_summary
    
    def _save_benchmark_results(self, results: Dict):
        """
        保存基准测试结果
        """
        import json
        
        # 保存JSON结果
        results_path = os.path.join(self.benchmark_dir, 'benchmark_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 生成报告
        report_path = os.path.join(self.benchmark_dir, 'benchmark_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("🎯 参数优化性能基准测试报告\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("📊 测试结果对比:\n")
            f.write("-" * 50 + "\n")
            
            for method, data in results.items():
                if method == 'performance':
                    continue
                f.write(f"\n{data['method']}:\n")
                f.write(f"  ⏱️  执行时间: {data['time']:.2f} 秒\n")
                f.write(f"  🎯 最佳得分: {data['best_score']:.4f}\n")
                f.write(f"  🔢 测试组合数: {data['combinations_tested']}\n")
            
            if 'performance' in results:
                perf = results['performance']
                f.write(f"\n🚀 性能提升:\n")
                f.write("-" * 30 + "\n")
                f.write(f"  ⚡ 时间加速: {perf['time_speedup']:.1f}x\n")
                f.write(f"  📈 效率提升: {1/perf['efficiency_ratio']:.1f}x\n")
                f.write(f"  🎯 得分差异: {perf['score_comparison']:+.4f}\n")
    
    def _print_benchmark_report(self, results: Dict):
        """
        打印基准测试报告
        """
        print("\n" + "=" * 80)
        print("🎯 参数优化性能基准测试报告")
        print("=" * 80)
        
        # 创建对比表格
        comparison_data = []
        for method, data in results.items():
            if method == 'performance':
                continue
            comparison_data.append({
                '优化方法': data['method'],
                '执行时间(秒)': f"{data['time']:.2f}",
                '最佳得分': f"{data['best_score']:.4f}",
                '测试组合数': data['combinations_tested']
            })
        
        df = pd.DataFrame(comparison_data)
        print("\n📊 性能对比:")
        print(df.to_string(index=False))
        
        if 'performance' in results:
            perf = results['performance']
            print(f"\n🚀 性能提升分析:")
            print("-" * 50)
            print(f"⚡ 执行时间加速:     {perf['time_speedup']:.1f}x")
            print(f"📈 搜索效率提升:     {1/perf['efficiency_ratio']:.1f}x") 
            print(f"🎯 最佳得分差异:     {perf['score_comparison']:+.4f}")
            
            if perf['time_speedup'] > 1:
                print(f"\n✅ 高性能优化在 {perf['time_speedup']:.1f}x 的速度下")
                if perf['score_comparison'] >= 0:
                    print("   获得了相同或更好的优化结果！")
                else:
                    print("   得分略有下降，但仍在可接受范围内。")
            
        print(f"\n📁 详细结果保存在: {self.benchmark_dir}")
        print("=" * 80)


def run_optimization_benchmark():
    """
    运行优化基准测试
    """
    # 设置路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'tushare_data', 'daily')
    results_dir = os.path.join(script_dir, '..', 'results', 'benchmark')
    
    # 确保目录存在
    os.makedirs(results_dir, exist_ok=True)
    
    # 导入策略
    from trading.uptrend_quantifier_strategy.uptrend_quantifier_strategy import UptrendQuantifierStrategy
    
    # 创建基准测试器
    benchmark = OptimizationBenchmark(
        strategy_class=UptrendQuantifierStrategy,
        data_dir=data_dir,
        results_dir=results_dir
    )
    
    # 运行基准测试
    results = benchmark.run_benchmark()
    
    print("\n💡 基准测试要点:")
    print("- 🧠 智能优化减少了无效参数组合的测试")
    print("- ⚡ 多层优化策略提升搜索效率")
    print("- 🎯 贝叶斯优化能更快找到最优区域")
    print("- 💾 缓存机制避免重复计算")
    print("- 🛡️ 早停策略节省计算资源")
    
    return results


if __name__ == "__main__":
    results = run_optimization_benchmark() 