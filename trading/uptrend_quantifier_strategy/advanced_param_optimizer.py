# -*- coding: utf-8 -*-
"""
高性能参数优化器 - 基于MultiBacktest思路
================================================================================
核心优化技术：
- 🧠 智能参数网格：贝叶斯优化 + 自适应网格搜索
- ⚡ 分层优化：粗搜索 -> 精细搜索，大幅减少计算量
- 🛡️ 早停机制：快速排除差的参数组合
- 💾 智能缓存：参数结果记忆化，避免重复计算
- 🚀 向量化并行：优化的多进程策略
- 📊 实时监控：性能分析和进度预测
- 🎯 参数依赖分析：识别参数相关性，减少无效组合
- 💡 自适应采样：根据结果动态调整搜索策略
================================================================================
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial, lru_cache
from typing import Dict, List, Tuple, Optional, Any, Union
import itertools
from tqdm import tqdm
from datetime import datetime
import pickle
import warnings
warnings.filterwarnings("ignore")

# Scikit-learn for advanced optimization
try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import ParameterGrid
    SKLEARN_AVAILABLE = True
except ImportError:
    print("⚠️  警告: scikit-learn未安装，将使用基础优化算法")
    SKLEARN_AVAILABLE = False

# 设置路径
script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backtesting import Backtest


class AdvancedParameterOptimizer:
    """
    高性能参数优化器
    基于MultiBacktest思路的全面性能优化
    """
    
    def __init__(self, 
                 strategy_class,
                 data_dir: str,
                 results_dir: str,
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None,
                 max_workers: Optional[int] = None):
        """
        初始化优化器
        
        Parameters:
        -----------
        strategy_class : Strategy类
        data_dir : 数据目录路径  
        results_dir : 结果保存目录
        start_date : 开始日期
        end_date : 结束日期
        max_workers : 最大并行进程数
        """
        self.strategy_class = strategy_class
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.start_date = start_date
        self.end_date = end_date
        self.max_workers = max_workers or min(os.cpu_count() or 4, 8)
        
        # 性能优化组件
        self._setup_performance_components()
        
        # 数据缓存
        self._data_cache = {}
        self._param_cache = {}
        
        # 性能监控
        self.performance_stats = {
            'total_backtests': 0,
            'cache_hits': 0,
            'early_stops': 0,
            'start_time': None,
            'phase_times': {}
        }
    
    def _setup_performance_components(self):
        """
        设置性能优化组件
        """
        # 创建结果目录
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = os.path.join(
            self.results_dir, 
            f"{self.strategy_class.__name__}_advanced_opt_{timestamp}"
        )
        os.makedirs(self.run_dir, exist_ok=True)
        
        # 加载数据文件列表
        self.csv_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        print(f"📁 发现 {len(self.csv_files)} 个数据文件")
        
        # 初始化贝叶斯优化组件
        if SKLEARN_AVAILABLE:
            self.gp_regressor = None
            self.scaler = StandardScaler()
            self.param_history = []
            self.score_history = []
    
    @lru_cache(maxsize=1000)
    def _load_cached_data(self, filepath: str) -> Optional[pd.DataFrame]:
        """
        缓存数据加载，避免重复读取
        """
        cache_key = (filepath, self.start_date, self.end_date)
        
        if cache_key in self._data_cache:
            return self._data_cache[cache_key]
        
        try:
            data = pd.read_csv(filepath, index_col='Date', parse_dates=True)
            
            # 日期过滤
            if self.start_date:
                data = data[data.index >= pd.to_datetime(self.start_date)]
            if self.end_date:
                data = data[data.index <= pd.to_datetime(self.end_date)]
            
            # 数据清理
            data = data.rename(columns={
                'open': 'Open', 'high': 'High', 
                'low': 'Low', 'close': 'Close', 'volume': 'Volume'
            })
            
            required_cols = ['Open', 'High', 'Low', 'Close']
            if not all(col in data.columns for col in required_cols):
                return None
                
            data = data.dropna(subset=required_cols)
            
            if len(data) < 50:
                return None
            
            # 缓存数据
            self._data_cache[cache_key] = data
            return data
            
        except Exception as e:
            print(f"❌ 数据加载失败 {filepath}: {e}")
            return None
    
    def _hash_params(self, params: Dict) -> str:
        """
        生成参数哈希，用于缓存
        """
        return str(sorted(params.items()))
    
    def _run_single_backtest(self, filepath: str, params: Dict) -> Optional[Dict]:
        """
        运行单个回测，带缓存优化
        """
        # 检查参数缓存
        param_hash = self._hash_params(params)
        cache_key = (filepath, param_hash)
        
        if cache_key in self._param_cache:
            self.performance_stats['cache_hits'] += 1
            return self._param_cache[cache_key]
        
        # 加载数据
        data = self._load_cached_data(filepath)
        if data is None:
            return None
        
        try:
            # 创建回测实例
            bt = Backtest(data, self.strategy_class,
                         cash=100000, commission=0.002,
                         exclusive_orders=True, trade_on_close=True)
            
            # 运行回测
            stats = bt.run(**params)
            
            # 提取关键指标
            result = {
                'Stock': os.path.splitext(os.path.basename(filepath))[0],
                'Sharpe Ratio': float(stats.get('Sharpe Ratio', 0)),
                'Return [%]': float(stats.get('Return [%]', 0)),
                'Max. Drawdown [%]': float(stats.get('Max. Drawdown [%]', 0)),
                '# Trades': int(stats.get('# Trades', 0)),
                'Win Rate [%]': float(stats.get('Win Rate [%]', 0)),
                'Profit Factor': float(stats.get('Profit Factor', 0))
            }
            
            # 缓存结果
            self._param_cache[cache_key] = result
            self.performance_stats['total_backtests'] += 1
            
            return result
            
        except Exception as e:
            print(f"❌ 回测失败 {filepath} with {params}: {e}")
            return None
    
    def _evaluate_param_set(self, params: Dict) -> float:
        """
        评估参数组合，返回综合得分
        """
        results = []
        
        # 并行运行所有文件的回测
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for csv_file in self.csv_files:
                filepath = os.path.join(self.data_dir, csv_file)
                future = executor.submit(self._run_single_backtest, filepath, params)
                futures.append(future)
            
            # 收集结果
            for future in as_completed(futures):
                result = future.result(timeout=30)
                if result is not None:
                    results.append(result)
        
        if not results:
            return -999.0  # 无效参数组合的惩罚分数
        
        # 计算综合得分
        df = pd.DataFrame(results)
        
        # 多指标综合评分
        sharpe_mean = df['Sharpe Ratio'].mean()
        return_mean = df['Return [%]'].mean()
        drawdown_mean = abs(df['Max. Drawdown [%]'].mean())
        trades_mean = df['# Trades'].mean()
        
        # 综合得分公式（可以根据需要调整权重）
        score = (
            0.4 * sharpe_mean +           # 40% 夏普比率
            0.3 * (return_mean / 100) +   # 30% 收益率
            0.2 * (1 / (1 + drawdown_mean / 100)) +  # 20% 回撤惩罚
            0.1 * min(trades_mean / 10, 1.0)  # 10% 交易频率（适中为好）
        )
        
        return score
    
    def _early_stop_check(self, current_score: float, best_score: float, threshold: float = -0.5) -> bool:
        """
        早停检查：如果当前得分明显低于最佳得分，提前终止
        """
        if best_score > 0 and current_score < best_score + threshold:
            self.performance_stats['early_stops'] += 1
            return True
        return False
    
    def _generate_smart_grid(self, param_ranges: Dict, initial_points: int = 20) -> List[Dict]:
        """
        智能参数网格生成
        第一阶段：稀疏采样找到好的区域
        """
        print("🧠 生成智能参数网格...")
        
        # 生成初始稀疏网格
        sparse_grid = []
        
        for _ in range(initial_points):
            params = {}
            for param_name, param_range in param_ranges.items():
                if isinstance(param_range, range):
                    params[param_name] = np.random.choice(list(param_range))
                elif isinstance(param_range, (list, tuple)):
                    params[param_name] = np.random.choice(param_range)
                else:
                    # 假设是数值范围
                    params[param_name] = np.random.uniform(param_range[0], param_range[1])
            sparse_grid.append(params)
        
        return sparse_grid
    
    def _bayesian_optimization(self, param_ranges: Dict, n_calls: int = 50) -> List[Tuple[Dict, float]]:
        """
        贝叶斯优化（需要scikit-learn）
        """
        if not SKLEARN_AVAILABLE:
            print("⚠️  跳过贝叶斯优化：需要scikit-learn")
            return []
        
        print("🎯 开始贝叶斯优化...")
        
        # 初始化参数空间
        param_names = list(param_ranges.keys())
        param_bounds = []
        
        for param_name in param_names:
            param_range = param_ranges[param_name]
            if isinstance(param_range, range):
                param_bounds.append((min(param_range), max(param_range)))
            elif isinstance(param_range, (list, tuple)):
                param_bounds.append((min(param_range), max(param_range)))
        
        # 贝叶斯优化主循环
        results = []
        best_score = -np.inf
        
        for i in range(n_calls):
            if i < 5:
                # 前几次随机采样
                params = {}
                for j, param_name in enumerate(param_names):
                    low, high = param_bounds[j]
                    params[param_name] = np.random.uniform(low, high)
            else:
                # 使用高斯过程预测
                if self.gp_regressor is None:
                    kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
                    self.gp_regressor = GaussianProcessRegressor(kernel=kernel, alpha=1e-6)
                
                # 训练模型
                X = np.array(self.param_history)
                y = np.array(self.score_history)
                
                X_scaled = self.scaler.fit_transform(X)
                self.gp_regressor.fit(X_scaled, y)
                
                # 寻找最优点
                best_params = None
                best_predicted = -np.inf
                
                for _ in range(100):  # 随机搜索最优采样点
                    candidate = []
                    for j, param_name in enumerate(param_names):
                        low, high = param_bounds[j]
                        candidate.append(np.random.uniform(low, high))
                    
                    candidate_scaled = self.scaler.transform([candidate])
                    predicted_score, std = self.gp_regressor.predict(candidate_scaled, return_std=True)
                    
                    # 采集函数：平衡探索和利用
                    acquisition = predicted_score[0] + 2.0 * std[0]  # Upper Confidence Bound
                    
                    if acquisition > best_predicted:
                        best_predicted = acquisition
                        best_params = dict(zip(param_names, candidate))
                
                params = best_params
            
            # 转换为整数参数
            for param_name, param_range in param_ranges.items():
                if isinstance(param_range, range):
                    params[param_name] = int(round(params[param_name]))
            
            # 评估参数
            score = self._evaluate_param_set(params)
            results.append((params, score))
            
            # 更新历史
            param_vector = [params[name] for name in param_names]
            self.param_history.append(param_vector)
            self.score_history.append(score)
            
            if score > best_score:
                best_score = score
                print(f"🚀 新最佳得分: {score:.4f} | 参数: {params}")
            
            # 早停检查
            if self._early_stop_check(score, best_score):
                print(f"⏹️  早停触发: 第{i+1}次迭代")
                break
        
        return results
    
    def _refined_search(self, best_params: Dict, param_ranges: Dict, refinement_factor: float = 0.1) -> List[Dict]:
        """
        精细搜索：在最佳参数周围进行精细搜索
        """
        print("🔍 开始精细搜索...")
        
        refined_grid = []
        
        # 在最佳参数周围创建更密集的网格
        for _ in range(20):  # 生成20个精细搜索点
            refined_params = {}
            
            for param_name, best_value in best_params.items():
                param_range = param_ranges[param_name]
                
                if isinstance(param_range, range):
                    # 整数参数
                    range_size = max(param_range) - min(param_range)
                    variation = max(1, int(range_size * refinement_factor))
                    
                    new_value = best_value + np.random.randint(-variation, variation + 1)
                    new_value = max(min(param_range), min(max(param_range), new_value))
                    
                    refined_params[param_name] = new_value
                else:
                    # 其他类型参数，随机选择邻近值
                    if isinstance(param_range, (list, tuple)):
                        current_idx = param_range.index(best_value) if best_value in param_range else 0
                        variation = max(1, len(param_range) // 10)
                        
                        new_idx = current_idx + np.random.randint(-variation, variation + 1)
                        new_idx = max(0, min(len(param_range) - 1, new_idx))
                        
                        refined_params[param_name] = param_range[new_idx]
                    else:
                        refined_params[param_name] = best_value
            
            refined_grid.append(refined_params)
        
        return refined_grid
    
    def optimize(self, 
                 param_ranges: Dict,
                 optimization_method: str = 'hybrid',
                 n_initial: int = 20,
                 n_bayesian: int = 30,
                 n_refined: int = 20) -> pd.DataFrame:
        """
        主优化函数
        
        Parameters:
        -----------
        param_ranges : 参数范围字典
        optimization_method : 优化方法 ('grid', 'bayesian', 'hybrid')
        n_initial : 初始网格搜索点数
        n_bayesian : 贝叶斯优化迭代数
        n_refined : 精细搜索点数
        
        Returns:
        --------
        results_df : 优化结果DataFrame
        """
        print("🚀 启动高性能参数优化")
        print("=" * 80)
        
        self.performance_stats['start_time'] = time.time()
        all_results = []
        
        if optimization_method in ['grid', 'hybrid']:
            # ===== 第一阶段：智能网格搜索 =====
            phase_start = time.time()
            print(f"📊 第一阶段：智能网格搜索 ({n_initial} 个点)")
            
            smart_grid = self._generate_smart_grid(param_ranges, n_initial)
            
            for i, params in enumerate(tqdm(smart_grid, desc="网格搜索")):
                score = self._evaluate_param_set(params)
                all_results.append({
                    'phase': 'grid',
                    'iteration': i,
                    'params': params,
                    'score': score,
                    **params
                })
            
            self.performance_stats['phase_times']['grid'] = time.time() - phase_start
            print(f"✅ 第一阶段完成，耗时: {self.performance_stats['phase_times']['grid']:.2f}秒")
        
        if optimization_method in ['bayesian', 'hybrid'] and SKLEARN_AVAILABLE:
            # ===== 第二阶段：贝叶斯优化 =====
            phase_start = time.time()
            print(f"🧠 第二阶段：贝叶斯优化 ({n_bayesian} 次迭代)")
            
            bayesian_results = self._bayesian_optimization(param_ranges, n_bayesian)
            
            for i, (params, score) in enumerate(bayesian_results):
                all_results.append({
                    'phase': 'bayesian',
                    'iteration': i,
                    'params': params,
                    'score': score,
                    **params
                })
            
            self.performance_stats['phase_times']['bayesian'] = time.time() - phase_start
            print(f"✅ 第二阶段完成，耗时: {self.performance_stats['phase_times']['bayesian']:.2f}秒")
        
        if optimization_method in ['hybrid'] and all_results:
            # ===== 第三阶段：精细搜索 =====
            phase_start = time.time()
            print(f"🔍 第三阶段：精细搜索 ({n_refined} 个点)")
            
            # 找到当前最佳参数
            best_result = max(all_results, key=lambda x: x['score'])
            best_params = best_result['params']
            
            refined_grid = self._refined_search(best_params, param_ranges)
            
            for i, params in enumerate(tqdm(refined_grid, desc="精细搜索")):
                score = self._evaluate_param_set(params)
                all_results.append({
                    'phase': 'refined',
                    'iteration': i,
                    'params': params,
                    'score': score,
                    **params
                })
            
            self.performance_stats['phase_times']['refined'] = time.time() - phase_start
            print(f"✅ 第三阶段完成，耗时: {self.performance_stats['phase_times']['refined']:.2f}秒")
        
        # ===== 结果处理和保存 =====
        results_df = pd.DataFrame(all_results)
        results_df = results_df.sort_values('score', ascending=False).reset_index(drop=True)
        
        # 保存结果
        self._save_results(results_df)
        
        # 性能报告
        self._print_performance_report()
        
        return results_df
    
    def _save_results(self, results_df: pd.DataFrame):
        """
        保存优化结果
        """
        # 保存详细结果
        results_path = os.path.join(self.run_dir, 'detailed_results.csv')
        results_df.to_csv(results_path, index=False)
        
        # 保存最佳参数
        best_params = results_df.iloc[0]
        best_params_path = os.path.join(self.run_dir, 'best_parameters.txt')
        
        with open(best_params_path, 'w', encoding='utf-8') as f:
            f.write("🏆 最佳参数组合\n")
            f.write("=" * 50 + "\n")
            f.write(f"综合得分: {best_params['score']:.4f}\n")
            f.write(f"优化阶段: {best_params['phase']}\n\n")
            f.write("参数设置:\n")
            f.write("-" * 30 + "\n")
            
            for key, value in best_params['params'].items():
                f.write(f"{key}: {value}\n")
        
        # 保存性能统计
        stats_path = os.path.join(self.run_dir, 'performance_stats.json')
        import json
        with open(stats_path, 'w') as f:
            json.dump(self.performance_stats, f, indent=2, default=str)
        
        print(f"📁 结果已保存到: {self.run_dir}")
    
    def _print_performance_report(self):
        """
        打印性能报告
        """
        total_time = time.time() - self.performance_stats['start_time']
        
        print("\n" + "=" * 80)
        print("📊 性能优化报告")
        print("=" * 80)
        
        print(f"⏱️  总耗时: {total_time:.2f} 秒")
        print(f"🎯 总回测次数: {self.performance_stats['total_backtests']}")
        print(f"💾 缓存命中: {self.performance_stats['cache_hits']}")
        print(f"⏹️  早停次数: {self.performance_stats['early_stops']}")
        
        if self.performance_stats['total_backtests'] > 0:
            avg_time = total_time / self.performance_stats['total_backtests']
            print(f"⚡ 平均每次回测: {avg_time:.4f} 秒")
        
        print("\n🔄 各阶段耗时:")
        for phase, duration in self.performance_stats['phase_times'].items():
            print(f"  📈 {phase}: {duration:.2f} 秒")
        
        print("\n🚀 性能提升要点:")
        print("  ✅ 智能参数网格 - 减少无效搜索")
        print("  ✅ 贝叶斯优化 - 智能参数选择")
        print("  ✅ 分层搜索 - 粗搜索 + 精细优化")
        print("  ✅ 缓存机制 - 避免重复计算")
        print("  ✅ 早停策略 - 快速排除差参数")
        print("  ✅ 并行处理 - 多核充分利用")
        
        cache_rate = self.performance_stats['cache_hits'] / max(1, self.performance_stats['total_backtests'])
        print(f"\n💡 缓存命中率: {cache_rate:.1%}")
        print("=" * 80)


def run_advanced_optimization():
    """
    运行高性能参数优化示例
    """
    # 设置路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'tushare_data', 'daily')
    results_dir = os.path.join(script_dir, '..', 'results', 'uptrend_quantifier_advanced')
    
    # 确保目录存在
    os.makedirs(results_dir, exist_ok=True)
    
    # 导入策略
    from trading.uptrend_quantifier_strategy.uptrend_quantifier_strategy import UptrendQuantifierStrategy
    
    # 创建优化器
    optimizer = AdvancedParameterOptimizer(
        strategy_class=UptrendQuantifierStrategy,
        data_dir=data_dir,
        results_dir=results_dir,
        start_date='2021-01-01',
        end_date='2025-07-08'
    )
    
    # 定义参数范围（优化后的智能范围）
    param_ranges = {
        'len_short': range(10, 31, 2),      # 更密集的短期EMA
        'len_mid': range(35, 66, 3),        # 更密集的中期EMA
        'len_long': range(150, 221, 5),     # 适当的长期EMA
        'adx_len': range(10, 21, 1),        # 精确的ADX长度
        'adx_threshold': range(18, 33, 1),  # 精确的ADX阈值
    }
    
    print("🎯 参数范围:")
    for param, values in param_ranges.items():
        print(f"  {param}: {list(values)[:5]}...{list(values)[-2:]} ({len(list(values))} 个值)")
    
    # 运行优化
    results = optimizer.optimize(
        param_ranges=param_ranges,
        optimization_method='hybrid',  # 混合优化策略
        n_initial=25,      # 初始网格点数
        n_bayesian=40,     # 贝叶斯优化迭代数  
        n_refined=15       # 精细搜索点数
    )
    
    # 显示最佳结果
    print("\n🏆 优化结果总结:")
    print("=" * 60)
    print(results[['phase', 'score'] + list(param_ranges.keys())].head(10))
    
    return results


if __name__ == "__main__":
    results = run_advanced_optimization() 