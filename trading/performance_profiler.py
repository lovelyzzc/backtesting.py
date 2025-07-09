# -*- coding: utf-8 -*-
"""
回测性能分析和优化工具
用于监控回测运行时间，识别性能瓶颈，并提供优化建议
"""

import time
import memory_profiler
import psutil
import os
import pandas as pd
from functools import wraps
from typing import Dict, List, Callable, Any
import cProfile
import pstats
from io import StringIO

class PerformanceProfiler:
    """
    性能分析器，用于监控回测过程的性能指标
    """
    
    def __init__(self):
        self.timings: Dict[str, List[float]] = {}
        self.memory_usage: List[float] = []
        self.cpu_usage: List[float] = []
        self.profiler = None
        
    def timer(self, name: str):
        """
        装饰器：测量函数执行时间
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                result = func(*args, **kwargs)
                end_time = time.perf_counter()
                
                execution_time = end_time - start_time
                if name not in self.timings:
                    self.timings[name] = []
                self.timings[name].append(execution_time)
                
                print(f"⏱️  {name}: {execution_time:.4f}秒")
                return result
            return wrapper
        return decorator
    
    @memory_profiler.profile
    def profile_memory(self, func: Callable, *args, **kwargs):
        """
        分析内存使用情况
        """
        return func(*args, **kwargs)
    
    def start_cpu_monitoring(self):
        """
        开始CPU使用率监控
        """
        self.cpu_usage.append(psutil.cpu_percent())
        
    def monitor_memory(self):
        """
        监控当前内存使用
        """
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        self.memory_usage.append(memory_mb)
        return memory_mb
    
    def profile_function(self, func: Callable, *args, **kwargs):
        """
        使用cProfile对函数进行详细分析
        """
        profiler = cProfile.Profile()
        profiler.enable()
        
        result = func(*args, **kwargs)
        
        profiler.disable()
        
        # 分析结果
        s = StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(20)  # 显示前20个最耗时的函数
        
        print("🔍 详细性能分析:")
        print(s.getvalue())
        
        return result
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        获取性能摘要报告
        """
        summary = {
            '函数执行时间': {},
            '平均内存使用': f"{sum(self.memory_usage) / len(self.memory_usage) if self.memory_usage else 0:.2f} MB",
            '峰值内存使用': f"{max(self.memory_usage) if self.memory_usage else 0:.2f} MB",
            '平均CPU使用': f"{sum(self.cpu_usage) / len(self.cpu_usage) if self.cpu_usage else 0:.2f}%"
        }
        
        for name, times in self.timings.items():
            summary['函数执行时间'][name] = {
                '总时间': f"{sum(times):.4f}秒",
                '平均时间': f"{sum(times) / len(times):.4f}秒",
                '调用次数': len(times),
                '最长时间': f"{max(times):.4f}秒"
            }
            
        return summary
    
    def print_summary(self):
        """
        打印性能摘要
        """
        summary = self.get_performance_summary()
        
        print("\n" + "="*60)
        print("📊 性能分析摘要报告")
        print("="*60)
        
        print(f"💾 平均内存使用: {summary['平均内存使用']}")
        print(f"🔺 峰值内存使用: {summary['峰值内存使用']}")
        print(f"⚡ 平均CPU使用: {summary['平均CPU使用']}")
        
        print("\n⏱️  函数执行时间统计:")
        for name, stats in summary['函数执行时间'].items():
            print(f"  {name}:")
            print(f"    总时间: {stats['总时间']}")
            print(f"    平均时间: {stats['平均时间']}")
            print(f"    调用次数: {stats['调用次数']}")
            print(f"    最长时间: {stats['最长时间']}")
            print()

def performance_optimization_tips():
    """
    性能优化建议
    """
    tips = [
        "🔧 使用numpy向量化操作替代pandas循环",
        "💾 启用数据缓存避免重复读取CSV文件", 
        "⚡ 使用ProcessPoolExecutor进行并行处理",
        "📊 减少不必要的DataFrame操作和复制",
        "🎯 预计算技术指标，避免重复计算",
        "🗂️ 使用更高效的数据类型（如int8替代int64）",
        "📈 批量处理文件而不是逐个处理",
        "🔍 使用cProfile识别性能瓶颈",
        "💡 考虑使用numba进行JIT编译加速",
        "📝 减少打印输出和日志记录的频率"
    ]
    
    print("\n" + "="*60)
    print("💡 性能优化建议")
    print("="*60)
    
    for tip in tips:
        print(tip)
    
    print("\n🎯 针对您的代码，主要优化已实施:")
    print("✅ 优化了trail_indicator函数，从O(N²)降到O(N)")
    print("✅ 添加了数据缓存机制")
    print("✅ 改进了并行处理策略")
    print("✅ 使用了更高效的数据类型")
    print("✅ 优化了策略中的重复计算")

# 使用示例装饰器
profiler = PerformanceProfiler()

def benchmark_optimization(original_func, optimized_func, *args, **kwargs):
    """
    对比优化前后的性能
    """
    print("🔄 性能对比测试开始...")
    
    # 测试原函数
    print("\n📊 测试原始版本...")
    start_time = time.perf_counter()
    start_memory = profiler.monitor_memory()
    
    try:
        original_result = original_func(*args, **kwargs)
        original_time = time.perf_counter() - start_time
        original_memory = profiler.monitor_memory() - start_memory
    except Exception as e:
        print(f"原始版本执行失败: {e}")
        return
    
    # 测试优化后的函数
    print("\n⚡ 测试优化版本...")
    start_time = time.perf_counter()
    start_memory = profiler.monitor_memory()
    
    try:
        optimized_result = optimized_func(*args, **kwargs)
        optimized_time = time.perf_counter() - start_time
        optimized_memory = profiler.monitor_memory() - start_memory
    except Exception as e:
        print(f"优化版本执行失败: {e}")
        return
    
    # 计算改进
    time_improvement = ((original_time - optimized_time) / original_time) * 100
    memory_improvement = ((original_memory - optimized_memory) / original_memory) * 100 if original_memory > 0 else 0
    
    print("\n" + "="*60)
    print("📈 性能对比结果")
    print("="*60)
    print(f"⏱️  执行时间:")
    print(f"   原始版本: {original_time:.4f}秒")
    print(f"   优化版本: {optimized_time:.4f}秒")
    print(f"   提升: {time_improvement:.1f}%")
    
    print(f"\n💾 内存使用:")
    print(f"   原始版本: {original_memory:.2f}MB")
    print(f"   优化版本: {optimized_memory:.2f}MB")
    print(f"   改进: {memory_improvement:.1f}%")

if __name__ == "__main__":
    performance_optimization_tips() 