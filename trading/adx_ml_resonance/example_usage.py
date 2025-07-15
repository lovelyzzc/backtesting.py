# -*- coding: utf-8 -*-
"""
ADX ML共振指标使用示例

本文件展示了如何使用ADX ML共振指标以及如何运行相关的单元测试。
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Setup Python Path
script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from trading.adx_ml_resonance.indicators import adx_ml_resonance_indicator, simple_resonance_indicator


def generate_sample_data(n_periods=200, seed=42):
    """生成示例价格数据"""
    np.random.seed(seed)
    
    # 生成带趋势的价格数据
    base_price = 100
    trend = np.linspace(0, 20, n_periods)  # 上升趋势
    noise = np.random.normal(0, 2, n_periods)
    
    close_prices = base_price + trend + noise.cumsum() * 0.1
    
    # 生成high和low价格
    daily_range = np.random.uniform(1, 3, n_periods)
    high_prices = close_prices + daily_range * 0.6
    low_prices = close_prices - daily_range * 0.4
    
    return pd.Series(high_prices), pd.Series(low_prices), pd.Series(close_prices)


def example_basic_usage():
    """基本使用示例"""
    print("=== ADX ML共振指标基本使用示例 ===\n")
    
    # 生成示例数据
    high, low, close = generate_sample_data(200)
    
    print(f"生成了{len(close)}个价格数据点")
    print(f"价格范围: {close.min():.2f} - {close.max():.2f}")
    
    # 使用默认参数计算指标
    adx, di_plus, di_minus, ml_st, ml_direction, resonance_signal = adx_ml_resonance_indicator(
        high, low, close,
        adx_length=14,           # ADX计算周期
        adx_threshold=25,        # ADX强势阈值
        ml_atr_len=14,          # ML SuperTrend ATR周期
        ml_fact=3.0,            # ML SuperTrend倍数
        training_data_period=100, # ML训练数据周期
        highvol=0.75,           # 高波动率百分位
        midvol=0.5,             # 中波动率百分位
        lowvol=0.25             # 低波动率百分位
    )
    
    # 分析结果
    buy_signals = np.sum(resonance_signal == 1)
    sell_signals = np.sum(resonance_signal == -1)
    no_signals = np.sum(resonance_signal == 0)
    
    print(f"\n指标计算结果:")
    print(f"- ADX平均值: {np.nanmean(adx):.2f}")
    print(f"- DI+平均值: {np.nanmean(di_plus):.2f}")
    print(f"- DI-平均值: {np.nanmean(di_minus):.2f}")
    print(f"- ML SuperTrend平均值: {np.nanmean(ml_st):.2f}")
    
    print(f"\n共振信号统计:")
    print(f"- 买入信号: {buy_signals}个")
    print(f"- 卖出信号: {sell_signals}个")
    print(f"- 无信号: {no_signals}个")
    
    # 找出信号位置
    if buy_signals > 0:
        buy_positions = np.where(resonance_signal == 1)[0]
        print(f"- 买入信号位置: {buy_positions[:5]}..." if len(buy_positions) > 5 else f"- 买入信号位置: {buy_positions}")
    
    if sell_signals > 0:
        sell_positions = np.where(resonance_signal == -1)[0]
        print(f"- 卖出信号位置: {sell_positions[:5]}..." if len(sell_positions) > 5 else f"- 卖出信号位置: {sell_positions}")


def example_simple_indicator():
    """简化版指标使用示例"""
    print("\n=== 简化版共振指标使用示例 ===\n")
    
    high, low, close = generate_sample_data(150)
    
    # 使用简化版指标
    buy_signal, sell_signal, adx, ml_direction = simple_resonance_indicator(
        high, low, close,
        adx_length=10,
        adx_threshold=20,
        ml_atr_len=10,
        ml_fact=3.0,
        training_data_period=80
    )
    
    # 分析结果
    buy_count = np.sum(buy_signal)
    sell_count = np.sum(sell_signal)
    
    print(f"简化版指标结果:")
    print(f"- 买入信号数量: {buy_count}")
    print(f"- 卖出信号数量: {sell_count}")
    print(f"- ADX平均值: {np.nanmean(adx):.2f}")
    
    if buy_count > 0:
        buy_positions = np.where(buy_signal)[0]
        print(f"- 买入信号位置: {buy_positions}")
    
    if sell_count > 0:
        sell_positions = np.where(sell_signal)[0]
        print(f"- 卖出信号位置: {sell_positions}")


def example_parameter_sensitivity():
    """参数敏感性分析示例"""
    print("\n=== 参数敏感性分析示例 ===\n")
    
    high, low, close = generate_sample_data(150)
    
    # 测试不同的ADX阈值
    thresholds = [15, 20, 25, 30, 35]
    
    print("不同ADX阈值的信号数量:")
    print("阈值\t买入\t卖出\t总计")
    print("-" * 30)
    
    for threshold in thresholds:
        _, _, _, _, _, resonance_signal = adx_ml_resonance_indicator(
            high, low, close,
            adx_threshold=threshold,
            adx_length=10,
            ml_atr_len=10,
            ml_fact=3.0,
            training_data_period=80
        )
        
        buy_count = np.sum(resonance_signal == 1)
        sell_count = np.sum(resonance_signal == -1)
        total_count = buy_count + sell_count
        
        print(f"{threshold}\t{buy_count}\t{sell_count}\t{total_count}")


def create_visualization():
    """创建可视化图表（如果matplotlib可用）"""
    try:
        print("\n=== 创建可视化图表 ===\n")
        
        high, low, close = generate_sample_data(100)
        
        # 计算指标
        adx, di_plus, di_minus, ml_st, ml_direction, resonance_signal = adx_ml_resonance_indicator(
            high, low, close,
            adx_length=10,
            adx_threshold=20,
            ml_atr_len=10,
            ml_fact=3.0,
            training_data_period=50
        )
        
        # 创建图表
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        
        # 价格和ML SuperTrend
        ax1.plot(close.values, label='Close', color='black', linewidth=1)
        ax1.plot(ml_st, label='ML SuperTrend', color='blue', linewidth=1)
        
        # 标记买入卖出信号
        buy_signals = np.where(resonance_signal == 1)[0]
        sell_signals = np.where(resonance_signal == -1)[0]
        
        if len(buy_signals) > 0:
            ax1.scatter(buy_signals, close.iloc[buy_signals], color='green', 
                       marker='^', s=50, label='买入信号')
        
        if len(sell_signals) > 0:
            ax1.scatter(sell_signals, close.iloc[sell_signals], color='red', 
                       marker='v', s=50, label='卖出信号')
        
        ax1.set_title('价格与ML SuperTrend及交易信号')
        ax1.legend()
        ax1.grid(True)
        
        # ADX和DI
        ax2.plot(adx, label='ADX', color='purple', linewidth=2)
        ax2.plot(di_plus, label='DI+', color='green', linewidth=1)
        ax2.plot(di_minus, label='DI-', color='red', linewidth=1)
        ax2.axhline(y=20, color='gray', linestyle='--', alpha=0.5, label='ADX阈值')
        ax2.set_title('ADX和方向指标')
        ax2.legend()
        ax2.grid(True)
        
        # ML方向和共振信号
        ax3.plot(ml_direction, label='ML方向', color='blue', linewidth=2)
        ax3.bar(range(len(resonance_signal)), resonance_signal, 
                alpha=0.3, label='共振信号', color='orange')
        ax3.set_title('ML方向和共振信号')
        ax3.legend()
        ax3.grid(True)
        
        plt.tight_layout()
        
        # 保存图表
        output_file = 'adx_ml_resonance_example.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"图表已保存为: {output_file}")
        plt.close()
        
    except ImportError:
        print("matplotlib未安装，跳过可视化")
    except Exception as e:
        print(f"创建可视化时出错: {e}")


def run_unit_tests():
    """运行单元测试"""
    print("\n=== 运行单元测试 ===\n")
    
    try:
        import subprocess
        result = subprocess.run(['python3', 'test_indicators.py'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ 所有单元测试通过！")
            print(result.stdout)
        else:
            print("❌ 单元测试失败：")
            print(result.stderr)
    except Exception as e:
        print(f"运行测试时出错: {e}")
        print("请手动运行: python3 test_indicators.py")


def main():
    """主函数"""
    print("ADX ML共振指标使用示例")
    print("=" * 50)
    
    try:
        # 基本使用示例
        example_basic_usage()
        
        # 简化版指标示例
        example_simple_indicator()
        
        # 参数敏感性分析
        example_parameter_sensitivity()
        
        # 创建可视化（可选）
        create_visualization()
        
        # 运行单元测试
        run_unit_tests()
        
    except Exception as e:
        print(f"\n运行示例时出错: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n示例运行完成！")


if __name__ == "__main__":
    main() 