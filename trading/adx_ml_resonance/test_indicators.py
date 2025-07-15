# -*- coding: utf-8 -*-
import unittest
import pandas as pd
import numpy as np
import os
import sys
from unittest.mock import patch

# Setup Python Path
script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from trading.adx_ml_resonance.indicators import adx_ml_resonance_indicator, simple_resonance_indicator


class TestADXMLResonanceIndicators(unittest.TestCase):
    """ADX ML共振指标单元测试"""
    
    def setUp(self):
        """设置测试数据"""
        np.random.seed(42)
        self.n_periods = 150
        
        # 生成模拟OHLC数据
        base_price = 100
        price_changes = np.random.normal(0, 1, self.n_periods)
        close_prices = [base_price]
        
        for change in price_changes[1:]:
            close_prices.append(close_prices[-1] + change)
            
        self.close = pd.Series(close_prices)
        self.high = self.close + np.random.uniform(0.5, 2.0, self.n_periods)
        self.low = self.close - np.random.uniform(0.5, 2.0, self.n_periods)
        
        # 确保价格关系正确
        self.high = np.maximum(self.high, self.close)
        self.low = np.minimum(self.low, self.close)
        
        self.high = pd.Series(self.high)
        self.low = pd.Series(self.low)
        
        self.default_params = {
            'adx_length': 10,
            'adx_threshold': 20,
            'ml_atr_len': 10,
            'ml_fact': 3.0,
            'training_data_period': 100,
            'highvol': 0.75,
            'midvol': 0.5,
            'lowvol': 0.25
        }

    def test_adx_ml_resonance_indicator_basic(self):
        """测试ADX ML共振指标基本功能"""
        result = adx_ml_resonance_indicator(
            self.high, self.low, self.close, **self.default_params
        )
        
        # 检查返回值数量和类型
        self.assertEqual(len(result), 6, "应该返回6个值")
        
        adx, di_plus, di_minus, ml_st, ml_direction, resonance_signal = result
        
        # 检查数组类型和长度
        for arr in result:
            self.assertIsInstance(arr, np.ndarray, "返回值应该是numpy数组")
            self.assertEqual(len(arr), len(self.close), "长度应该与输入数据一致")
        
        # 检查信号值范围
        unique_signals = np.unique(resonance_signal)
        valid_signals = {-1, 0, 1}
        self.assertTrue(
            set(unique_signals).issubset(valid_signals),
            f"共振信号应该只包含-1,0,1，实际: {unique_signals}"
        )

    def test_simple_resonance_indicator_basic(self):
        """测试简化版共振指标基本功能"""
        result = simple_resonance_indicator(
            self.high, self.low, self.close,
            adx_length=10, adx_threshold=20,
            ml_atr_len=10, ml_fact=3.0,
            training_data_period=100
        )
        
        self.assertEqual(len(result), 4, "应该返回4个值")
        
        buy_signal, sell_signal, adx, ml_direction = result
        
        # 检查信号类型
        self.assertEqual(buy_signal.dtype, bool, "买入信号应该是布尔类型")
        self.assertEqual(sell_signal.dtype, bool, "卖出信号应该是布尔类型")
        
        # 检查买入和卖出信号不会同时触发
        simultaneous_signals = buy_signal & sell_signal
        self.assertFalse(np.any(simultaneous_signals), "买入和卖出信号不应该同时触发")

    def test_input_data_conversion(self):
        """测试输入数据自动转换"""
        # 使用numpy数组输入
        result1 = adx_ml_resonance_indicator(
            np.array(self.high), np.array(self.low), np.array(self.close),
            **self.default_params
        )
        
        # 使用pandas Series输入
        result2 = adx_ml_resonance_indicator(
            self.high, self.low, self.close, **self.default_params
        )
        
        self.assertEqual(len(result1), 6, "numpy数组输入应该正常工作")
        self.assertEqual(len(result2), 6, "pandas Series输入应该正常工作")

    def test_edge_case_insufficient_data(self):
        """测试数据不足的边界情况"""
        short_high = pd.Series([101, 102, 103, 104, 105])
        short_low = pd.Series([99, 98, 97, 98, 99])
        short_close = pd.Series([100, 101, 102, 103, 104])
        
        result = adx_ml_resonance_indicator(
            short_high, short_low, short_close,
            adx_length=10, adx_threshold=20,
            ml_atr_len=10, ml_fact=3.0,
            training_data_period=100
        )
        
        self.assertEqual(len(result), 6, "数据不足时也应该返回6个值")
        
        for arr in result:
            self.assertEqual(len(arr), 5, "返回数组长度应该与输入一致")

    def test_edge_case_constant_prices(self):
        """测试价格不变的边界情况"""
        constant_price = 100
        constant_length = 50
        
        constant_high = pd.Series([constant_price] * constant_length)
        constant_low = pd.Series([constant_price] * constant_length)
        constant_close = pd.Series([constant_price] * constant_length)
        
        result = adx_ml_resonance_indicator(
            constant_high, constant_low, constant_close, **self.default_params
        )
        
        self.assertEqual(len(result), 6, "常数价格应该能正常处理")
        
        resonance_signal = result[5]
        # 常数价格时应该主要是无信号
        zero_signals = np.sum(resonance_signal == 0)
        total_signals = len(resonance_signal)
        self.assertGreater(zero_signals / total_signals, 0.8, "常数价格时应该主要是无信号")

    def test_nan_handling(self):
        """测试NaN值处理"""
        high_with_nan = pd.Series(self.high.copy())
        low_with_nan = pd.Series(self.low.copy())
        close_with_nan = pd.Series(self.close.copy())
        
        # 插入一些NaN值
        high_with_nan.iloc[10] = np.nan
        low_with_nan.iloc[15] = np.nan
        close_with_nan.iloc[20] = np.nan
        
        result = adx_ml_resonance_indicator(
            high_with_nan, low_with_nan, close_with_nan, **self.default_params
        )
        
        self.assertEqual(len(result), 6, "含NaN数据应该能正常处理")
        
        # 检查输出包含有限值
        for i, arr in enumerate(result):
            finite_count = np.isfinite(arr).sum()
            self.assertGreater(finite_count, 0, f"输出数组{i}应该包含有限值")

    def test_signal_consistency(self):
        """测试信号一致性"""
        result1 = adx_ml_resonance_indicator(
            self.high, self.low, self.close, **self.default_params
        )
        
        result2 = adx_ml_resonance_indicator(
            self.high, self.low, self.close, **self.default_params
        )
        
        # 比较所有输出数组
        for i, (arr1, arr2) in enumerate(zip(result1, result2)):
            np.testing.assert_array_equal(arr1, arr2, f"多次计算结果{i}应该一致")

    @patch('trading.adx_ml_resonance.indicators.adx_indicator')
    @patch('trading.adx_ml_resonance.indicators.ml_adaptive_super_trend')
    def test_signal_generation_logic(self, mock_ml_st, mock_adx):
        """测试信号生成逻辑"""
        n = 10
        
        # 设置mock返回值 - ADX强势
        mock_adx.return_value = (
            np.array([25] * n),  # ADX > 20
            np.array([20, 25, 30, 35, 30, 25, 20, 25, 30, 35]),  # DI+
            np.array([25, 20, 25, 20, 25, 30, 35, 30, 25, 20])   # DI-
        )
        
        # ML方向变化
        ml_direction = np.array([-1, -1, -1, 1, 1, 1, -1, -1, -1, -1])
        mock_ml_st.return_value = (np.array([100] * n), ml_direction)
        
        test_data = pd.Series([100] * n)
        
        result = adx_ml_resonance_indicator(
            test_data, test_data, test_data, **self.default_params
        )
        
        resonance_signal = result[5]
        
        # 验证Mock被调用
        mock_adx.assert_called_once()
        mock_ml_st.assert_called_once()
        
        # 检查信号逻辑
        self.assertEqual(len(resonance_signal), n, "信号长度应该正确")

    def test_different_thresholds(self):
        """测试不同ADX阈值的影响"""
        result_low = adx_ml_resonance_indicator(
            self.high, self.low, self.close,
            **{**self.default_params, 'adx_threshold': 10}
        )
        
        result_high = adx_ml_resonance_indicator(
            self.high, self.low, self.close,
            **{**self.default_params, 'adx_threshold': 40}
        )
        
        # 两种设置都应该能正常运行
        self.assertEqual(len(result_low), 6, "低阈值设置应该正常工作")
        self.assertEqual(len(result_high), 6, "高阈值设置应该正常工作")


def run_simple_performance_test():
    """简单性能测试"""
    import time
    
    print("\n=== 性能测试 ===")
    
    np.random.seed(42)
    n = 500
    close = pd.Series(100 + np.cumsum(np.random.randn(n) * 0.01))
    high = close + np.random.uniform(0.5, 2.0, n)
    low = close - np.random.uniform(0.5, 2.0, n)
    
    start_time = time.time()
    
    result = adx_ml_resonance_indicator(
        high, low, close,
        adx_length=14, adx_threshold=25,
        ml_atr_len=14, ml_fact=3.0,
        training_data_period=100
    )
    
    end_time = time.time()
    
    print(f"处理{n}个数据点耗时: {end_time - start_time:.3f}秒")
    print(f"结果包含{np.sum(result[5] != 0)}个非零信号")


if __name__ == '__main__':
    # 运行单元测试
    print("开始运行ADX ML共振指标单元测试...")
    unittest.main(verbosity=2, exit=False)
    
    # 运行性能测试
    run_simple_performance_test()
    
    print("\n测试完成！") 