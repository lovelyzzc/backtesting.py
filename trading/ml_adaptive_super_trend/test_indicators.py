# -*- coding: utf-8 -*-
import unittest
import pandas as pd
import numpy as np
import os
import sys
import traceback
from indicators import atr, pine_supertrend, ml_adaptive_super_trend

# --- Setup Python Path to allow imports from parent directory ---
# This ensures that we can import the 'trading' module
script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

class TestMlAdaptiveSuperTrend(unittest.TestCase):
    def test_indicator_run(self):
        """
        Tests if the ml_adaptive_super_trend indicator runs without errors on sample data
        and helps to debug the underlying issue.
        """
        # 1. Generate realistic sample data
        np.random.seed(42)
        n_periods = 250  # Using a longer period to ensure enough data for training
        price = 100 + np.random.randn(n_periods).cumsum()
        high_series = pd.Series(price + np.random.uniform(0, 1, size=n_periods))
        low_series = pd.Series(price - np.random.uniform(0, 1, size=n_periods))
        close_series = pd.Series(price)
        
        # 2. Define parameters as used in the backtest
        params = {
            'high': high_series,
            'low': low_series,
            'close': close_series,
            'atr_len': 10,
            'fact': 3.0,
            'training_data_period': 100,
            'highvol': 0.75,
            'midvol': 0.5,
            'lowvol': 0.25
        }
        
        # 3. Run the indicator and catch any exception
        try:
            st, direction = ml_adaptive_super_trend(**params)
            
            # 4. Basic assertions to check the validity of the output
            self.assertIsInstance(st, pd.Series, "SuperTrend output should be a pandas Series")
            self.assertIsInstance(direction, pd.Series, "Direction output should be a pandas Series")
            self.assertEqual(len(st), n_periods, "Output Series should have the same length as input")
            self.assertEqual(len(direction), n_periods, "Output Series should have the same length as input")
            self.assertFalse(st.isnull().all(), "SuperTrend series should not be all NaN")
            self.assertFalse(direction.isnull().all(), "Direction series should not be all NaN")
            print("\nUnit test passed: ml_adaptive_super_trend ran successfully.")

        except Exception as e:
            self.fail(f"ml_adaptive_super_trend raised an exception:\n\n{e}\n\nTraceback:\n{traceback.format_exc()}")

    def test_indicator_run_with_data(self):
        """
        Tests if the ml_adaptive_super_trend indicator runs without errors on sample data
        and helps to debug the underlying issue.
        """
        # 1. Generate realistic sample data
        np.random.seed(42)
        n_periods = 250  # Using a longer period to ensure enough data for training
        price = 100 + np.random.randn(n_periods).cumsum()
        high_series = pd.Series(price + np.random.uniform(0, 1, size=n_periods))
        low_series = pd.Series(price - np.random.uniform(0, 1, size=n_periods))
        close_series = pd.Series(price)
        
        # 2. Define parameters as used in the backtest
        params = {
            'high': high_series,
            'low': low_series,
            'close': close_series,
            'atr_len': 10,
            'fact': 3.0,
            'training_data_period': 100,
            'highvol': 0.75,
            'midvol': 0.5,
            'lowvol': 0.25
        }
        
        # 3. Run the indicator and catch any exception
        try:
            st, direction = ml_adaptive_super_trend(**params)
            
            # 4. Basic assertions to check the validity of the output
            self.assertIsInstance(st, pd.Series, "SuperTrend output should be a pandas Series")
            self.assertIsInstance(direction, pd.Series, "Direction output should be a pandas Series")
            self.assertEqual(len(st), n_periods, "Output Series should have the same length as input")
            self.assertEqual(len(direction), n_periods, "Output Series should have the same length as input")
            self.assertFalse(st.isnull().all(), "SuperTrend series should not be all NaN")
            self.assertFalse(direction.isnull().all(), "Direction series should not be all NaN")
            print("\nUnit test passed: ml_adaptive_super_trend ran successfully.")

        except Exception as e:
            self.fail(f"ml_adaptive_super_trend raised an exception:\n\n{e}\n\nTraceback:\n{traceback.format_exc()}")

if __name__ == '__main__':
    unittest.main()

# 创建一些模拟数据进行测试
np.random.seed(42)
n = 200

# 生成模拟的OHLC数据
close = pd.Series(100 + np.cumsum(np.random.randn(n) * 0.01), name='close')
high = close + np.random.uniform(0, 2, n)
low = close - np.random.uniform(0, 2, n)
open_price = close.shift(1).fillna(close.iloc[0]) + np.random.uniform(-1, 1, n)

print("测试数据创建完成")
print(f"数据长度: {len(close)}")
print(f"Close价格范围: {close.min():.2f} - {close.max():.2f}")

try:
    print("\n测试 ATR 函数...")
    atr_values = atr(high, low, close, 10)
    print(f"ATR计算成功，非NaN值数量: {atr_values.notna().sum()}")
    print(f"ATR范围: {atr_values.min():.4f} - {atr_values.max():.4f}")

    print("\n测试 pine_supertrend 函数...")
    st, direction = pine_supertrend(close, high, low, 3.0, atr_values)
    print(f"SuperTrend计算成功，非NaN值数量: {st.notna().sum()}")
    print(f"Direction非NaN值数量: {direction.notna().sum()}")

    print("\n测试 ml_adaptive_super_trend 函数...")
    ml_st, ml_direction = ml_adaptive_super_trend(high, low, close, 10, 3.0, 100, 0.75, 0.5, 0.25)
    print(f"ML SuperTrend计算成功，非NaN值数量: {ml_st.notna().sum()}")
    print(f"ML Direction非NaN值数量: {ml_direction.notna().sum()}")

    print("\n所有测试通过！")

except Exception as e:
    print(f"测试失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc() 