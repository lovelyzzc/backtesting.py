# -*- coding: utf-8 -*-
# @Author: lovelyzzc
# @Date: 2024-07-31
#
# backtesting.py momentum_trail_strategy
#
# Before running, please ensure you have the required libraries installed:
# pip install backtesting pandas pandas-ta tqdm
#
# This script is a Python implementation of the "Momentum Trail Strategy"
# from the provided Pine Script. It iterates through all stock data files
# in the 'stock_data_cleaned' directory and runs a backtest for each.

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='pandas_ta')

import os
import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas_ta as ta
from tqdm import tqdm

# ━━━━━━━━━━━━━━━━ 1. HELPER FUNCTIONS (from Pine Script) ━━━━━━━━━━━━━━━━

def double_smooth(src, length, smth_len):
    """Applies a double EMA smoothing to the source series."""
    first = ta.ema(src, length=length)
    return ta.ema(first, length=smth_len)

def trail_indicator(src, length, mult):
    """
    Calculates the trailing stop bands and direction based on the Pine Script logic.
    This version fixes the logic by correctly applying the WMA smoothing *before*
    the direction is calculated, which is faithful to the original Pine Script.
    """
    basis = ta.wma(src, length=int(length / 2))
    vola = ta.hma(abs(src.diff()), length=length)

    upper_band = basis + vola * mult
    lower_band = basis - vola * mult

    # State variables to hold the final smoothed values, equivalent to Pine's `var`
    upper = pd.Series(np.nan, index=src.index)
    lower = pd.Series(np.nan, index=src.index)
    direction = pd.Series(1, index=src.index)

    # Intermediate series that are the source for the WMA, built iteratively
    intermediate_upper_src = pd.Series(np.nan, index=src.index)
    intermediate_lower_src = pd.Series(np.nan, index=src.index)

    wma_length = length * 3

    for i in range(len(src)):
        if i == 0:
            intermediate_upper_src.iloc[i] = upper_band.iloc[i] if not pd.isna(upper_band.iloc[i]) else 0
            intermediate_lower_src.iloc[i] = lower_band.iloc[i] if not pd.isna(lower_band.iloc[i]) else 0
        else:
            # Determine the source for the WMA based on the *previous* smoothed value
            last_upper = upper.iloc[i-1]
            last_lower = lower.iloc[i-1]

            if upper_band.iloc[i] < last_upper or src.iloc[i-1] > last_upper:
                intermediate_upper_src.iloc[i] = upper_band.iloc[i]
            else:
                intermediate_upper_src.iloc[i] = last_upper
            
            if lower_band.iloc[i] > last_lower or src.iloc[i-1] < last_lower:
                intermediate_lower_src.iloc[i] = lower_band.iloc[i]
            else:
                intermediate_lower_src.iloc[i] = last_lower

        # Calculate the WMA for the current step.
        # pandas_ta.wma is not iterative, so we apply it to the series up to the current point
        # and take the last value. The original implementation was slow, causing a hang.
        # OPTIMIZATION: Pass only the necessary window of data to WMA instead of the
        # expanding series. This changes complexity from O(N^2) to O(N).
        if i >= wma_length - 1:
            start_idx = i + 1 - wma_length
            upper.iloc[i] = ta.wma(intermediate_upper_src.iloc[start_idx:i+1], length=wma_length).iloc[-1]
            lower.iloc[i] = ta.wma(intermediate_lower_src.iloc[start_idx:i+1], length=wma_length).iloc[-1]
        else: # Not enough data for WMA, use the intermediate value
             upper.iloc[i] = intermediate_upper_src.iloc[i]
             lower.iloc[i] = intermediate_lower_src.iloc[i]

        # Now, calculate direction using the *current* smoothed values
        if i > 0:
            last_dir = direction.iloc[i-1]
            if last_dir == -1 and src.iloc[i] > upper.iloc[i]:
                direction.iloc[i] = 1
            elif last_dir == 1 and src.iloc[i] < lower.iloc[i]:
                direction.iloc[i] = -1
            else:
                direction.iloc[i] = last_dir

    # The returned 'direction' is now correctly calculated.
    # The smoothed upper/lower bands are also returned for potential plotting/analysis.
    return direction, upper, lower

def momentum_indicator(close, osc_len, smth_len, trail_len, trail_mult):
    """Calculates the final momentum and direction signal."""
    # backtesting.py passes numpy arrays for performance, but pandas-ta expects Series.
    close_series = pd.Series(close)

    price = ta.hma(close_series, length=int(osc_len / 3))
    
    if price is None:
        # This can happen if the input series is too short for the indicator.
        # Return an array of NaNs which backtesting.py can handle.
        return np.full_like(close, np.nan)

    pc = price.diff()
    
    double_pc = double_smooth(pc, osc_len, smth_len)
    double_abs_pc = double_smooth(abs(pc), osc_len, smth_len)
    
    # Avoid division by zero
    mom = 100 * (double_pc / double_abs_pc.where(double_abs_pc != 0, 1))
    mom.fillna(0, inplace=True) # Fill any potential NaNs in mom
    
    direction, _, _ = trail_indicator(mom, trail_len, trail_mult)
    return direction

# ━━━━━━━━━━━━━━━━ 2. STRATEGY DEFINITION ━━━━━━━━━━━━━━━━

class MomentumTrailStrategy(Strategy):
    # Strategy parameters
    osc_len = 21
    trail_mult = 12.0
    smth_len = 21
    trail_len = 5

    def init(self):
        """
        Initialize the strategy by pre-calculating the indicators.
        """
        self.direction = self.I(
            momentum_indicator,
            self.data.Close,
            self.osc_len,
            self.smth_len,
            self.trail_len,
            self.trail_mult
        )

    def next(self):
        """
        Define the trading logic for each bar.
        This is a stop-and-reverse strategy.
        """
        # A long signal is generated when the direction flips from -1 to 1.
        if self.direction[-2] == -1 and self.direction[-1] == 1:
            if self.position.is_short:
                self.position.close()
            self.buy()

        # A short signal is generated when the direction flips from 1 to -1.
        elif self.direction[-2] == 1 and self.direction[-1] == -1:
            if self.position.is_long:
                self.position.close()
            self.sell()


# ━━━━━━━━━━━━━━━━ 3. BACKTEST EXECUTION ━━━━━━━━━━━━━━━━

def run_backtest_for_file(filepath):
    """Loads a single CSV and runs the backtest."""
    try:
        data = pd.read_csv(filepath, index_col='Date', parse_dates=True)
    except Exception as e:
        print(f"Could not read {filepath}: {e}")
        return None

    # --- Data Cleaning and Preparation ---
    # Ensure correct column names for backtesting.py: Open, High, Low, Close, Volume
    # The library is case-sensitive.
    data = data.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    })

    # Ensure required columns are present
    required_cols = ['Open', 'High', 'Low', 'Close']
    if not all(col in data.columns for col in required_cols):
        #print(f"Skipping {filepath}: Missing one of the required columns (Open, High, Low, Close).")
        return None
        
    # Drop rows with missing values that might affect calculations
    data.dropna(subset=required_cols, inplace=True)
    
    if len(data) < 50: # Need enough data for indicator calculation
        #print(f"Skipping {filepath}: Not enough data rows.")
        return None

    bt = Backtest(data, MomentumTrailStrategy,
                  cash=10000,
                  commission=.00075, # 0.075%
                  exclusive_orders=True) # Stop-and-reverse behavior

    stats = bt.run()
    return stats


if __name__ == "__main__":
    # Construct the path to the data directory relative to the script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'stock_data_cleaned')

    if not os.path.isdir(data_dir):
        print(f"错误: 数据目录 '{data_dir}' 未找到.")
        exit()

    # Get all CSV files from the directory
    all_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

    if not all_files:
        print(f"在 '{data_dir}' 中未找到CSV文件.")
        exit()

    print(f"--- 开始对 {len(all_files)} 支股票进行回测 ---")

    all_stats = []

    # Use tqdm for a progress bar
    for filename in tqdm(all_files, desc="回测进度"):
        filepath = os.path.join(data_dir, filename)
        stats = run_backtest_for_file(filepath)

        if stats is not None:
            # Add a stock identifier to the results Series
            stats['Stock'] = filename.replace('.csv', '')
            all_stats.append(stats)

    if not all_stats:
        print("\n--- 所有回测均未产生有效结果 ---")
    else:
        # Combine all stats into a single DataFrame
        results_df = pd.DataFrame(all_stats)
        results_df.set_index('Stock', inplace=True)

        # Drop columns that are not suitable for aggregation (like complex objects)
        results_df.drop(columns=['_strategy', '_trades', '_equity_curve'], inplace=True, errors='ignore')

        print("\n\n--- 汇总回测结果 ---")
        with pd.option_context('display.max_rows', None, 'display.width', 1000):
            print(results_df)

        print("\n\n--- 汇总统计信息 ---")
        # The describe() method provides a good summary for numeric columns
        with pd.option_context('display.width', 1000):
            print(results_df.describe())

        # Optionally, save the results to a CSV file
        try:
            results_df.to_csv('momentum_trail_backtest_results.csv')
            print("\n结果已保存至 'momentum_trail_backtest_results.csv'")
        except Exception as e:
            print(f"\n保存结果失败: {e}")