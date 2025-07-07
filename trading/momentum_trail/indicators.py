# -*- coding: utf-8 -*-
# @Author: lovelyzzc
# @Date: 2024-07-31

import pandas as pd
import numpy as np
import pandas_ta as ta

# ━━━━━━━━━━━━━━━━ 1. INDICATOR DEFINITIONS ━━━━━━━━━━━━━━━━

def double_smooth(src, length, smth_len):
    """Applies a double EMA smoothing to the source series."""
    first = ta.ema(src, length=length)
    if first is None:
        return None
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
            wma_upper = ta.wma(intermediate_upper_src.iloc[start_idx:i+1], length=wma_length)
            wma_lower = ta.wma(intermediate_lower_src.iloc[start_idx:i+1], length=wma_length)

            if wma_upper is not None and not wma_upper.empty:
                upper.iloc[i] = wma_upper.iloc[-1]
            else: # Not enough data for WMA, use the intermediate value
                upper.iloc[i] = intermediate_upper_src.iloc[i]

            if wma_lower is not None and not wma_lower.empty:
                lower.iloc[i] = wma_lower.iloc[-1]
            else: # Not enough data for WMA, use the intermediate value
                lower.iloc[i] = intermediate_lower_src.iloc[i]
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
    
    if double_pc is None or double_abs_pc is None:
        return np.full_like(close, np.nan)
    
    # Avoid division by zero
    divisor = double_abs_pc.where(double_abs_pc != 0, 1)
    mom = 100 * (double_pc / divisor)

    if isinstance(mom, pd.Series):
        mom.fillna(0, inplace=True) # Fill any potential NaNs in mom
    elif pd.isna(mom):
        mom = 0

    direction, _, _ = trail_indicator(mom, trail_len, trail_mult)
    return direction 