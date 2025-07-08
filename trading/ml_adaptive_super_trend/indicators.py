import pandas as pd
import numpy as np

def atr(high, low, close, length: int):
    """Calculates Average True Range (ATR) using Wilder's method to match Pine Script ta.atr()"""
    # Convert backtesting._Array objects to pandas Series if needed
    if not isinstance(high, pd.Series):
        high = pd.Series(high)
    if not isinstance(low, pd.Series):
        low = pd.Series(low)
    if not isinstance(close, pd.Series):
        close = pd.Series(close)
    
    tr = pd.DataFrame()
    tr['h_l'] = high - low
    tr['h_pc'] = (high - close.shift(1)).abs()
    tr['l_pc'] = (low - close.shift(1)).abs()
    true_range = tr[['h_l', 'h_pc', 'l_pc']].max(axis=1)
    
    # Wilder's smoothing method (matches Pine Script ta.atr)
    atr_values = pd.Series(np.nan, index=true_range.index)
    
    # First value is simple average of first 'length' TR values
    for i in range(len(true_range)):
        if i < length - 1:
            continue
        elif i == length - 1:
            atr_values.iloc[i] = true_range.iloc[:i+1].mean()
        else:
            # Wilder's formula: ((n-1) * prev_atr + current_tr) / n
            atr_values.iloc[i] = ((length - 1) * atr_values.iloc[i-1] + true_range.iloc[i]) / length
    
    return atr_values

def pine_supertrend(close, high, low, factor: float, atr_values):
    """
    Python implementation of the SuperTrend indicator that exactly matches the provided PineScript logic.
    """
    # Convert backtesting._Array objects to pandas Series if needed
    if not isinstance(close, pd.Series):
        close = pd.Series(close)
    if not isinstance(high, pd.Series):
        high = pd.Series(high)
    if not isinstance(low, pd.Series):
        low = pd.Series(low)
    if not isinstance(atr_values, pd.Series):
        atr_values = pd.Series(atr_values)
    
    src = (high + low) / 2
    upper_band = src + factor * atr_values
    lower_band = src - factor * atr_values

    supertrend_series = pd.Series(np.nan, index=close.index)
    direction_series = pd.Series(np.nan, index=close.index)

    for i in range(len(close)):
        if i == 0:
            # Initial values - no previous data
            direction_series.iloc[i] = 1
            supertrend_series.iloc[i] = lower_band.iloc[i] if direction_series.iloc[i] == -1 else upper_band.iloc[i]
            continue
            
        prev_lower_band = lower_band.iloc[i-1] if not pd.isna(lower_band.iloc[i-1]) else 0
        prev_upper_band = upper_band.iloc[i-1] if not pd.isna(upper_band.iloc[i-1]) else 0
        prev_close = close.iloc[i-1]
        prev_supertrend = supertrend_series.iloc[i-1]
        
        # Update bands according to Pine Script logic
        # lowerBand := lowerBand > prevLowerBand or close[1] < prevLowerBand ? lowerBand : prevLowerBand
        if not pd.isna(lower_band.iloc[i]) and not pd.isna(prev_lower_band):
            if lower_band.iloc[i] > prev_lower_band or prev_close < prev_lower_band:
                pass  # keep current lower_band
            else:
                lower_band.iloc[i] = prev_lower_band
        
        # upperBand := upperBand < prevUpperBand or close[1] > prevUpperBand ? upperBand : prevUpperBand
        if not pd.isna(upper_band.iloc[i]) and not pd.isna(prev_upper_band):
            if upper_band.iloc[i] < prev_upper_band or prev_close > prev_upper_band:
                pass  # keep current upper_band
            else:
                upper_band.iloc[i] = prev_upper_band

        # Determine direction according to Pine Script logic
        if pd.isna(atr_values.iloc[i-1]) if i > 0 else True:
            direction_series.iloc[i] = 1
        elif prev_supertrend == prev_upper_band:
            direction_series.iloc[i] = -1 if close.iloc[i] > upper_band.iloc[i] else 1
        else:
            direction_series.iloc[i] = 1 if close.iloc[i] < lower_band.iloc[i] else -1
            
        # Determine supertrend value
        supertrend_series.iloc[i] = lower_band.iloc[i] if direction_series.iloc[i] == -1 else upper_band.iloc[i]
            
    return supertrend_series, direction_series

def ml_adaptive_super_trend(high, low, close, atr_len: int = 10, fact: float = 3, training_data_period: int = 100, highvol: float = 0.75, midvol: float = 0.5, lowvol: float = 0.25):
    """
    Python implementation of the Machine Learning Adaptive SuperTrend indicator that exactly matches Pine Script logic.
    """
    # Convert backtesting._Array objects to pandas Series if needed
    if not isinstance(high, pd.Series):
        high = pd.Series(high)
    if not isinstance(low, pd.Series):
        low = pd.Series(low)
    if not isinstance(close, pd.Series):
        close = pd.Series(close)
    
    # 边界情况检查
    if len(close) < atr_len:
        # 数据不足时，返回简单的SuperTrend
        simple_atr = pd.Series(0.01, index=close.index)  # 使用默认ATR值
        return pine_supertrend(close, high, low, fact, simple_atr)
    
    volatility = atr(high, low, close, atr_len)
    
    # 如果ATR全为NaN或0，使用默认值
    if volatility.isna().all() or (volatility == 0).all():
        default_atr = (high - low).mean() if not (high - low).isna().all() else 0.01
        volatility = pd.Series(default_atr, index=close.index)
    
    assigned_centroids = pd.Series(np.nan, index=close.index)

    for i in range(training_data_period - 1, len(close)):
        if pd.isna(volatility.iloc[i]) or volatility.iloc[i] <= 0:
            # 使用当前或前一个有效的volatility值
            if i > 0 and not pd.isna(assigned_centroids.iloc[i-1]):
                assigned_centroids.iloc[i] = assigned_centroids.iloc[i-1]
            else:
                assigned_centroids.iloc[i] = volatility.ffill().fillna(0.01).iloc[i]
            continue
            
        vol_window = volatility.iloc[i - (training_data_period - 1) : i + 1]
        
        # 移除NaN值
        vol_window = vol_window.dropna()
        if len(vol_window) == 0:
            assigned_centroids.iloc[i] = 0.01  # 默认值
            continue
            
        upper = vol_window.max()
        lower = vol_window.min()
        
        if upper == lower or upper - lower < 1e-10:  # 处理数值精度问题
            assigned_centroids.iloc[i] = volatility.iloc[i] if not pd.isna(volatility.iloc[i]) else 0.01
            continue

        # Initial centroids (matching Pine Script order: high, medium, low)
        high_vol_centroid = lower + (upper - lower) * highvol
        medium_vol_centroid = lower + (upper - lower) * midvol
        low_vol_centroid = lower + (upper - lower) * lowvol

        # K-Means clustering - track previous centroids for convergence
        hv_prev = [high_vol_centroid]  # Like amean array in Pine Script
        mv_prev = [medium_vol_centroid]  # Like bmean array in Pine Script  
        lv_prev = [low_vol_centroid]  # Like cmean array in Pine Script
        
        max_iterations = 100
        for iteration in range(max_iterations):
            # Get current centroids
            current_hv = hv_prev[0] if hv_prev else high_vol_centroid
            current_mv = mv_prev[0] if mv_prev else medium_vol_centroid  
            current_lv = lv_prev[0] if lv_prev else low_vol_centroid
            
            # Assign points to clusters
            hv_cluster, mv_cluster, lv_cluster = [], [], []
            for vol in vol_window:
                dist_hv = abs(vol - current_hv)
                dist_mv = abs(vol - current_mv)
                dist_lv = abs(vol - current_lv)
                
                # Match Pine Script logic exactly: three independent if statements
                if dist_hv < dist_mv and dist_hv < dist_lv:
                    hv_cluster.append(vol)
                
                if dist_mv < dist_hv and dist_mv < dist_lv:
                    mv_cluster.append(vol)
                
                if dist_lv < dist_hv and dist_lv < dist_mv:
                    lv_cluster.append(vol)
            
            # Calculate new centroids
            new_hv = np.mean(hv_cluster) if hv_cluster else current_hv
            new_mv = np.mean(mv_cluster) if mv_cluster else current_mv
            new_lv = np.mean(lv_cluster) if lv_cluster else current_lv
            
            # Check for NaN and replace if necessary
            if np.isnan(new_hv): new_hv = current_hv
            if np.isnan(new_mv): new_mv = current_mv
            if np.isnan(new_lv): new_lv = current_lv
                
            # Check convergence (like Pine Script while condition)
            converged = True
            if len(hv_prev) > 1 and abs(hv_prev[0] - new_hv) > 1e-10:  # 使用数值精度容差
                converged = False
            if len(mv_prev) > 1 and abs(mv_prev[0] - new_mv) > 1e-10:
                converged = False  
            if len(lv_prev) > 1 and abs(lv_prev[0] - new_lv) > 1e-10:
                converged = False
                
            # Update arrays (insert at beginning like Pine Script unshift)
            hv_prev.insert(0, new_hv)
            mv_prev.insert(0, new_mv)
            lv_prev.insert(0, new_lv)
            
            if converged and iteration > 0:
                break

        # Assign centroid to current volatility (matching Pine Script logic exactly)
        current_vol = volatility.iloc[i]
        hv_new = hv_prev[0]
        mv_new = mv_prev[0]  
        lv_new = lv_prev[0]
        
        # Calculate distances (matching Pine Script order)
        vdist_a = abs(current_vol - hv_new)  # high volatility distance
        vdist_b = abs(current_vol - mv_new)  # medium volatility distance
        vdist_c = abs(current_vol - lv_new)  # low volatility distance
        
        distances = [vdist_a, vdist_b, vdist_c]
        centroids = [hv_new, mv_new, lv_new]  # Keep original order, no sorting
        
        cluster_idx = np.argmin(distances)  # 0=high, 1=medium, 2=low
        assigned_centroids.iloc[i] = centroids[cluster_idx]

    # 确保assigned_centroids中有足够的非NaN值
    if assigned_centroids.isna().all():
        # 如果全为NaN，使用原始ATR
        assigned_centroids = volatility.fillna(0.01)
    else:
        # 向前填充NaN值
        assigned_centroids = assigned_centroids.ffill().bfill().fillna(0.01)

    # Calculate SuperTrend using the pine_supertrend function with adaptive ATR
    st, direction = pine_supertrend(close, high, low, fact, assigned_centroids)
    
    return st, direction 