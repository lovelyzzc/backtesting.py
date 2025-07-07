import pandas as pd
import numpy as np

def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    """Calculates Average True Range (ATR)"""
    tr = pd.DataFrame()
    tr['h_l'] = high - low
    tr['h_pc'] = (high - close.shift(1)).abs()
    tr['l_pc'] = (low - close.shift(1)).abs()
    true_range = tr[['h_l', 'h_pc', 'l_pc']].max(axis=1)
    return pd.Series(true_range.ewm(alpha=1/length, adjust=False).mean())

def pine_supertrend(close: pd.Series, high: pd.Series, low: pd.Series, factor: float, atr_values: pd.Series):
    """
    Python implementation of the SuperTrend indicator based on the provided PineScript.
    """
    src = (high + low) / 2
    upper_band = src + factor * atr_values
    lower_band = src - factor * atr_values

    supertrend_series = pd.Series(np.nan, index=close.index)
    direction_series = pd.Series(np.nan, index=close.index)

    # Initial values
    if len(close) > 0:
        direction_series.iloc[0] = 1

    for i in range(1, len(close)):
        # Update bands based on previous close
        if close.iloc[i-1] > upper_band.iloc[i-1]:
             lower_band.iloc[i] = max(lower_band.iloc[i], lower_band.iloc[i-1])

        if close.iloc[i-1] < lower_band.iloc[i-1]:
            upper_band.iloc[i] = min(upper_band.iloc[i], upper_band.iloc[i-1])

        # Determine direction
        if close.iloc[i] > upper_band.iloc[i]:
            direction_series.iloc[i] = 1
        elif close.iloc[i] < lower_band.iloc[i]:
            direction_series.iloc[i] = -1
        else:
            direction_series.iloc[i] = direction_series.iloc[i-1]
        
        # Determine supertrend value
        if direction_series.iloc[i] == 1:
            supertrend_series.iloc[i] = lower_band.iloc[i]
        else:
            supertrend_series.iloc[i] = upper_band.iloc[i]
            
    return supertrend_series, direction_series

def ml_adaptive_super_trend(high: pd.Series, low: pd.Series, close: pd.Series, atr_len=10, fact=3, training_data_period=100, highvol=0.75, midvol=0.5, lowvol=0.25):
    """
    Python implementation of the Machine Learning Adaptive SuperTrend indicator.
    """
    volatility = atr(high, low, close, atr_len)
    
    assigned_centroids = pd.Series(np.nan, index=close.index)

    for i in range(training_data_period -1, len(close)):
        vol_window = volatility.iloc[i - (training_data_period - 1) : i + 1]
        
        upper = vol_window.max()
        lower = vol_window.min()
        
        if upper == lower:
            assigned_centroids.iloc[i] = volatility.iloc[i]
            continue

        c1 = lower + (upper - lower) * highvol
        c2 = lower + (upper - lower) * midvol
        c3 = lower + (upper - lower) * lowvol

        # K-Means clustering
        max_iterations = 100
        for _ in range(max_iterations):
            c1_prev, c2_prev, c3_prev = c1, c2, c3
            
            cluster1, cluster2, cluster3 = [], [], []
            for vol in vol_window:
                dist1 = abs(vol - c1)
                dist2 = abs(vol - c2)
                dist3 = abs(vol - c3)
                
                if dist1 <= dist2 and dist1 <= dist3:
                    cluster1.append(vol)
                elif dist2 < dist1 and dist2 < dist3:
                    cluster2.append(vol)
                else:
                    cluster3.append(vol)
            
            c1 = np.mean(cluster1) if cluster1 else c1_prev
            c2 = np.mean(cluster2) if cluster2 else c2_prev
            c3 = np.mean(cluster3) if cluster3 else c3_prev
            
            if pd.isna(c1): c1 = c1_prev
            if pd.isna(c2): c2 = c2_prev
            if pd.isna(c3): c3 = c3_prev
                
            if c1 == c1_prev and c2 == c2_prev and c3 == c3_prev:
                break

        # Assign centroid to current volatility
        current_vol = volatility.iloc[i]
        centroids = sorted([c1, c2, c3]) # low, medium, high
        
        dist1 = abs(current_vol - centroids[0])
        dist2 = abs(current_vol - centroids[1])
        dist3 = abs(current_vol - centroids[2])
        
        distances = [dist1, dist2, dist3]
        
        assigned_centroids.iloc[i] = centroids[np.argmin(distances)]

    # Calculate SuperTrend with the adaptive ATR (assigned_centroids)
    src = (high + low) / 2
    upper_band = src + fact * assigned_centroids
    lower_band = src - fact * assigned_centroids

    st = pd.Series(np.nan, index=close.index)
    direction = pd.Series(np.nan, index=close.index)

    if len(close) > 0:
        direction.iloc[0] = 1

    for i in range(1, len(close)):
        if pd.isna(assigned_centroids.iloc[i]):
            st.iloc[i] = st.iloc[i-1]
            direction.iloc[i] = direction.iloc[i-1]
            upper_band.iloc[i] = upper_band.iloc[i-1]
            lower_band.iloc[i] = lower_band.iloc[i-1]
            continue

        prev_lower = lower_band.iloc[i-1]
        prev_upper = upper_band.iloc[i-1]
        
        # Update bands
        if not ((lower_band.iloc[i] > prev_lower) or (close.iloc[i-1] < prev_lower)):
            lower_band.iloc[i] = prev_lower
        
        if not ((upper_band.iloc[i] < prev_upper) or (close.iloc[i-1] > prev_upper)):
            upper_band.iloc[i] = prev_upper

        # Determine direction
        if pd.isna(st.iloc[i-1]):
             direction.iloc[i] = 1
        elif st.iloc[i-1] == prev_upper:
            direction.iloc[i] = -1 if close.iloc[i] > upper_band.iloc[i] else 1
        else: # prev st == prev_lower
            direction.iloc[i] = 1 if close.iloc[i] < lower_band.iloc[i] else -1
            
        # Determine supertrend value
        st.iloc[i] = lower_band.iloc[i] if direction.iloc[i] == -1 else upper_band.iloc[i]

    return st, direction 