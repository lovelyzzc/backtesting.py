import pandas as pd
import numpy as np

def _wilders_rma(series: pd.Series, length: int) -> pd.Series:
    """Wilder's Smoothing (Running Moving Average) - 更鲁棒的版本."""
    if len(series) < length:
        # 如果数据不够，返回简单移动平均
        return series.rolling(window=min(len(series), length)).mean()
    
    # 使用更稳定的EMA计算方式
    alpha = 1.0 / length
    result = series.ewm(alpha=alpha, adjust=False).mean()
    return result

def adx_indicator(high: pd.Series, low: pd.Series, close: pd.Series, length: int):
    """
    计算ADX, +DI, -DI指标 - 更鲁棒的版本
    处理各种边界情况和异常数据
    """
    # 数据验证
    if len(high) < length + 1:
        # 数据不足时，返回简单的默认值
        n = len(high)
        return np.full(n, 20.0), np.full(n, 20.0), np.full(n, 20.0)
    
    # 确保数据是数值类型并且没有无穷大值
    high = pd.Series(high).replace([np.inf, -np.inf], np.nan).ffill().bfill()
    low = pd.Series(low).replace([np.inf, -np.inf], np.nan).ffill().bfill()
    close = pd.Series(close).replace([np.inf, -np.inf], np.nan).ffill().bfill()
    
    # 基本数据验证
    if high.isna().all() or low.isna().all() or close.isna().all():
        n = len(high)
        return np.full(n, 20.0), np.full(n, 20.0), np.full(n, 20.0)
    
    # 计算前一天的价格，使用安全的方式处理第一个值
    close_prev = close.shift(1)
    high_prev = high.shift(1)
    low_prev = low.shift(1)
    
    # 用当前值填充第一个NaN (更保守的方法)
    close_prev.iloc[0] = close.iloc[0]
    high_prev.iloc[0] = high.iloc[0]
    low_prev.iloc[0] = low.iloc[0]

    # True Range计算 - 使用更稳定的方法
    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    # 确保TR不为零
    tr = tr.replace(0, 1e-10)

    # Directional Movement计算
    v1 = high - high_prev
    v2 = low_prev - low
    
    # 使用更安全的条件判断
    dm_plus = np.where((v1 > v2) & (v1 > 0), v1, 0)
    dm_minus = np.where((v2 > v1) & (v2 > 0), v2, 0)
    
    dm_plus_series = pd.Series(dm_plus, index=high.index)
    dm_minus_series = pd.Series(dm_minus, index=high.index)

    # 使用Wilder's smoothing计算ATR和DM
    try:
        atr = _wilders_rma(tr, length)
        dm_plus_smooth = _wilders_rma(dm_plus_series, length)
        dm_minus_smooth = _wilders_rma(dm_minus_series, length)
    except Exception:
        # 如果Wilder's smoothing失败，使用简单移动平均
        atr = tr.rolling(window=length).mean()
        dm_plus_smooth = dm_plus_series.rolling(window=length).mean()
        dm_minus_smooth = dm_minus_series.rolling(window=length).mean()
    
    # 确保ATR不为零
    atr = atr.replace(0, 1e-10).fillna(1e-10)
    
    # 计算DI
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr

    # ADX计算
    di_sum = di_plus + di_minus
    
    # 安全的除法运算
    dx = pd.Series(np.zeros(len(di_plus)), index=di_plus.index)
    valid_mask = (di_sum > 1e-10)
    dx[valid_mask] = 100 * np.abs(di_plus[valid_mask] - di_minus[valid_mask]) / di_sum[valid_mask]
    
    # ADX是DX的简单移动平均
    try:
        adx = dx.rolling(window=length).mean()
    except Exception:
        adx = pd.Series(np.full(len(dx), 20.0), index=dx.index)

    # 处理NaN值 - 使用更保守的方法
    # 用合理的默认值填充
    adx = adx.fillna(20.0)
    di_plus = di_plus.fillna(20.0)
    di_minus = di_minus.fillna(20.0)
    
    # 确保所有值都在合理范围内
    adx = np.clip(adx.values, 0, 100)
    di_plus = np.clip(di_plus.values, 0, 100)
    di_minus = np.clip(di_minus.values, 0, 100)

    # backtesting.py indicator functions should return tuple of numpy arrays
    return adx, di_plus, di_minus 