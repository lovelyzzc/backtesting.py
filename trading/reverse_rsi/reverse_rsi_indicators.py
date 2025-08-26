
"""
reverse_rsi_indicators.py

Indicators for backtesting.py scanner based on "Reverse RSI Signals":
- Reverse-RSI price bands for target RSI levels (e.g., 70/50/30) computed from Wilder's RMA recursion
- Supertrend computed on the RSI=50 price (mid band)
- Regular (price/RSI) divergences using pivot highs/lows

Each function is written to be compatible with backtesting.py's Strategy.I(...),
i.e., it accepts numpy arrays and returns a numpy array of the same length.

Author: ChatGPT
License: MIT
"""

from __future__ import annotations
from typing import Tuple, Dict
import numpy as np


# ---------- Utilities ----------

def _sma(x: np.ndarray, length: int) -> np.ndarray:
    n = len(x)
    out = np.full(n, np.nan, dtype=float)
    if length <= 0:
        return out
    if n < length:
        return out
    csum = np.cumsum(np.nan_to_num(x, nan=0.0))
    # SMA for window [i-length+1, i]
    out[length-1:] = (csum[length-1:] - np.concatenate(([0.0], csum[:-length])) ) / length
    return out


def _ema(x: np.ndarray, length: int) -> np.ndarray:
    """Classic EMA with alpha = 2/(length+1). NaNs are propagated until the first non-NaN window accumulates."""
    n = len(x)
    out = np.full(n, np.nan, dtype=float)
    if length <= 1:
        return x.astype(float)
    alpha = 2.0 / (length + 1.0)
    # seed with SMA of first `length` non-nan values where possible
    valid = ~np.isnan(x)
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        return out
    # Find first index where we have at least `length` valid values before/at it
    # Simple approach: take first index >= length-1
    start = max(length - 1, int(idx[0]))
    # But ensure there are at least `length` valid in the prefix [0..start]
    # If not, advance until so
    while start < n and np.sum(valid[:start+1]) < length:
        start += 1
    if start >= n:
        return out
    # Seed: SMA of the last `length` values ending at `start`
    vals = x.copy()
    vals[~valid] = np.nan
    seed_window = vals[:start+1]
    # Take last `length` non-nan values from seed_window
    good = seed_window[~np.isnan(seed_window)]
    if good.size >= length:
        seed = np.mean(good[-length:])
    else:
        return out
    out[start] = seed
    # Iterate forward
    for i in range(start+1, n):
        xi = x[i]
        if np.isnan(xi):
            out[i] = out[i-1]
        else:
            out[i] = alpha * xi + (1 - alpha) * out[i-1]
    return out


def _rma(values: np.ndarray, length: int) -> np.ndarray:
    """Wilder's RMA: rma[i] = (rma[i-1]*(length-1) + values[i]) / length
    Seed with SMA of first `length` values.
    """
    n = len(values)
    out = np.full(n, np.nan, dtype=float)
    if length <= 0 or n == 0:
        return out
    if n < length:
        return out
    # Seed with SMA of first `length` values
    seed = np.nanmean(values[1:length+1]) if not np.isnan(values[1:length+1]).all() else np.nanmean(values[:length])
    out[length] = seed
    for i in range(length+1, n):
        prev = out[i-1]
        val = values[i]
        if np.isnan(prev) or np.isnan(val):
            out[i] = np.nan if np.isnan(prev) else prev  # maintain continuity if possible
        else:
            out[i] = (prev * (length - 1) + val) / length
    return out


def rsi(src: np.ndarray, length: int = 14) -> np.ndarray:
    """Standard RSI (Wilder) for reference & divergence checks."""
    src = np.asarray(src, dtype=float)
    n = len(src)
    if n == 0:
        return np.array([], dtype=float)
    chg = np.diff(src, prepend=src[0])
    gain = np.where(chg > 0, chg, 0.0)
    loss = np.where(chg < 0, -chg, 0.0)
    up = _rma(gain, length)
    dn = _rma(loss, length)
    rs = up / np.maximum(dn, 1e-10)
    out = 100.0 - (100.0 / (1.0 + rs))
    out[:length+1] = np.nan  # unreliable seed
    return out


# ---------- Reverse-RSI price for a target level ----------

def _reverse_rsi_price(level: float,
                       src: np.ndarray,
                       length: int,
                       upPrev: np.ndarray,
                       dnPrev: np.ndarray) -> np.ndarray:
    """Given target RSI level, previous RMA up/dn, and previous price, compute
    the price for current bar that would make RSI == level.
    This mirrors the algebraic inversion used in Pine.
    """
    n = len(src)
    out = np.full(n, np.nan, dtype=float)
    if n == 0:
        return out
    lv = min(max(level, 1e-6), 100.0 - 1e-6)
    RS = lv / (100.0 - lv)
    L = float(length)

    srcPrev = np.roll(src, 1)
    srcPrev[0] = np.nan

    A = (upPrev * (L - 1.0))
    B = (dnPrev * (L - 1.0))

    # Solve two branches for c (price change)
    c_pos = RS * B - A                    # assumes c >= 0 (up move)
    c_neg = B - (A / np.maximum(RS, 1e-12))  # assumes c < 0 (down move), c will be <= 0 ideally

    # Choose branch by sign of c_pos; if it's >= 0, use it; else use c_neg
    choose_pos = (c_pos >= 0.0)
    c = np.where(choose_pos, c_pos, c_neg)

    out = srcPrev + c
    # Invalidate where inputs are nan
    invalid = np.isnan(srcPrev) | np.isnan(A) | np.isnan(B)
    out[invalid] = np.nan
    return out


def _compute_up_dn_prev(src: np.ndarray, length: int) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Wilder RMA of gains/losses and then shift by 1 bar for 'prev'."""
    src = np.asarray(src, dtype=float)
    chg = np.diff(src, prepend=src[0])
    gain = np.where(chg > 0, chg, 0.0)
    loss = np.where(chg < 0, -chg, 0.0)
    up = _rma(gain, length)
    dn = _rma(loss, length)
    upPrev = np.roll(up, 1); upPrev[0] = np.nan
    dnPrev = np.roll(dn, 1); dnPrev[0] = np.nan
    return upPrev, dnPrev


def reverse_rsi_band(src: np.ndarray,
                     length: int = 14,
                     level: float = 70.0,
                     smooth: bool = True,
                     smooth_len: int = 14) -> np.ndarray:
    """Return price band corresponding to RSI == level on the *current* bar,
    computed using prev RMA state. Optionally EMA-smooth it.
    """
    src = np.asarray(src, dtype=float)
    upPrev, dnPrev = _compute_up_dn_prev(src, length)
    band = _reverse_rsi_price(level, src, length, upPrev, dnPrev)
    if smooth:
        band = _ema(band, max(2, int(smooth_len)))
    return band


def rsi_upper_band(src: np.ndarray, length: int = 14,
                   level_high: float = 70.0,
                   smooth: bool = True,
                   smooth_len: int = 14) -> np.ndarray:
    return reverse_rsi_band(src, length, level_high, smooth, smooth_len)


def rsi_lower_band(src: np.ndarray, length: int = 14,
                   level_low: float = 30.0,
                   smooth: bool = True,
                   smooth_len: int = 14) -> np.ndarray:
    return reverse_rsi_band(src, length, level_low, smooth, smooth_len)


def rsi_mid_band(src: np.ndarray, length: int = 14,
                 smooth: bool = True,
                 smooth_len: int = 14) -> np.ndarray:
    return reverse_rsi_band(src, length, 50.0, smooth, smooth_len)


def rsi_upper_outer(src: np.ndarray, length: int = 14, level_high: float = 70.0,
                    smooth: bool = True, smooth_len: int = 14, band_factor: float = 0.25) -> np.ndarray:
    ob = reverse_rsi_band(src, length, level_high, smooth, smooth_len)
    mid = reverse_rsi_band(src, length, 50.0, smooth, smooth_len)
    return mid + (1.0 + band_factor) * (ob - mid)


def rsi_lower_outer(src: np.ndarray, length: int = 14, level_low: float = 30.0,
                    smooth: bool = True, smooth_len: int = 14, band_factor: float = 0.25) -> np.ndarray:
    os = reverse_rsi_band(src, length, level_low, smooth, smooth_len)
    mid = reverse_rsi_band(src, length, 50.0, smooth, smooth_len)
    return mid + (1.0 + band_factor) * (os - mid)


# ---------- Cross/Enter-Band signals ----------

def entered_upper_band(src: np.ndarray, length: int = 14, level_high: float = 70.0,
                       smooth: bool = True, smooth_len: int = 14) -> np.ndarray:
    """Boolean (0/1) series: close crosses above the upper band."""
    src = np.asarray(src, dtype=float)
    ob = rsi_upper_band(src, length, level_high, smooth, smooth_len)
    prev = np.roll(src, 1); prev[0] = np.nan
    sig = (src > ob) & (prev <= np.roll(ob, 1))
    out = np.where(sig, 1.0, 0.0)
    out[np.isnan(ob) | np.isnan(prev)] = 0.0
    return out


def entered_lower_band(src: np.ndarray, length: int = 14, level_low: float = 30.0,
                       smooth: bool = True, smooth_len: int = 14) -> np.ndarray:
    """Boolean (0/1) series: close crosses below the lower band."""
    src = np.asarray(src, dtype=float)
    osb = rsi_lower_band(src, length, level_low, smooth, smooth_len)
    prev = np.roll(src, 1); prev[0] = np.nan
    sig = (src < osb) & (prev >= np.roll(osb, 1))
    out = np.where(sig, 1.0, 0.0)
    out[np.isnan(osb) | np.isnan(prev)] = 0.0
    return out


# ---------- Supertrend on mid band ----------

def supertrend_on_mid(src: np.ndarray, length: int = 14,
                      st_atr_len: int = 10, st_factor: float = 2.4,
                      smooth: bool = True, smooth_len: int = 14) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Supertrend on the RSI=50 price (mid band).
    Returns (supertrend_value, direction), with direction: +1 uptrend, -1 downtrend.
    """
    src = np.asarray(src, dtype=float)
    mid = rsi_mid_band(src, length, smooth, smooth_len)
    n = len(mid)
    st = np.full(n, np.nan, dtype=float)
    direction = np.full(n, 0.0, dtype=float)

    # Custom ATR using mid as a 'close-only' series
    prev_mid = np.roll(mid, 1); prev_mid[0] = np.nan
    tr = np.abs(mid - prev_mid)
    atr = _rma(tr, st_atr_len)

    basic_upper = mid + st_factor * atr
    basic_lower = mid - st_factor * atr

    final_upper = np.full(n, np.nan, dtype=float)
    final_lower = np.full(n, np.nan, dtype=float)

    for i in range(n):
        if i == 0 or np.isnan(basic_upper[i]) or np.isnan(basic_lower[i]):
            final_upper[i] = np.nan
            final_lower[i] = np.nan
            st[i] = np.nan
            direction[i] = 0.0
            continue

        if np.isnan(final_upper[i-1]):
            final_upper[i] = basic_upper[i]
        else:
            final_upper[i] = basic_upper[i] if (basic_upper[i] < final_upper[i-1] or mid[i-1] > final_upper[i-1]) else final_upper[i-1]

        if np.isnan(final_lower[i-1]):
            final_lower[i] = basic_lower[i]
        else:
            final_lower[i] = basic_lower[i] if (basic_lower[i] > final_lower[i-1] or mid[i-1] < final_lower[i-1]) else final_lower[i-1]

        if i == 1 or np.isnan(st[i-1]):
            st[i] = final_lower[i] if mid[i] >= final_lower[i] else final_upper[i]
        else:
            if st[i-1] == final_upper[i-1]:
                st[i] = final_upper[i] if mid[i] <= final_upper[i] else final_lower[i]
            else:  # st[i-1] == final_lower[i-1]
                st[i] = final_lower[i] if mid[i] >= final_lower[i] else final_upper[i]

        direction[i] = 1.0 if st[i] == final_lower[i] else -1.0

    return st, direction


def trend_shift_bullish(src: np.ndarray, length: int = 14,
                        st_atr_len: int = 10, st_factor: float = 2.4,
                        smooth: bool = True, smooth_len: int = 14) -> np.ndarray:
    """Boolean (0/1): Supertrend direction crosses from -1 to +1."""
    _, d = supertrend_on_mid(src, length, st_atr_len, st_factor, smooth, smooth_len)
    prev = np.roll(d, 1); prev[0] = np.nan
    out = (prev == -1.0) & (d == 1.0)
    out = np.where(out, 1.0, 0.0)
    out[np.isnan(prev)] = 0.0
    return out


def trend_shift_bearish(src: np.ndarray, length: int = 14,
                        st_atr_len: int = 10, st_factor: float = 2.4,
                        smooth: bool = True, smooth_len: int = 14) -> np.ndarray:
    """Boolean (0/1): Supertrend direction crosses from +1 to -1."""
    _, d = supertrend_on_mid(src, length, st_atr_len, st_factor, smooth, smooth_len)
    prev = np.roll(d, 1); prev[0] = np.nan
    out = (prev == 1.0) & (d == -1.0)
    out = np.where(out, 1.0, 0.0)
    out[np.isnan(prev)] = 0.0
    return out


# ---------- Divergence detection ----------

def _pivots(series: np.ndarray, lbL: int, lbR: int, mode: str) -> np.ndarray:
    """Return boolean array where series[i] is a pivot high/low with lookbacks lbL/lbR."""
    n = len(series)
    out = np.zeros(n, dtype=bool)
    for i in range(lbL, n-lbR):
        window = series[i-lbL:i+lbR+1]
        if np.any(np.isnan(window)):
            continue
        if mode == "low":
            if series[i] == np.min(window):
                out[i] = True
        else:
            if series[i] == np.max(window):
                out[i] = True
    return out


def regular_bullish_divergence(src: np.ndarray, length: int = 14,
                               lbL: int = 3, lbR: int = 1,
                               min_sep: int = 5, max_sep: int = 60) -> np.ndarray:
    """Boolean (0/1): Price makes lower low, RSI makes higher low between consecutive pivots."""
    src = np.asarray(src, dtype=float)
    r = rsi(src, length)
    piv_p = _pivots(src, lbL, lbR, mode="low")
    piv_r = _pivots(r,   lbL, lbR, mode="low")

    idx_p = np.flatnonzero(piv_p)
    idx_r = np.flatnonzero(piv_r)

    out = np.zeros(len(src), dtype=float)
    if idx_p.size < 2 or idx_r.size < 2:
        return out

    # Pair nearest previous pivot for simplicity
    last_p = None; last_r = None
    for i in range(len(src)):
        if piv_p[i]:
            if last_p is None:
                last_p = i
            else:
                # find nearest prior rsi pivot <= i
                cand = idx_r[idx_r <= i]
                if cand.size > 0:
                    curr_r = cand[-1]
                    # find previous rsi pivot before curr_r
                    prev_r_candidates = idx_r[idx_r < curr_r]
                    if prev_r_candidates.size > 0:
                        prev_r = prev_r_candidates[-1]
                        sep = i - last_p
                        if min_sep <= sep <= max_sep:
                            if src[i] < src[last_p] and r[curr_r] > r[prev_r]:
                                out[i] = 1.0
                last_p = i
    return out


def regular_bearish_divergence(src: np.ndarray, length: int = 14,
                               lbL: int = 3, lbR: int = 1,
                               min_sep: int = 5, max_sep: int = 60) -> np.ndarray:
    """Boolean (0/1): Price makes higher high, RSI makes lower high between consecutive pivots."""
    src = np.asarray(src, dtype=float)
    r = rsi(src, length)
    piv_p = _pivots(src, lbL, lbR, mode="high")
    piv_r = _pivots(r,   lbL, lbR, mode="high")

    idx_p = np.flatnonzero(piv_p)
    idx_r = np.flatnonzero(piv_r)

    out = np.zeros(len(src), dtype=float)
    if idx_p.size < 2 or idx_r.size < 2:
        return out

    last_p = None
    for i in range(len(src)):
        if piv_p[i]:
            if last_p is None:
                last_p = i
            else:
                cand = idx_r[idx_r <= i]
                if cand.size > 0:
                    curr_r = cand[-1]
                    prev_r_candidates = idx_r[idx_r < curr_r]
                    if prev_r_candidates.size > 0:
                        prev_r = prev_r_candidates[-1]
                        sep = i - last_p
                        if min_sep <= sep <= max_sep:
                            if src[i] > src[last_p] and r[curr_r] < r[prev_r]:
                                out[i] = 1.0
                last_p = i
    return out


# ---------- Convenience: one-call scanner pack ----------

def reverse_rsi_scanner(src: np.ndarray,
                        length: int = 14,
                        st_atr_len: int = 10, st_factor: float = 2.4,
                        level_high: float = 70.0, level_low: float = 30.0,
                        smooth: bool = True, smooth_len: int = 14) -> Dict[str, np.ndarray]:
    """Compute all bands and boolean scan signals in one shot.
    Returns a dict of numpy arrays keyed by:
      - 'ob', 'os', 'mid', 'upper_outer', 'lower_outer'
      - 'entered_upper', 'entered_lower'
      - 'st', 'st_dir', 'trend_shift_bull', 'trend_shift_bear'
      - 'bull_div', 'bear_div'
    """
    src = np.asarray(src, dtype=float)
    ob = rsi_upper_band(src, length, level_high, smooth, smooth_len)
    osb = rsi_lower_band(src, length, level_low,  smooth, smooth_len)
    mid = rsi_mid_band(src, length, smooth, smooth_len)
    upper_outer = rsi_upper_outer(src, length, level_high, smooth, smooth_len)
    lower_outer = rsi_lower_outer(src, length, level_low,  smooth, smooth_len)

    ent_up = entered_upper_band(src, length, level_high, smooth, smooth_len)
    ent_lo = entered_lower_band(src, length, level_low,  smooth, smooth_len)

    st, d = supertrend_on_mid(src, length, st_atr_len, st_factor, smooth, smooth_len)
    ts_bull = np.where((np.roll(d,1)==-1.0) & (d==1.0), 1.0, 0.0); ts_bull[0] = 0.0
    ts_bear = np.where((np.roll(d,1)== 1.0) & (d==-1.0), 1.0, 0.0); ts_bear[0] = 0.0

    bull_div = regular_bullish_divergence(src, length, lbL=3, lbR=1, min_sep=5, max_sep=60)
    bear_div = regular_bearish_divergence(src, length, lbL=3, lbR=1, min_sep=5, max_sep=60)

    return {
        "ob": ob, "os": osb, "mid": mid,
        "upper_outer": upper_outer, "lower_outer": lower_outer,
        "entered_upper": ent_up, "entered_lower": ent_lo,
        "st": st, "st_dir": d,
        "trend_shift_bull": ts_bull, "trend_shift_bear": ts_bear,
        "bull_div": bull_div, "bear_div": bear_div,
    }


# ---------- backtesting.py usage helpers ----------

def example_usage_with_backtesting():
    """
    Example (pseudo-usage) with backtesting.py:

    from backtesting import Backtest, Strategy
    import pandas as pd
    import numpy as np
    from reverse_rsi_indicators import (rsi_upper_band, rsi_lower_band, rsi_mid_band,
                                        entered_upper_band, entered_lower_band,
                                        supertrend_on_mid, trend_shift_bullish, trend_shift_bearish)

    class Strat(Strategy):
        def init(self):
            close = self.data.Close
            self.ob = self.I(rsi_upper_band, close, 14, 70.0, True, 14, plot=False)
            self.os = self.I(rsi_lower_band, close, 14, 30.0, True, 14, plot=False)
            self.mid = self.I(rsi_mid_band,   close, 14, True, 14, plot=False)

            # Signals (boolean 0/1 arrays)
            self.ent_up = self.I(entered_upper_band, close, 14, 70.0, True, 14, plot=False)
            self.ent_lo = self.I(entered_lower_band, close, 14, 30.0, True, 14, plot=False)

            st, d = supertrend_on_mid(close, 14, 10, 2.4, True, 14)  # two arrays
            self.st   = self.I(lambda x: st, close, plot=False)
            self.st_d = self.I(lambda x: d,  close, plot=False)

            self.ts_bull = self.I(trend_shift_bullish, close, 14, 10, 2.4, True, 14, plot=False)
            self.ts_bear = self.I(trend_shift_bearish, close, 14, 10, 2.4, True, 14, plot=False)

        def next(self):
            # Scanner-like rules (no trading), or define entries/exits
            i = len(self.data) - 1
            if self.ts_bull[-1] == 1 and self.ent_lo[-1] == 1:
                pass  # do something or record

    """
    pass
