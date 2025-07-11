# -*- coding: utf-8 -*-
"""
🎯 完全修复的优化策略实现
================================================================================
确保与策略1的逻辑绝对一致，只优化性能，不改变任何交易逻辑
================================================================================
"""

import backtesting
import pandas as pd
import numpy as np
import pandas_ta as ta

class UptrendQuantifierStrategy(backtesting.Strategy):
    """
    完全修复的优化策略实现
    与策略1保持100%逻辑一致性，只优化性能
    """
    
    # 策略参数 - 与策略1完全一致
    len_short = 20
    len_mid = 50
    len_long = 200
    adx_len = 14
    adx_threshold = 25
    stop_loss_pct = 0.05  # 与策略1保持一致
    
    def init(self):
        """
        初始化方法 - 使用与策略1完全相同的逻辑
        """
        # 数据长度检查
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        data_len = len(close)

        required_len = max(self.len_short, self.len_mid, self.len_long, self.adx_len) + 1
        if data_len < required_len:
            self._indicators_ready = False
            return
        
        self._indicators_ready = True
        
        # 初始化入场价格跟踪 - 与策略1完全一致
        self.entry_price = None
        
        # 使用与策略1完全相同的指标计算方式
        close_series = pd.Series(close)
        self.ema_short = self.I(self._calculate_ema, close_series, self.len_short)
        self.ema_mid = self.I(self._calculate_ema, close_series, self.len_mid)  
        self.ema_long = self.I(self._calculate_ema, close_series, self.len_long)

        # 使用与策略1完全相同的DMI计算方式
        adx_data, dmp_data, dmn_data = self._calculate_dmi(high, low, close, self.adx_len)
        self.adx = self.I(lambda: adx_data)
        self.di_plus = self.I(lambda: dmp_data)
        self.di_minus = self.I(lambda: dmn_data)
        
        # 预计算信号 - 使用与策略1完全相同的逻辑
        self._precompute_signals()
        
        # 状态跟踪 - 与策略1完全一致
        self._bar_index = 0
        self._last_signal_bar = -1
        self._stop_loss_multiplier = 1 - self.stop_loss_pct

    def _calculate_ema(self, series: pd.Series, length: int):
        """EMA计算 - 与策略1完全一致"""
        ema_result = ta.ema(series, length=length)
        return ema_result.values if ema_result is not None else np.full(len(series), np.nan)

    def _calculate_dmi(self, high, low, close, length):
        """DMI计算 - 与策略1完全一致"""
        try:
            high_series = pd.Series(high) if not isinstance(high, pd.Series) else high
            low_series = pd.Series(low) if not isinstance(low, pd.Series) else low
            close_series = pd.Series(close) if not isinstance(close, pd.Series) else close
            
            dmi_df = ta.adx(high_series, low_series, close_series, length=length)
            
            if dmi_df is not None and not dmi_df.empty:
                adx_col = f'ADX_{length}'
                dmp_col = f'DMP_{length}'
                dmn_col = f'DMN_{length}'
                
                return (
                    dmi_df[adx_col].values if adx_col in dmi_df.columns else np.full(len(close), np.nan),
                    dmi_df[dmp_col].values if dmp_col in dmi_df.columns else np.full(len(close), np.nan),
                    dmi_df[dmn_col].values if dmn_col in dmi_df.columns else np.full(len(close), np.nan)
                )
            else:
                return (np.full(len(close), np.nan), np.full(len(close), np.nan), np.full(len(close), np.nan))
                
        except Exception as e:
            print(f"DMI计算错误: {e}")
            return (np.full(len(close), np.nan), np.full(len(close), np.nan), np.full(len(close), np.nan))

    def _precompute_signals(self):
        """
        预计算信号 - 使用与策略1完全相同的逻辑
        """
        data_len = len(self.data.Close)
        
        # 获取数据 - 与策略1完全一致
        close_values = np.array(self.data.Close)
        ema_short_values = np.array(self.ema_short)
        ema_mid_values = np.array(self.ema_mid)
        ema_long_values = np.array(self.ema_long)
        adx_values = np.array(self.adx)
        di_plus_values = np.array(self.di_plus)
        di_minus_values = np.array(self.di_minus)
        
        # 条件计算 - 与策略1完全相同的逻辑
        price_above_mid = close_values > ema_mid_values
        trend_direction_up = di_plus_values > di_minus_values
        trend_strength_ok = adx_values > self.adx_threshold
        ma_bullish_alignment = (ema_short_values > ema_mid_values) & (ema_mid_values > ema_long_values)
        
        # 有效数据检查 - 与策略1完全一致
        valid_data = (~np.isnan(ema_short_values) & ~np.isnan(ema_mid_values) & 
                     ~np.isnan(ema_long_values) & ~np.isnan(adx_values) & 
                     ~np.isnan(di_plus_values) & ~np.isnan(di_minus_values))
        
        # 综合条件 - 与策略1完全一致
        self._uptrend_conditions = (valid_data & price_above_mid & trend_direction_up & 
                                   trend_strength_ok & ma_bullish_alignment)
        
        # 入场信号计算 - 与策略1完全一致
        prev_uptrend = np.concatenate([[False], self._uptrend_conditions[:-1]])
        self._entry_signals = self._uptrend_conditions & ~prev_uptrend

    def next(self):
        """
        交易逻辑 - 与策略1完全一致
        """
        if not self._indicators_ready:
            return

        current_idx = self._bar_index
        current_close = self.data.Close[-1]
        
        # 止损检查 - 与策略1完全一致
        if self.position and self.entry_price is not None:
            stop_loss_price = self.entry_price * self._stop_loss_multiplier
            if current_close <= stop_loss_price:
                self.position.close()
                self.entry_price = None
                self._bar_index += 1
                return
        
        # 入场信号 - 与策略1完全一致
        if (not self.position and 
            current_idx < len(self._entry_signals) and 
            self._entry_signals[current_idx]):
            self.buy()
            self.entry_price = current_close
            self._last_signal_bar = current_idx

        # 出场信号 - 使用与策略1完全相同的crossover逻辑
        elif self.position:
            from backtesting.lib import crossover
            if crossover(self.ema_mid, self.data.Close):
                self.position.close()
                self.entry_price = None
            
        self._bar_index += 1 