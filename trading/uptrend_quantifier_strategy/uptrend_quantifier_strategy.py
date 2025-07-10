# -*- coding: utf-8 -*-
"""
This file defines the Uptrend Quantifier trading strategy based on the provided Pine Script.
The strategy identifies strong uptrends using a combination of EMAs and ADX/DMI indicators.
性能优化版本：基于MultiBacktest思路，采用向量化计算和批量处理
"""
from backtesting import Strategy
from backtesting.lib import crossover
import pandas as pd
import pandas_ta as ta
import numpy as np
from typing import Optional

class UptrendQuantifierStrategy(Strategy):
    """
    Implements the Uptrend Quantifier trading strategy.
    
    Entry Signal: A strong uptrend is confirmed for the first time.
                  - EMAs in bullish alignment (short > mid > long).
                  - Price is above the mid-term EMA.
                  - ADX is above a threshold, indicating a strong trend.
                  - DI+ is above DI-, indicating an uptrend direction.
    Exit Signal:  Close price crosses below the mid-term EMA.
    Stop Loss:    Customizable percentage loss from entry price.
    
    性能优化(基于MultiBacktest思路):
    - 向量化全部信号计算
    - 预计算所有条件状态  
    - 批量处理替代逐bar计算
    - 共享内存减少数组访问
    - 使用numpy广播操作
    """

    # --- Strategy Parameters for Optimization ---
    len_short = 20
    len_mid = 50
    len_long = 200
    adx_len = 14
    adx_threshold = 25
    stop_loss_pct = 0.10  # 10% 止损

    def init(self):
        """
        Initialize the indicators for the strategy.
        基于MultiBacktest思路的向量化初始化
        """
        # 直接使用原始数据，避免转换开销
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        data_len = len(close)

        # Check for sufficient data length to avoid errors
        required_len = max(self.len_short, self.len_mid, self.len_long, self.adx_len) + 1
        if data_len < required_len:
            self._indicators_ready = False
            return
        
        self._indicators_ready = True
        
        # Initialize entry tracking for stop-loss logic
        self.entry_price = None
        
        # ===== 核心优化：向量化预计算所有指标 =====
        # print("🚀 启动向量化指标计算...")
        
        # 1. 批量计算所有EMA指标
        close_series = pd.Series(close)
        self.ema_short = self.I(self._calculate_ema_vectorized, close_series, self.len_short)
        self.ema_mid = self.I(self._calculate_ema_vectorized, close_series, self.len_mid)  
        self.ema_long = self.I(self._calculate_ema_vectorized, close_series, self.len_long)

        # 2. 一次性计算所有DMI组件（避免重复计算）
        adx_data, dmp_data, dmn_data = self._calculate_dmi_vectorized(high, low, close, self.adx_len)
        self.adx = self.I(lambda: adx_data)
        self.di_plus = self.I(lambda: dmp_data)
        self.di_minus = self.I(lambda: dmn_data)
        
        # ===== 核心优化：向量化预计算所有交易信号 =====
        # print("📊 向量化预计算交易信号...")
        self._precompute_signals_vectorized()
        
        # 交易状态追踪
        self._bar_index = 0
        self._last_signal_bar = -1
        
        # print("✅ 向量化初始化完成")

    def _calculate_ema_vectorized(self, series: pd.Series, length: int):
        """向量化EMA计算，使用pandas_ta的高效实现"""
        ema_result = ta.ema(series, length=length)
        return ema_result.values if ema_result is not None else np.full(len(series), np.nan)

    def _calculate_dmi_vectorized(self, high, low, close, length):
        """
        向量化DMI计算，一次性返回所有组件
        基于MultiBacktest的批量处理思路
        """
        try:
            # 使用pandas Series确保兼容性
            high_series = pd.Series(high) if not isinstance(high, pd.Series) else high
            low_series = pd.Series(low) if not isinstance(low, pd.Series) else low
            close_series = pd.Series(close) if not isinstance(close, pd.Series) else close
            
            # 一次性计算所有DMI组件
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

    def _precompute_signals_vectorized(self):
        """
        向量化预计算所有交易信号
        基于MultiBacktest的批量处理思路，大幅提升性能
        """
        data_len = len(self.data.Close)
        
        # ===== 批量获取所有数据，减少重复访问 =====
        close_values = np.array(self.data.Close)
        ema_short_values = np.array(self.ema_short)
        ema_mid_values = np.array(self.ema_mid)
        ema_long_values = np.array(self.ema_long)
        adx_values = np.array(self.adx)
        di_plus_values = np.array(self.di_plus)
        di_minus_values = np.array(self.di_minus)
        
        # ===== 向量化条件计算（关键优化）=====
        # 使用numpy广播和布尔索引，避免循环
        
        # 条件1：价格位置 (最便宜的计算)
        price_above_mid = close_values > ema_mid_values
        
        # 条件2：趋势方向
        trend_direction_up = di_plus_values > di_minus_values
        
        # 条件3：趋势强度
        trend_strength_ok = adx_values > self.adx_threshold
        
        # 条件4：MA排列 (最复杂的计算)
        ma_bullish_alignment = (ema_short_values > ema_mid_values) & (ema_mid_values > ema_long_values)
        
        # ===== 组合所有条件（向量化布尔运算）=====
        # 检查无效值
        valid_data = (~np.isnan(ema_short_values) & ~np.isnan(ema_mid_values) & 
                     ~np.isnan(ema_long_values) & ~np.isnan(adx_values) & 
                     ~np.isnan(di_plus_values) & ~np.isnan(di_minus_values))
        
        # 综合所有条件
        self._uptrend_conditions = (valid_data & price_above_mid & trend_direction_up & 
                                   trend_strength_ok & ma_bullish_alignment)
        
        # ===== 预计算入场信号（向量化）=====
        # 当前bar满足条件但前一个bar不满足 -> 入场信号
        prev_uptrend = np.concatenate([[False], self._uptrend_conditions[:-1]])
        self._entry_signals = self._uptrend_conditions & ~prev_uptrend
        
        # ===== 预计算止损价格数组（向量化）=====
        self._stop_loss_multiplier = 1 - self.stop_loss_pct
        
        # print(f"📈 向量化预计算完成: {np.sum(self._entry_signals)} 个潜在入场信号")

    def next(self):
        """
        优化的交易逻辑：基于预计算的向量化信号
        避免重复计算，大幅提升性能
        """
        if not self._indicators_ready:
            return

        current_idx = self._bar_index
        current_close = self.data.Close[-1]
        
        # ===== 快速止损检查（向量化预计算）=====
        if self.position and self.entry_price is not None:
            # 使用预计算的止损乘数，避免除法
            stop_loss_price = self.entry_price * self._stop_loss_multiplier
            if current_close <= stop_loss_price:
                self.position.close()
                self.entry_price = None
                self._bar_index += 1
                return
        
        # ===== 基于预计算信号的快速交易决策 =====
        # 入场信号：直接查询预计算的信号数组
        if (not self.position and 
            current_idx < len(self._entry_signals) and 
            self._entry_signals[current_idx]):
            self.buy()
            self.entry_price = current_close
            self._last_signal_bar = current_idx

        # 出场信号：价格跌破中期EMA
        elif self.position and crossover(self.ema_mid, self.data.Close): # type: ignore
            self.position.close()
            self.entry_price = None
            
        self._bar_index += 1

    @staticmethod
    def create_vectorized_backtest(data, **kwargs):
        """
        静态方法：创建向量化优化的回测实例
        基于MultiBacktest的批量处理思路
        """
        from backtesting import Backtest
        return Backtest(data, UptrendQuantifierStrategy, **kwargs)
    
    @classmethod
    def batch_optimize(cls, data_list, param_grid, **backtest_kwargs):
        """
        批量优化方法，模拟MultiBacktest的并行处理思路
        """
        from backtesting import Backtest
        from concurrent.futures import ProcessPoolExecutor
        import os
        
        def run_single_optimization(data):
            bt = Backtest(data, cls, **backtest_kwargs)
            return bt.optimize(**param_grid)
        
        max_workers = min(os.cpu_count() or 4, len(data_list))
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(run_single_optimization, data_list))
            
        return results 