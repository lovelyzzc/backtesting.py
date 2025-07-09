# -*- coding: utf-8 -*-
"""
This file defines the Range Filter trading strategy based on the provided Pine Script.
The strategy buys when the price crosses over an upper band and sells when it crosses
below a lower band. The bands are calculated using SMA and ATR.
"""
from backtesting import Strategy
from backtesting.lib import crossover
import pandas as pd
import pandas_ta as ta
import numpy as np

class RangeFilterStrategy(Strategy):
    """
    Implements the Range Filter trading strategy.
    
    Entry Signal: Close price crosses above the Upper Band (SMA + ATR * Multiplier).
    Exit Signal:  Close price crosses below the Lower Band (SMA - ATR * Multiplier).
    """

    # --- Strategy Parameters ---
    # These will be optimized by the framework.
    # We use n_len for both SMA and ATR periods to match the Pine Script logic.
    n_len = 10 
    atr_multiplier = 2.0

    def init(self):
        """
        This method is called once at the beginning of the backtest.
        We use it to initialize our indicators.
        """
        # --- 安全检查 (关键修复) ---
        # 如果数据长度小于指标计算所需的回看周期，则不初始化指标，
        # 从而避免在数据不足时导致回测崩溃。
        if len(self.data.Close) < self.n_len:
            self._indicators_ready = False
            return
        
        self._indicators_ready = True
        
        # 将数据转换为pandas Series以供pandas_ta使用
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        
        # 1. 计算简单移动平均线 (SMA)
        self.ma = self.I(ta.sma, close, length=self.n_len)
        
        # 2. 计算平均真实波幅 (ATR)
        self.atr = self.I(ta.atr, high, low, close, length=self.n_len)
        
        # 3. 计算上下轨
        self.upper_band = self.ma + self.atr * self.atr_multiplier
        self.lower_band = self.ma - self.atr * self.atr_multiplier

    def next(self):
        """
        This method is called for each candlestick in the data.
        This is where the trading logic lives.
        """
        # 如果因为数据不足导致指标未被初始化，则直接跳过
        if not self._indicators_ready:
            return
        
        # --- 入场条件 ---
        # 如果当前无仓位，且收盘价上穿上轨，则买入
        if not self.position and crossover(self.data.Close, self.upper_band): # type: ignore
            self.buy()

        # --- 出场条件 ---
        # 如果当前持仓，且收盘价下穿下轨，则平仓
        elif self.position and crossover(self.lower_band, self.data.Close): # type: ignore
            self.position.close() 