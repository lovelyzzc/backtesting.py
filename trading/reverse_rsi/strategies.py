# -*- coding: utf-8 -*-
"""
Reverse RSI策略实现

基于Reverse RSI Signals指标的交易策略，包含：
- RSI反向价格水平突破信号
- SuperTrend趋势确认
- RSI发散信号
- 风险管理（止损止盈）
"""

from backtesting import Strategy
from backtesting.lib import crossover
import numpy as np

from trading.reverse_rsi.indicators import reverse_rsi_indicator


class ReverseRsiStrategy(Strategy):
    """
    Reverse RSI策略
    
    策略逻辑：
    1. 买入信号：
       - 价格突破超卖价格水平(os_price)向上 + SuperTrend转为看涨
       - 或者出现看涨发散信号 + SuperTrend看涨
    
    2. 卖出信号：
       - 价格突破超买价格水平(ob_price)向下 + SuperTrend转为看跌  
       - 或者出现看跌发散信号 + SuperTrend看跌
    
    3. 风险管理：
       - 止损：固定百分比止损
       - 止盈：固定百分比止盈
       - 最大持仓天数限制
    """
    
    # 策略参数
    rsi_length = 14           # RSI计算周期
    smooth_bands = True       # 是否平滑价格带
    st_factor = 2.4          # SuperTrend因子
    st_atr_len = 10          # SuperTrend ATR周期
    div_lookback = 3         # 发散检测回看周期
    
    # 信号权重参数
    use_price_breakout = True    # 是否使用价格突破信号
    use_divergence = True        # 是否使用发散信号
    use_trend_filter = True      # 是否使用趋势过滤
    
    # 风险管理参数
    stop_loss_pct = 0.05        # 5% 止损
    take_profit_pct = 0.12      # 12% 止盈
    max_holding_days = 10       # 最大持仓天数
    
    def init(self):
        """初始化策略指标"""
        # 计算Reverse RSI指标
        (self.ob_price, self.os_price, self.mid_price, 
         self.st_value, self.st_direction, 
         self.bull_divergence, self.bear_divergence) = self.I(
            reverse_rsi_indicator,
            self.data.High,
            self.data.Low, 
            self.data.Close,
            self.data.Volume,
            rsi_length=self.rsi_length,
            smooth_bands=self.smooth_bands,
            st_factor=self.st_factor,
            st_atr_len=self.st_atr_len,
            div_lookback=self.div_lookback
        )
        
        # 持仓计数器
        self.holding_days = 0
        self.entry_price = None
        
    def next(self):
        """策略主逻辑"""
        current_price = self.data.Close[-1]
        
        # 更新持仓天数
        if self.position:
            self.holding_days += 1
        else:
            self.holding_days = 0
            self.entry_price = None
        
        # 风险管理 - 强制平仓条件
        if self.position:
            # 最大持仓天数限制
            if self.holding_days >= self.max_holding_days:
                self.position.close()
                return
                
            # 止损止盈
            if self.entry_price is not None:
                if self.position.is_long:
                    # 多头止损止盈
                    stop_loss_price = self.entry_price * (1 - self.stop_loss_pct)
                    take_profit_price = self.entry_price * (1 + self.take_profit_pct)
                    
                    if current_price <= stop_loss_price or current_price >= take_profit_price:
                        self.position.close()
                        return
                        
                elif self.position.is_short:
                    # 空头止损止盈
                    stop_loss_price = self.entry_price * (1 + self.stop_loss_pct)
                    take_profit_price = self.entry_price * (1 - self.take_profit_pct)
                    
                    if current_price >= stop_loss_price or current_price <= take_profit_price:
                        self.position.close()
                        return
        
        # 检查数据有效性
        if (len(self.ob_price) < 2 or len(self.os_price) < 2 or 
            len(self.st_direction) < 2 or np.isnan(self.ob_price[-1]) or 
            np.isnan(self.os_price[-1]) or np.isnan(self.st_direction[-1])):
            return
        
        # 生成交易信号
        buy_signal = False
        sell_signal = False
        
        # 1. 价格突破信号
        if self.use_price_breakout:
            # 买入：价格向上突破超卖水平
            price_breakout_buy = (current_price > self.os_price[-1] and 
                                self.data.Close[-2] <= self.os_price[-2])
            
            # 卖出：价格向下突破超买水平  
            price_breakout_sell = (current_price < self.ob_price[-1] and
                                 self.data.Close[-2] >= self.ob_price[-2])
            
            if price_breakout_buy:
                buy_signal = True
            if price_breakout_sell:
                sell_signal = True
        
        # 2. 发散信号
        if self.use_divergence and len(self.bull_divergence) > 0 and len(self.bear_divergence) > 0:
            if self.bull_divergence[-1]:
                buy_signal = True
            if self.bear_divergence[-1]:
                sell_signal = True
        
        # 3. SuperTrend趋势过滤
        if self.use_trend_filter:
            # SuperTrend转为看涨 (从看跌-1转为看涨1)
            trend_shift_bullish = (self.st_direction[-1] == 1 and 
                                 len(self.st_direction) > 1 and 
                                 self.st_direction[-2] == -1)
            
            # SuperTrend转为看跌 (从看涨1转为看跌-1)
            trend_shift_bearish = (self.st_direction[-1] == -1 and 
                                 len(self.st_direction) > 1 and 
                                 self.st_direction[-2] == 1)
            
            # 只有在趋势确认的情况下才交易
            if buy_signal and self.st_direction[-1] != -1:
                buy_signal = False
            if sell_signal and self.st_direction[-1] != 1:
                sell_signal = False
                
            # 趋势转换信号
            if trend_shift_bullish:
                buy_signal = True
            if trend_shift_bearish:
                sell_signal = True
        
        # 执行交易
        if not self.position:
            if buy_signal:
                self.buy()
                self.entry_price = current_price
                self.holding_days = 0
            elif sell_signal:
                self.sell()
                self.entry_price = current_price
                self.holding_days = 0
        else:
            # 已有持仓时的反向信号处理
            if self.position.is_long and sell_signal:
                self.position.close()
                # 可选：立即开空仓
                # self.sell()
                # self.entry_price = current_price
                # self.holding_days = 0
            elif self.position.is_short and buy_signal:
                self.position.close()
                # 可选：立即开多仓
                # self.buy()
                # self.entry_price = current_price
                # self.holding_days = 0


class ReverseRsiLongOnlyStrategy(ReverseRsiStrategy):
    """
    Reverse RSI纯多头策略
    只做多，不做空
    """
    
    def next(self):
        """策略主逻辑 - 仅多头版本"""
        current_price = self.data.Close[-1]
        
        # 更新持仓天数
        if self.position:
            self.holding_days += 1
        else:
            self.holding_days = 0
            self.entry_price = None
        
        # 风险管理 - 强制平仓条件
        if self.position and self.position.is_long:
            # 最大持仓天数限制
            if self.holding_days >= self.max_holding_days:
                self.position.close()
                return
                
            # 止损止盈
            if self.entry_price is not None:
                stop_loss_price = self.entry_price * (1 - self.stop_loss_pct)
                take_profit_price = self.entry_price * (1 + self.take_profit_pct)
                
                if current_price <= stop_loss_price or current_price >= take_profit_price:
                    self.position.close()
                    return
        
        # 检查数据有效性
        if (len(self.ob_price) < 2 or len(self.os_price) < 2 or 
            len(self.st_direction) < 2 or np.isnan(self.ob_price[-1]) or 
            np.isnan(self.os_price[-1]) or np.isnan(self.st_direction[-1])):
            return
        
        # 生成交易信号
        buy_signal = False
        sell_signal = False
        
        # 1. 价格突破信号
        if self.use_price_breakout:
            # 买入：价格向上突破超卖水平
            price_breakout_buy = (current_price > self.os_price[-1] and 
                                self.data.Close[-2] <= self.os_price[-2])
            
            # 卖出：价格向下突破超买水平  
            price_breakout_sell = (current_price < self.ob_price[-1] and
                                 self.data.Close[-2] >= self.ob_price[-2])
            
            if price_breakout_buy:
                buy_signal = True
            if price_breakout_sell:
                sell_signal = True
        
        # 2. 发散信号
        if self.use_divergence and len(self.bull_divergence) > 0 and len(self.bear_divergence) > 0:
            if self.bull_divergence[-1]:
                buy_signal = True
            if self.bear_divergence[-1]:
                sell_signal = True  # 用作平仓信号
        
        # 3. SuperTrend趋势过滤
        if self.use_trend_filter:
            # SuperTrend转为看涨 (从看跌-1转为看涨1)
            trend_shift_bullish = (self.st_direction[-1] == 1 and 
                                 len(self.st_direction) > 1 and 
                                 self.st_direction[-2] == -1)
            
            # SuperTrend转为看跌 (从看涨1转为看跌-1)
            trend_shift_bearish = (self.st_direction[-1] == -1 and 
                                 len(self.st_direction) > 1 and 
                                 self.st_direction[-2] == 1)
            
            # 只有在趋势确认的情况下才买入
            if buy_signal and self.st_direction[-1] != 1:
                buy_signal = False
                
            # 趋势转换信号
            if trend_shift_bullish:
                buy_signal = True
            if trend_shift_bearish:
                sell_signal = True  # 用作平仓信号
        
        # 执行交易
        if not self.position:
            if buy_signal:
                self.buy()
                self.entry_price = current_price
                self.holding_days = 0
        else:
            # 已有多头持仓时的平仓信号
            if self.position.is_long and sell_signal:
                self.position.close() 