# -*- coding: utf-8 -*-
# @Author: lovelyzzc
# @Date: 2024-07-31

from backtesting import Strategy
from .indicators import momentum_indicator
import numpy as np

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
        
        # 预计算一些常用值，避免重复计算
        self._last_direction = 0
        self._position_entry_bar = None

    def next(self):
        """
        Define the trading logic for each bar.
        This is a long-only strategy with optimized performance.
        """
        current_direction = self.direction[-1]
        previous_direction = self.direction[-2] if len(self.direction) > 1 else 0

        # Exit conditions:
        # 1. 5% stop-loss (handled by `sl` in `buy()`).
        # 2. Take-profit after holding for 3 days.
        if self.position.is_long:
            # 优化：只计算一次当前bar索引
            current_bar = len(self.data) - 1
            
            # 使用预存的进入bar而不是查找最后一笔交易
            if self._position_entry_bar is not None:
                holding_days = current_bar - self._position_entry_bar
                if holding_days >= 3:
                    self.position.close()
                    self._position_entry_bar = None
                    return

        # Entry signal: if direction flips from -1 to 1, and we are not in a position, then buy.
        # 优化：避免重复条件检查
        if (previous_direction == -1 and current_direction == 1 and 
            not self.position):
            
            # 计算止损价格，使用numpy操作提升性能
            stop_loss_price = self.data.Close[-1] * 0.95
            self.buy(sl=stop_loss_price)
            
            # 记录进入bar
            self._position_entry_bar = len(self.data) - 1 