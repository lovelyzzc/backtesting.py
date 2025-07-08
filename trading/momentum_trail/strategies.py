# -*- coding: utf-8 -*-
# @Author: lovelyzzc
# @Date: 2024-07-31

from backtesting import Strategy
from .indicators import momentum_indicator

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

    def next(self):
        """
        Define the trading logic for each bar.
        This is a long-only strategy.
        """

        # Exit conditions:
        # 1. 5% stop-loss (handled by `sl` in `buy()`).
        # 2. Take-profit after holding for 3 days.
        if self.position.is_long:
            if len(self.data) - 1 - self.trades[-1].entry_bar >= 3:
                self.position.close()
                return

        # Entry signal: if direction flips from -1 to 1, and we are not in a position, then buy.
        # A 5% stop-loss is attached to the order.
        if self.direction[-2] == -1 and self.direction[-1] == 1:
            if not self.position:
                self.buy(sl=self.data.Close[-1] * 0.95) 