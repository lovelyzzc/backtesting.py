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
        # A long signal is generated when the direction flips from -1 to 1.
        if self.direction[-2] == -1 and self.direction[-1] == 1:
            if self.position.is_short:
                self.position.close()
            self.buy()

        # A short signal (sell signal) only closes the long position, but does not open a short one.
        elif self.direction[-2] == 1 and self.direction[-1] == -1:
            if self.position.is_long:
                self.position.close() 