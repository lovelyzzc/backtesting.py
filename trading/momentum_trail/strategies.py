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
        self._peak_price = 0.0

    def next(self):
        """
        Define the trading logic for each bar.
        This is a long-only strategy.
        """

        # If we are in a long position, manage the position with our custom trailing stop for profit-taking
        # and check for the primary exit signal. The 5% stop-loss is handled automatically by the `sl`
        # parameter in the `buy()` call.
        if self.position.is_long:
            # Update the peak price since entry.
            if self._peak_price == 0.: # First bar of a new trade
                self._peak_price = self.trades[-1].entry_price
            self._peak_price = max(self._peak_price, self.data.High[-1])

            # 1. Take-profit rule: Close if price drops 5% from the peak.
            if self.data.Close[-1] < self._peak_price * 0.95:
                self.position.close()
                return

            # 2. Primary exit signal from the original strategy.
            if self.direction[-2] == 1 and self.direction[-1] == -1:
                self.position.close()
                return

        # Entry signal: if direction flips from -1 to 1, and we are not in a position, then buy.
        # A 5% stop-loss is attached to the order.
        if self.direction[-2] == -1 and self.direction[-1] == 1:
            if not self.position:
                self.buy(sl=self.data.Close[-1] * 0.95)
                # Reset peak price for the new trade. It will be initialized on the next bar.
                self._peak_price = 0.0 