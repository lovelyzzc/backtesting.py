from backtesting import Strategy
from backtesting.lib import crossover

from trading.adx.indicators import adx_indicator


class AdxStrategy(Strategy):
    """
    A long-only strategy based on the ADX and Directional Movement Index (DI).
    - Enters a long position when:
        - DI+ crosses above DI-
        - ADX is above a threshold
        - The difference between DI+ and DI- is above a certain threshold.
    - Exits when DI- crosses above DI+ or after 3 days.
    - A 5% stop-loss is used.
    """
    di_length = 10
    adx_threshold = 20
    di_diff_threshold = 5
    max_holding_days = 3

    def init(self):
        """
        Initialize the strategy by calculating the ADX indicators.
        """
        self.adx, self.di_plus, self.di_minus = self.I(
            adx_indicator,
            self.data.High,
            self.data.Low,
            self.data.Close,
            length=self.di_length,
        )

    def next(self):
        """
        Define the logic for the next trading iteration.
        """
        current_bar_index = len(self.data.Close) - 1
        is_adx_strong = self.adx[-1] > self.adx_threshold
        di_diff = self.di_plus[-1] - self.di_minus[-1]

        # If a position is open, check for exit conditions
        if self.position:
            # Time-based exit
            if current_bar_index - self.trades[-1].entry_bar >= self.max_holding_days:
                self.position.close()
            # Indicator-based exit
            elif crossover(self.di_minus, self.di_plus):
                self.position.close()
        
        # If no position is open, check for entry conditions
        else:
            if (crossover(self.di_plus, self.di_minus) and
                    is_adx_strong and
                    di_diff > self.di_diff_threshold):
                self.buy(sl=self.data.Close[-1] * 0.95) 