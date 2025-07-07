from backtesting import Strategy
from backtesting.lib import crossover

from trading.ml_adaptive_super_trend.indicators import ml_adaptive_super_trend

class MlAdaptiveSuperTrendStrategy(Strategy):
    atr_len = 10
    fact = 3.0
    training_data_period = 100
    highvol = 0.75
    midvol = 0.5
    lowvol = 0.25

    def init(self):
        self.st, self.direction = self.I(
            ml_adaptive_super_trend,
            self.data.High,
            self.data.Low,
            self.data.Close,
            atr_len=self.atr_len,
            fact=self.fact,
            training_data_period=self.training_data_period,
            highvol=self.highvol,
            midvol=self.midvol,
            lowvol=self.lowvol,
        )

    def next(self):
        # Bullish signal: direction crosses from -1 (down) to 1 (up)
        if self.direction[-2] == -1 and self.direction[-1] == 1:
            self.position.close()
            self.buy()
        # Bearish signal: direction crosses from 1 (up) to -1 (down)
        elif self.direction[-2] == 1 and self.direction[-1] == -1:
            self.position.close()
            self.sell() 