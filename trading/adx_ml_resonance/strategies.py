from backtesting import Strategy
from backtesting.lib import crossover

from trading.adx_ml_resonance.indicators import adx_ml_resonance_indicator


class AdxMlResonanceStrategy(Strategy):
    """
    ADX和ML SuperTrend共振策略（高级版本）
    
    策略逻辑：
    - 买入信号需要：ADX强势 + ML转涨 + (DI+ - DI-) > di_threshold + 成交量放大
    - 卖出信号需要：ADX强势 + ML转跌 + (DI- - DI+) > di_threshold + 成交量放大
    
    包含DI差值条件的增强版逻辑，确保趋势方向确认。
    新增成交量过滤：要求当前成交量大于过去20根K线的平均成交量。
    """
    
    # 策略参数
    adx_length = 10
    adx_threshold = 25
    ml_atr_len = 10
    ml_fact = 3.0
    training_data_period = 100
    highvol = 0.75
    midvol = 0.5
    lowvol = 0.25
    
    # DI差值阈值参数
    di_threshold = 5  # DI+和DI-的差值阈值
    
    # 成交量过滤参数
    volume_ma_period = 5  # 成交量移动平均周期
    
    # 风控参数
    stop_loss_pct = 0.04  # 4% 止损
    take_profit_pct = 0.12  # 12% 止盈
    max_holding_days = 3

    def init(self):
        """初始化策略指标"""
        # 使用增强版共振指标（包含DI差值条件）
        (self.adx, self.di_plus, self.di_minus, 
         self.ml_st, self.ml_direction, self.resonance_signal) = self.I(
            adx_ml_resonance_indicator,
            self.data.High,
            self.data.Low,
            self.data.Close,
            adx_length=self.adx_length,
            adx_threshold=self.adx_threshold,
            ml_atr_len=self.ml_atr_len,
            ml_fact=self.ml_fact,
            training_data_period=self.training_data_period,
            highvol=self.highvol,
            midvol=self.midvol,
            lowvol=self.lowvol,
            di_threshold=self.di_threshold,  # 启用DI差值条件
        )
        
        # 计算成交量移动平均
        self.volume_ma = self.I(self._volume_ma, self.data.Volume, self.volume_ma_period)

    def _volume_ma(self, volume, period):
        """计算成交量移动平均"""
        import pandas as pd
        return pd.Series(volume).rolling(window=period).mean()

    def _check_volume_filter(self):
        """检查成交量过滤条件"""
        import pandas as pd
        import numpy as np
        
        # 确保有足够的数据计算成交量平均值
        if len(self.data.Volume) < self.volume_ma_period:
            return False
        
        # 当前成交量大于过去20根K线的平均成交量
        current_volume = self.data.Volume[-1]
        avg_volume = self.volume_ma[-1]
        
        # 如果平均成交量为NaN或0，则跳过成交量过滤
        if pd.isna(avg_volume) or avg_volume == 0:
            return True
            
        return current_volume > avg_volume

    def next(self):
        """执行交易逻辑"""
        import pandas as pd
        
        current_bar_index = len(self.data.Close) - 1
        
        # 如果有持仓，检查退出条件
        if self.position:
            # 时间止损
            if current_bar_index - self.trades[-1].entry_bar >= self.max_holding_days:
                self.position.close()
                return
            
            # 共振反向信号止损
            if self.position.is_long and self.resonance_signal[-1] == -1:
                self.position.close()
                return
            elif self.position.is_short and self.resonance_signal[-1] == 1:
                self.position.close()
                return
            
            # ADX趋势减弱止损
            if self.adx[-1] < self.adx_threshold * 0.8:  # ADX降到阈值的80%以下
                self.position.close()
                return
        
        # 如果没有持仓，检查入场条件
        else:
            # 检查成交量过滤条件
            volume_confirmed = self._check_volume_filter()
            
            # 买入信号：共振信号为1 + 成交量确认
            if self.resonance_signal[-1] == 1 and volume_confirmed:
                self.buy(
                    sl=self.data.Close[-1] * (1 - self.stop_loss_pct),
                    tp=self.data.Close[-1] * (1 + self.take_profit_pct)
                )
            
            # 卖出信号：共振信号为-1 + 成交量确认
            elif self.resonance_signal[-1] == -1 and volume_confirmed:
                self.sell(
                    sl=self.data.Close[-1] * (1 + self.stop_loss_pct),
                    tp=self.data.Close[-1] * (1 - self.take_profit_pct)
                ) 