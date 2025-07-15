from backtesting import Strategy
from backtesting.lib import crossover

from trading.adx_ml_resonance.indicators import simple_resonance_indicator, adx_ml_resonance_indicator


class AdxMlResonanceStrategy(Strategy):
    """
    基于ADX和ML自适应SuperTrend共振的交易策略
    
    策略逻辑：
    1. 当ADX显示强趋势且ML SuperTrend转为看涨时买入
    2. 当ADX显示强趋势且ML SuperTrend转为看跌时卖出
    3. 设置止损和止盈
    
    现在使用完整的K-means聚类算法来计算ML自适应SuperTrend
    """
    
    # 策略参数
    adx_length = 10
    adx_threshold = 25
    ml_atr_len = 10
    ml_fact = 3.0
    training_data_period = 100
    
    # K-means聚类参数
    highvol = 0.75
    midvol = 0.5
    lowvol = 0.25
    
    # 风控参数
    stop_loss_pct = 0.05  # 5% 止损
    take_profit_pct = 0.15  # 15% 止盈
    max_holding_days = 10  # 最大持仓天数

    def init(self):
        """初始化策略指标"""
        self.buy_signal, self.sell_signal, self.adx, self.ml_direction = self.I(
            simple_resonance_indicator,
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
        )

    def next(self):
        """执行交易逻辑"""
        current_bar_index = len(self.data.Close) - 1
        
        # 如果有持仓，检查退出条件
        if self.position:
            # 时间止损：持仓超过最大天数
            if current_bar_index - self.trades[-1].entry_bar >= self.max_holding_days:
                self.position.close()
                return
            
            # 指标止损：ML SuperTrend方向反转
            if self.position.is_long and self.sell_signal[-1]:
                self.position.close()
                return
            elif self.position.is_short and self.buy_signal[-1]:
                self.position.close()
                return
        
        # 如果没有持仓，检查入场条件
        else:
            # 买入信号
            if self.buy_signal[-1]:
                self.buy(
                    sl=self.data.Close[-1] * (1 - self.stop_loss_pct),
                    tp=self.data.Close[-1] * (1 + self.take_profit_pct)
                )
            
            # 卖出信号
            elif self.sell_signal[-1]:
                self.sell(
                    sl=self.data.Close[-1] * (1 + self.stop_loss_pct),
                    tp=self.data.Close[-1] * (1 - self.take_profit_pct)
                )


class AdvancedAdxMlResonanceStrategy(Strategy):
    """
    进阶版ADX和ML SuperTrend共振策略
    
    在AdxMlResonanceStrategy基础上，额外增加DI条件：
    - 买入信号需要：ADX强势 + ML转涨 + (DI+ - DI-) > di_threshold
    - 卖出信号需要：ADX强势 + ML转跌 + (DI- - DI+) > di_threshold
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

    def next(self):
        """执行交易逻辑"""
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
            # 买入信号：共振信号为1
            if self.resonance_signal[-1] == 1:
                self.buy(
                    sl=self.data.Close[-1] * (1 - self.stop_loss_pct),
                    tp=self.data.Close[-1] * (1 + self.take_profit_pct)
                )
            
            # 卖出信号：共振信号为-1
            elif self.resonance_signal[-1] == -1:
                self.sell(
                    sl=self.data.Close[-1] * (1 + self.stop_loss_pct),
                    tp=self.data.Close[-1] * (1 - self.take_profit_pct)
                )


class ConservativeResonanceStrategy(Strategy):
    """
    保守版共振策略
    
    只做多，信号要求更严格
    现在使用完整的K-means聚类算法来计算ML自适应SuperTrend
    """
    
    # 策略参数
    adx_length = 21  # 更长的ADX周期
    adx_threshold = 30  # 更高的ADX阈值
    ml_atr_len = 14
    ml_fact = 2.5  # 更敏感的SuperTrend
    training_data_period = 150  # 更长的训练周期
    
    # K-means聚类参数（使用更保守的设置）
    highvol = 0.8   # 更高的高波动率阈值
    midvol = 0.5
    lowvol = 0.2    # 更低的低波动率阈值
    
    # 风控参数
    stop_loss_pct = 0.03  # 3% 止损
    take_profit_pct = 0.10  # 10% 止盈
    max_holding_days = 20

    def init(self):
        """初始化策略指标"""
        self.buy_signal, self.sell_signal, self.adx, self.ml_direction = self.I(
            simple_resonance_indicator,
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
        )

    def next(self):
        """执行交易逻辑 - 只做多"""
        current_bar_index = len(self.data.Close) - 1
        
        # 如果有持仓，检查退出条件
        if self.position:
            # 时间止损
            if current_bar_index - self.trades[-1].entry_bar >= self.max_holding_days:
                self.position.close()
                return
            
            # 卖出信号或ADX减弱
            if self.sell_signal[-1] or self.adx[-1] < self.adx_threshold * 0.7:
                self.position.close()
                return
        
        # 如果没有持仓，只检查买入条件
        else:
            # 额外条件：价格在上升趋势中（简单移动平均）
            if len(self.data.Close) >= 20:
                sma_20 = sum(self.data.Close[-20:]) / 20
                price_above_sma = self.data.Close[-1] > sma_20
            else:
                price_above_sma = True
            
            # 买入信号 + 价格趋势确认
            if self.buy_signal[-1] and price_above_sma:
                self.buy(
                    sl=self.data.Close[-1] * (1 - self.stop_loss_pct),
                    tp=self.data.Close[-1] * (1 + self.take_profit_pct)
                ) 