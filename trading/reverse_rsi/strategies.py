import numpy as np
from backtesting import Strategy
from reverse_rsi_indicators import (
    rsi_upper_band, rsi_mid_band, rsi_lower_band,
    rsi_upper_outer, rsi_lower_outer,
    entered_upper_band, entered_lower_band,
    supertrend_on_mid, trend_shift_bullish, trend_shift_bearish,
    regular_bullish_divergence, regular_bearish_divergence,  # 只用牛背离
)

class ScanStratLongOnly(Strategy):
    # —— 常用可调参数 —— #
    len = 14
    st_atr_len = 10
    st_factor = 2.4
    level_high = 70.0
    level_low  = 30.0
    band_factor = 0.25
    risk_R = 1.5     # 目标至少 1.5R；若 upper_outer 更远会取更大者
    sl_pad = 0.001   # 止损缓冲

    def init(self):
        c = self.data.Close

        # 价格带
        self.ob  = self.I(rsi_upper_band,  c, self.len, self.level_high, True, 14, plot=False)
        self.mid = self.I(rsi_mid_band,    c, self.len, True, 14, plot=False)
        self.os  = self.I(rsi_lower_band,  c, self.len, self.level_low,  True, 14, plot=False)
        self.uouter = self.I(rsi_upper_outer, c, self.len, self.level_high, True, 14, self.band_factor, plot=False)
        self.louter = self.I(rsi_lower_outer, c, self.len, self.level_low,  True, 14, self.band_factor, plot=False)

        # 入带信号
        self.ent_up = self.I(entered_upper_band, c, self.len, self.level_high, True, 14, plot=False)
        self.ent_lo = self.I(entered_lower_band, c, self.len, self.level_low,  True, 14, plot=False)

        # Supertrend（基于 mid）
        st, d = supertrend_on_mid(c, self.len, self.st_atr_len, self.st_factor, True, 14)
        self.st   = self.I(lambda x: st, c, plot=False)
        self.st_d = self.I(lambda x: d,  c, plot=False)
        self.ts_bull = self.I(trend_shift_bullish, c, self.len, self.st_atr_len, self.st_factor, True, 14, plot=False)
        self.ts_bear = self.I(trend_shift_bearish, c, self.len, self.st_atr_len, self.st_factor, True, 14, plot=False)

        # 背离（只用牛背离）
        self.bull_div = self.I(regular_bullish_divergence, c, self.len, 3, 1, 5, 60, plot=False)

    def next(self):
        c = self.data.Close[-1]
        mid = self.mid[-1]
        st  = self.st[-1]
        dir_up = (self.st_d[-1] == 1.0)
        long_setup = False

        # —— 入场：只做多（无持仓时）—— #
        if not self.position:
            # 1) 趋势切换做多
            if self.ts_bull[-1] == 1 and np.isfinite(mid) and c > mid:
                long_setup = True

            # 2) 上升趋势回撤重获：刚进下带 + 重新站回 mid
            if not long_setup and dir_up and self.ent_lo[-1] == 1:
                prev_c   = self.data.Close[-2] if len(self.data) > 1 else np.nan
                prev_mid = self.mid[-2]        if len(self.mid)  > 1 else np.nan
                if np.isfinite(mid) and prev_c <= prev_mid and c > mid:
                    long_setup = True

            # 3) 上升趋势中的牛背离 + 在 st 之上
            if not long_setup and dir_up and self.bull_div[-1] == 1 and np.isfinite(st) and c > st:
                long_setup = True

            if long_setup:
                lo = self.louter[-1]
                uo = self.uouter[-1]

                # 初始止损：优先用 lower_outer，其次用 supertrend，最后回退 3%
                if np.isfinite(lo):
                    sl = lo * (1.0 - self.sl_pad)
                elif np.isfinite(st):
                    sl = st * (1.0 - self.sl_pad)
                else:
                    sl = c * 0.97

                # 目标：max( risk_R * (c - sl), upper_outer - c )
                if np.isfinite(uo) and np.isfinite(sl) and c > sl:
                    tp = c + max(self.risk_R * (c - sl), uo - c)
                else:
                    tp = c * 1.06  # 回退目标

                self.buy(sl=sl, tp=tp)

        # —— 出场：只对多头 —— #
        if self.position:
            # 1) 熊切换；2) 跌破 supertrend 基线
            if self.ts_bear[-1] == 1 or (np.isfinite(st) and c < st):
                self.position.close()
