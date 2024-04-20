import math

# Import the backtrader platform
import backtrader as bt
import backtrader.indicators as btind

from src.utils import PriceType, TimePeriod


class SmaStrategy(bt.Strategy):
    params = (("diff_ratio", 0.01),)

    def log(self, txt, dt=None):
        """Logging function fot this strategy"""
        dt = dt or self.datas[0].datetime.date(0)
        print(f"{dt.isoformat()}, {txt}")

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.trade_close = self.getdatabyname(PriceType.NON_ADJUSTED.value).close

        self.data_close = self.getdatabyname(PriceType.ADJUSTED.value).close
        self.data_low = self.getdatabyname(PriceType.ADJUSTED.value).low
        self.data_open = self.getdatabyname(PriceType.ADJUSTED.value).open
        self.week_ma = btind.SMA(self.data_close, period=TimePeriod.WEEK)
        self.half_month_ma = btind.SMA(self.data_close, period=TimePeriod.HALF_MONTH)
        self.month_ma = btind.SMA(self.data_close, period=TimePeriod.MONTH)
        self.quarter_ma = btind.SMA(self.data_close, period=TimePeriod.QUARTER)
        self.half_year_ma = btind.SMA(self.data_close, period=TimePeriod.QUARTER * 2)
        self.year_ma = btind.SMA(self.data_close, period=TimePeriod.YEAR)

        self.week_half_month_diff = (
            self.week_ma - self.half_month_ma
        ) / self.half_month_ma
        self.half_month_ma_month_diff = (
            self.half_month_ma - self.month_ma
        ) / self.month_ma

        # To keep track of pending orders and buy price/commission
        self.order = None
        self.buyprice = None
        self.buycomm = None

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    "BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f"
                    % (order.executed.price, order.executed.value, order.executed.comm)
                )

                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:  # Sell
                self.log(
                    "SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f"
                    % (order.executed.price, order.executed.value, order.executed.comm)
                )

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("Order Canceled/Margin/Rejected")

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log("OPERATION PROFIT, GROSS %.2f, NET %.2f" % (trade.pnl, trade.pnlcomm))

    def next(self):
        # Simply log the closing price of the series from the reference
        self.log(f"{self.trade_close[0]}, {self.position.size}")
        if self.position.size > 0:
            if (self.trade_close[0] - self.position.price) / self.trade_close[0] > 0.05:
                self.log(f"BUY MORE CREATE, {self.trade_close[0]}")
                self.order = self.buy(
                    PriceType.NON_ADJUSTED.value, price=self.trade_close[0]
                )
            elif (self.trade_close[0] - self.position.price) / self.trade_close[
                0
            ] < -0.05:
                self.log(f"SELL CREATE FOR 5%, {self.trade_close[0]}")
                self.order = self.sell(
                    PriceType.NON_ADJUSTED.value,
                    price=self.trade_close[0],
                    size=math.ceil(self.position.size / 2),
                )
            elif self.data_low[0] < self.week_ma[0]:
                self.log(f"SELL CREATE FOR CROSS 5MA, {self.trade_close[0]}")
                self.order = self.sell(
                    PriceType.NON_ADJUSTED.value,
                    price=self.trade_close[0],
                    size=math.ceil(self.position.size),
                )
        elif (
            self.week_half_month_diff[0] < self.params.diff_ratio
            and self.half_month_ma_month_diff[0] < self.params.diff_ratio
            and self.data_close[0] > self.data_open[0]
            and self.data_close[0] > self.week_ma[0]
            and self.data_close[0] > self.half_month_ma[0]
            and self.data_close[0] > self.month_ma[0]
            and self.data_close[-1] > self.week_ma[-1]
            and self.data_close[-1] > self.half_year_ma[-1]
            and self.data_close[-1] > self.month_ma[-1]
            and self.data_low[-2] < self.week_ma[-2]
            and self.data_low[-2] < self.half_month_ma[-2]
            and self.data_low[-2] < self.month_ma[-2]
        ):
            self.log(f"BUY CREATE, {self.trade_close[0]}")
            self.order = self.buy(
                PriceType.NON_ADJUSTED.value, price=self.trade_close[0]
            )
