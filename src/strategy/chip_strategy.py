from datetime import datetime, timedelta
from typing import Dict, List
import numpy as np

from src.data_store.data_store import DataStore
from src.platform.platform import Platform
from src.utils.common import DataCategory, Instrument, Market, OrderSide, DataColumn, TechnicalIndicator, DataCategoryColumn
from src.strategy.strategy import Strategy


class ChipStrategy(Strategy):
    def __init__(
        self,
        platform: Platform,
        data_store: DataStore,
        cash: float,
        config: Dict[str, any],
        log_level: str = "INFO",
    ):
        super().__init__(platform, data_store, cash, config, log_level)
        

    def prepare_data(self, start_date: datetime, end_date: datetime):
        data_start_date = start_date - timedelta(days=250) # extract more data for technical indicator
        [self.open_, self.high_, self.low_, self.close_, self.volume_], dates, codes = self.data_store.get_data(
            market=Market.TW,
            instrument=Instrument.Stock,
            data_category=DataCategory.Daily_Price,
            data_columns=[DataColumn.Open, DataColumn.High, DataColumn.Low, DataColumn.Close, DataColumn.Volume],
            start_date=data_start_date,
            end_date=end_date,
        )
        

        # expected to have the same shape as close
        self.foreign_total_holdings_ratio_, self.local_self_holdings_ratio_, self.local_investor_holdings_ratio_ = self.data_store.get_aligned_data(
            target_dates=dates,
            target_codes=codes,
            market=Market.TW,
            instrument=Instrument.Stock,
            data_category=DataCategory.Chip,
            data_columns=[
                DataColumn.Foreign_Total_Holdings_Ratio,
                DataColumn.Local_Self_Holdings_Ratio,
                DataColumn.Local_Investor_Holdings_Ratio,
            ],
            fill_missing_date=False
        )

        self.foreign_total_holdings_ratio_sma_ = self.data_store.get_technical_indicator(TechnicalIndicator.SMA, self.foreign_total_holdings_ratio_, self.config["chip_strategy"]["foreign_total_holdings_ratio_sma_period"])
        self.local_self_holdings_ratio_sma_ = self.data_store.get_technical_indicator(TechnicalIndicator.SMA, self.local_self_holdings_ratio_, self.config["chip_strategy"]["local_self_holdings_ratio_sma_period"])
        self.local_investor_holdings_ratio_sma_ = self.data_store.get_technical_indicator(TechnicalIndicator.SMA, self.local_investor_holdings_ratio_, self.config["chip_strategy"]["local_self_holdings_ratio_sma_period"])

        self.get_signal()
        self.trading_dates = self.slice_data(dates, start_date, end_date)
        self.trading_codes = codes


    def get_signal(self):
        self.in_signal_array_ = self.foreign_total_holdings_ratio_ > self.foreign_total_holdings_ratio_sma_
        self.out_signal_array_ = self.foreign_total_holdings_ratio_ < self.foreign_total_holdings_ratio_sma_
        

    def step(self, trading_date: datetime):
        super().step(trading_date)

        strategy_one_config = self.config["strategy_one"]
        stop_loss_ratio = strategy_one_config["stop_loss_ratio"]

        codes = self.get_trading_codes()
        open_price, high_price, low_price, close_price, in_signal, out_signal = self.open_[-1], self.high_[-1], self.low_[-1], self.close_[-1], self.in_signal_array_[-1], self.out_signal_array_[-1]
        
        # adjust position
        for order_record in list(self.holdings.values()):
            code = order_record.order.code
            code_idx = codes.index(code)
            stop_loss_price = order_record.info["stop_loss_price"]

            if close_price[code_idx] < stop_loss_price:
                self.cover_order(
                    order_record.order.order_id,
                    stop_loss_price,
                    order_record.order.volume,
                    "stop_loss"
                )
                continue

            order_record.info["holding_days"] += 1
            if out_signal[code_idx]:
                self.cover_order(
                    order_record.order.order_id,
                    close_price[code_idx],
                    order_record.order.volume,
                    "out_signal"
                )


        fixed_cash = 100000
        # make decision
        for idx, signal in enumerate(in_signal):
            if not signal:
                continue

            code = codes[idx]
            price = close_price[idx]
            if np.isnan(price):
                print(f"price is nan for {code} at {trading_date}, skip")
                continue

            volume = signal * fixed_cash // price
            stop_loss_price = price * (1 - stop_loss_ratio)
            self.place_order(
                market=Market.TW,
                instrument=Instrument.Stock,
                code=code,
                price=price,
                volume=volume,
                side=OrderSide.Buy,
                info= { "stop_loss_price": stop_loss_price, "holding_days": 0 }
            )
