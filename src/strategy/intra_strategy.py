import numpy as np
import pandas as pd
import json
from datetime import datetime
from typing import Dict
from pathlib import Path

from src.data_provider.base_provider import BaseProvider
from src.data_store.data_store import DataStore, DataColumn
from src.platform.platform import Platform
from src.utils.common import DataCategory, DataProvider, Instrument, Market, TimeFrame, OrderSide
from src.strategy.strategy import Strategy
from src.data_processor.future_record_processor import FutureRecordProcessor


class IntraStrategy(Strategy):
    def __init__(
        self,
        platform: Platform,
        data_store: DataStore,
        cash: float,
        config: Dict[str, any],
        log_level: str = "INFO",
    ):
        super().__init__(platform, data_store, cash, config, log_level)

    def prepare_data(self, start_date: datetime, end_date: datetime, result_dir: Path, full_record: bool):
        data_store = DataStore()
        tx_data, dates, codes = (
            self.data_store.get_data(
                Market.TW,
                Instrument.Future,
                DataCategory.Minute_Price,
                [
                    DataColumn.Open,
                    DataColumn.High,
                    DataColumn.Low,
                    DataColumn.Close,
                    DataColumn.Volume,
                ],
                start_date=start_date,
            )
        )
        twse_data = (
            data_store.get_aligned_data(
                dates,
                ["twse"],
                Market.TW,
                Instrument.StockIndex,
                DataCategory.Minute_Index,
                [DataColumn.Open, DataColumn.High, DataColumn.Low, DataColumn.Close],
                False,
            )
        )
        tx_df = pd.DataFrame(np.stack([data[:, 0].T for data in tx_data]).T, columns=['open', 'high', 'low', 'close', 'volume'], index=dates)
        twse_df = pd.DataFrame(np.stack([data[:, 0].T for data in twse_data]).T, columns=['open', 'high', 'low', 'close'], index=dates)
        tx_df = tx_df.resample('5min').agg(
          open=("open", "first"),
          high=("high", "max"),
          low=("low", "min"),
          close=("close", "last"),
          volume=("volume", "sum")
        )
        twse_df = twse_df.resample('5min').agg(
          open=("open", "first"),
          high=("high", "max"),
          low=("low", "min"),
          close=("close", "last"),
        )
        mask = tx_df.isna().any(axis=1)
        tx_df = tx_df[~mask]
        twse_df = twse_df[~mask]

        self.open_ = tx_df["open"].values.reshape(-1, 1)
        self.high_ = tx_df["high"].values.reshape(-1, 1)
        self.low_ = tx_df["low"].values.reshape(-1, 1)
        self.close_ = tx_df["close"].values.reshape(-1, 1)
        self.volume_ = tx_df["volume"].values.reshape(-1, 1)
        dates = list(tx_df.index.to_pydatetime())

        self.twse_open_ = twse_df["open"].values.reshape(-1, 1)
        self.twse_high_ = twse_df["high"].values.reshape(-1, 1)
        self.twse_low_ = twse_df["low"].values.reshape(-1, 1)
        self.twse_close_ = twse_df["close"].values.reshape(-1, 1)


        self.twse_df = twse_df
        self.day_twse_df = (
            self.twse_df.resample("D")
            .agg(
                open=("open", "first"),
                high=("high", "max"),
                low=("low", "min"),
                close=("close", "last"),
            )
            .dropna()
        )
        self.day_gap_size = (self.day_twse_df["open"] - self.day_twse_df["close"].shift(1))
        self.long_gaps = []
        self.short_gaps = []
        self.filled_gaps = []

        self.trading_dates = self.slice_data(dates, start_date, end_date)
        self.trading_codes = codes

    def step(self, trading_date: datetime):
        self.current_trading_date = trading_date
        open_price = self.open_[-1][0]
        high_price = self.high_[-1][0]
        low_price = self.low_[-1][0]
        close_price = self.close_[-1][0]
        twse_open_price = self.twse_open_[-1][0]
        twse_high_price = self.twse_high_[-1][0]
        twse_low_price = self.twse_low_[-1][0]
        twse_close_price = self.twse_close_[-1][0]
        is_night_market = FutureRecordProcessor.is_night_market(trading_date)
        is_twse_opened = FutureRecordProcessor.is_twse_open(trading_date)
        is_twse_open_bar = FutureRecordProcessor.is_twse_open_bar(trading_date)
        is_twse_close_bar = FutureRecordProcessor.is_twse_close_bar(trading_date)
        # config
        stop_loss_ratio = 2
        stop_profit_ratio = 3
        gap_size_threshold_ratio = 0.005
        print(self.current_trading_date, len(self.long_gaps), len(self.short_gaps))
        # check stop loss and stop profit
        self.check_order()

        # identify gap
        self.identify_gap(gap_size_threshold_ratio)

        # check if gap is filled
        self.check_gap_filled()

        # check if nearest gap if tested
        self.trade_gap_tested(stop_loss_ratio, stop_profit_ratio)

    def identify_gap(self, gap_size_threshold: float):
        is_twse_open_bar = FutureRecordProcessor.is_twse_open_bar(self.current_trading_date)
        twse_open_price = self.twse_open_[-1][0]

        if not is_twse_open_bar:
            return
        if pd.Timestamp(self.current_trading_date.date()) not in self.day_twse_df.index:
            return
        
        date_idx = self.day_gap_size.index.get_loc(pd.Timestamp(self.current_trading_date.date()))
        if date_idx < 10:
            return

        prev_twse_close_price = self.day_twse_df.iloc[date_idx-1]["close"]
        gap_size = self.day_gap_size[date_idx]
        # calculate the mean gap size of the last 5 days

        if gap_size == 0:
            return

        if gap_size > 0 and gap_size > gap_size_threshold:
            gap = {
                "open_date": self.current_trading_date,
                "size": gap_size,
                "side": "long",
                "upper": twse_open_price,
                "lower": prev_twse_close_price,
                "order_id": None,
            }
            self.long_gaps.append(gap)
            # sort self.long_gaps by upper
            self.long_gaps = sorted(self.long_gaps, key=lambda x: x["upper"])
        elif gap_size < 0 and gap_size < -gap_size_threshold:
            gap = {
                "open_date": self.current_trading_date,
                "size": -gap_size,
                "side": "short",
                "upper": prev_twse_close_price,
                "lower": twse_open_price,
                "order_id": None,
            }
            self.short_gaps.append(gap)
            # sort self.short_gaps by lower
            self.short_gaps = sorted(
                self.short_gaps, key=lambda x: x["lower"], reverse=True
            )

    def check_gap_filled(self):
        is_twse_opened = FutureRecordProcessor.is_twse_open(self.current_trading_date)
        twse_high_price = self.twse_high_[-1][0]
        twse_low_price = self.twse_low_[-1][0]
        close_price = self.close_[-1][0]
        if not is_twse_opened:
            return

        lower_sorted_long_gaps = sorted(self.long_gaps, key=lambda x: x["lower"])
        while len(lower_sorted_long_gaps) > 0:
            long_gap = lower_sorted_long_gaps[-1]
            if twse_low_price < long_gap["lower"]:
                if long_gap["order_id"] is not None and not self.order_record_dict[long_gap["order_id"]].is_covered:
                    self.cover_order(long_gap["order_id"], close_price, 1, "gap_filled")

                long_gap["close_date"] = self.current_trading_date
                filled_gap = lower_sorted_long_gaps.pop()
                self.long_gaps.pop(self.long_gaps.index(filled_gap))
                self.filled_gaps.append(filled_gap)
            else:
                break

        upper_sorted_short_gaps = sorted(self.short_gaps, key=lambda x: x["upper"], reverse=True)
        while len(upper_sorted_short_gaps) > 0:
            short_gap = upper_sorted_short_gaps[-1]
            if twse_high_price > short_gap["upper"]:
                if short_gap["order_id"] is not None and not self.order_record_dict[short_gap["order_id"]].is_covered:
                    self.cover_order(short_gap["order_id"], close_price, 1, "gap_filled")

                short_gap["close_date"] = self.current_trading_date
                filled_gap = upper_sorted_short_gaps.pop()
                self.short_gaps.pop(self.short_gaps.index(filled_gap))
                self.filled_gaps.append(filled_gap)
            else:
                break

    def trade_gap_tested(self, stop_loss_ratio: float, stop_profit_ratio: float):
        close_price = self.close_[-1][0]

        if len(self.long_gaps) > 0 and self.long_gaps[-1]["order_id"] is None:
            long_gap = self.long_gaps[-1]
            if  (self.low_[-3:] < long_gap["upper"]).all():
                order_id = self.place_order(
                    Market.TW,
                    Instrument.Future,
                    "TX",
                    close_price,
                    1,
                    OrderSide.Buy,
                    info={
                        "stop_loss_price": close_price
                        - long_gap["size"] * stop_loss_ratio,
                        "stop_profit_price": close_price
                        + long_gap["size"] * stop_profit_ratio,
                    },
                    custom_record_field={"gap": long_gap},
                )
                long_gap["order_id"] = order_id
                return True

        if len(self.short_gaps) > 0 and self.short_gaps[-1]["order_id"] is None:
            short_gap = self.short_gaps[-1]
            if (self.high_[-3:] > short_gap["lower"]).all():
                order_id = self.place_order(
                    Market.TW,
                    Instrument.Future,
                    "TX",
                    close_price,
                    1,
                    OrderSide.Sell,
                    info={
                        "stop_loss_price": close_price + short_gap["size"] * stop_loss_ratio,
                        "stop_profit_price": close_price - short_gap["size"] * stop_profit_ratio,
                    },
                    custom_record_field={"gap": short_gap},
                )
                short_gap["order_id"] = order_id
                return True

    def check_order(self):
        close_price = self.close_[-1][0]
        for order_record in list(self.holdings.values()):
            order = order_record.order
            stop_loss_price = order_record.info["stop_loss_price"]
            stop_profit_price = order_record.info["stop_profit_price"]
            if order_record.order.side == OrderSide.Buy:
                if close_price < stop_loss_price:
                    self.cover_order(order.order_id, close_price, 1, "stop_loss")
                elif close_price >= stop_profit_price:
                    self.cover_order(order.order_id, stop_profit_price, 1, "stop_profit")
            elif order_record.order.side == OrderSide.Sell:
                if close_price > stop_loss_price:
                    self.cover_order(order.order_id, close_price, 1, "stop_loss")
                elif close_price <= stop_profit_price:
                    self.cover_order(order.order_id, stop_profit_price, 1, "stop_profit")

    def end(self, result_dir: Path, full_record):
        super().end(result_dir, full_record)
        result_dir.mkdir(parents=True, exist_ok=True)

        if full_record:
            gaps = self.long_gaps + self.short_gaps + self.filled_gaps
            with open(result_dir / "gaps.json", "w") as fp:
                json.dump(gaps, fp, default=str)