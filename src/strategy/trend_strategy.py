from datetime import datetime, timedelta
from typing import Dict, List
import numpy as np

from src.data_provider.base_provider import BaseProvider
from src.data_store.data_store import DataStore
from src.platform.platform import Platform
from src.utils.common import DataCategory, Instrument, Market, OrderSide, DataColumn, TechnicalIndicator
from src.utils.order import CoverOrderRecord, OrderRecord
from src.strategy.strategy import Strategy
from src.data_transformer.data_transformer import DataTransformer

class TrendStrategy(Strategy):
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

        market_index_, _, _ = self.data_store.get_data(
            market=Market.TW,
            instrument=Instrument.StockIndex,
            data_category=DataCategory.Market_Index,
            data_columns=[DataColumn.Close],
            selected_codes=["Y9999"],
            start_date=data_start_date,
            end_date=end_date,
        )

        self.relative_strength_ = DataTransformer.get_relative_strength(codes, self.close_, market_index_)
        self.relative_strength_sma_ = self.data_store.get_technical_indicator(TechnicalIndicator.SMA, self.relative_strength_, self.config["strategy_one"]["rs_sma_period"])
        
        # expected to have the same shape as close
        # self.eps_, self.recurring_eps_ = self.data_store.get_aligned_data(target_dates=dates, target_codes=codes, market=Market.TW, instrument=Instrument.Stock, data_category=DataCategory.Finance_Report, data_columns=[DataColumn.EPS, DataColumn.Recurring_EPS])

        # local min & max array
        self.signal_one_, _ = DataTransformer.get_signal_one(self.config, self.close_, self.low_, self.high_, self.volume_, self.relative_strength_sma_)
        self.trading_dates = self.slice_data(dates, start_date, end_date)
        self.trading_codes = codes


    def step(self, trading_date: datetime):
        super().step(trading_date)
        strategy_one_config = self.config["strategy_one"]
        holding_days = strategy_one_config["holding_days"]
        stop_loss_ratio = strategy_one_config["stop_loss_ratio"]
        rs_threshold = strategy_one_config["rs_threshold"]

        codes = self.get_trading_codes()
        open_price, high_price, low_price, close_price, signal_one, relative_strength_sma_ = self.open_[-1], self.high_[-1], self.low_[-1], self.close_[-1], self.signal_one_[-1], self.relative_strength_sma_[-1]
        
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
            if order_record.info["holding_days"] >= holding_days:
                self.cover_order(
                    order_record.order.order_id,
                    close_price[code_idx],
                    order_record.order.volume,
                    "holding_days"
                )
        
        fixed_cash = 100000
        # make decision
        for idx, signal in enumerate(signal_one):
            if signal == 0:
                continue
            
            # filter rs threshold
            if relative_strength_sma_[idx] < rs_threshold:
                continue
            
            # add volume by rs
            # if relative_strength_sma_5[idx] >= 0.95:
            #     fixed_cash *= 1.8
            # elif relative_strength_sma_5[idx] >= 0.9:
            #     fixed_cash *= 1.2
            # elif relative_strength_sma_4[idx] >= 0.95:
            #     fixed_cash *= 1.4
            


            # # filter eps
            # if not (self.recurring_eps_[-1, idx] >= self.recurring_eps_[-120:, idx]).all():
            #     continue
            
            # min_eps = self.recurring_eps_[-120:, idx].min()
            # if (self.recurring_eps_[-1, idx] - self.recurring_eps_[-120:, idx].min()) / np.abs(min_eps) < 0.1:
            #     continue


            code = codes[idx]
            price = close_price[idx]
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
