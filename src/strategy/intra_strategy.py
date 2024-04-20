from datetime import datetime
from typing import Dict

from src.data_provider.base_provider import BaseProvider
from src.data_store.data_store import DataStore
from src.utils.common import DataCategory, DataProvider, Instrument, Market, TimeFrame


class IntraStrategy:
    cash: float
    cash_history: Dict[datetime, float]
    data_provider_dict: Dict[str, BaseProvider]
    data_store: DataStore

    def __init__(self, cash, data_store: DataStore):
        self.cash = cash
        self.cash_history = dict()
        self.data_store = data_store

        self.prepare_data()

    def prepare_data(self):
        # the data shape is (date[old:new], code)
        (
            self.open,
            self.high,
            self.low,
            self.close,
            self.volume,
        ) = self.data_store.get_data(
            Market.TW, Instrument.Stock, DataCategory.Daily_Price
        )

        # expected to have the same shape as close
        self.eps = self.data_store.get_aligned_data(
            self.close, Market.TW, Instrument.Stock, DataCategory.EPS
        )
        self.eps = self.data_store.get_aligned_data(self.close, self.eps, self.close)

        # chip
        self.chip_fv = self.data_store.get_aligned_data(
            self.close, Market.TW, Instrument.Stock, DataCategory.Chip_Foreign_Volume
        )
        self.chip_liv = self.data_store.get_aligned_data(
            self.close,
            Market.TW,
            Instrument.Stock,
            DataCategory.Chip_Local_Investor_Volume,
        )
        self.chip_lsv = self.data_store.get_aligned_data(
            self.close, Market.TW, Instrument.Stock, DataCategory.Chip_Local_Self_Volume
        )

    def step(self, trading_date: datetime):
        # for each iteration, the data will be truncated to the :trading_date
        self.data_store.step_data(trading_date, unit=TimeFrame.Minute)
        self.cash_history[trading_date] = self.cash

        # recognize signal
        # make decision
        # execute order
        # update cash
        pass
