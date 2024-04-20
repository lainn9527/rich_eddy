from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from src.config import config
from src.data_provider import BaseProvider
from src.utils.common import DataCategory, Instrument, Market, TimeFrame, DataColumn, TechnicalIndicator


@dataclass
class Data:
    value: np.ndarray
    index: List[datetime]  # should be date
    column: List[str]  # should be code

# data shape is (date[old:new], code)
class DataStore:
    start_date: datetime
    end_date: datetime
    data_id_provider_dict: Dict[str, BaseProvider]
    tack_data = dict()
    codes = None

    def __init__(
        self,
        start_date: datetime = None,
        end_date: datetime = None,
        codes: List[str] = None,
    ) -> None:
        self.start_date = start_date
        self.end_date = end_date
        self.codes = codes
        self.data_id_provider_dict = self.build_data_id_provider_dict(config["data_provider"])
        self.track_data = dict()


    def get_data(
        self,
        market: Market,
        instrument: Instrument,
        data_category: DataCategory,
        data_columns: List[DataColumn],
        selected_codes: List[str] = None,
        start_date: datetime = None,
        end_date: datetime = None,
    ):
        if type(data_columns) is not list:
            data_columns = [data_columns]

        np_arrays = []
        dates = None
        codes = None
        for data_column in data_columns:
            data_id = self.build_data_id(market, instrument, data_category, data_column)
            if data_id in self.track_data:
                np_array, dates, codes =  self.track_data[data_id]["data"], self.track_data[data_id]["index"], self.track_data[data_id]["column"]
                np_arrays.append(np_array)
                continue

            data_provider = self.get_data_provider(data_id)
            np_array, dates, codes = data_provider.get_np_array(data_column, selected_codes, start_date, end_date)
            np_arrays.append(np_array)

            self.track_data[data_id] = {
                "data": np_array,
                "index": dates,
                "column": codes,
            }

        copy_np_arrays = [np_array.copy() for np_array in np_arrays]
        copy_np_arrays = copy_np_arrays if len(copy_np_arrays) > 1 else copy_np_arrays[0]
        return copy_np_arrays, dates, codes


    def get_aligned_data(
        self,
        target_dates: List[datetime],
        target_codes: List[str],
        market: Market,
        instrument: Instrument,
        data_category: DataCategory,
        data_columns: List[DataColumn],
    ):
        # the data shape will be align with target
        if type(data_columns) is not list:
            data_columns = [data_columns]
        
        np_arrays = []
        for data_column in data_columns:
            data_id = self.build_data_id(market, instrument, data_category, data_column)
            if data_id in self.track_data:
                np_array = self.track_data[data_id]
                continue

            data_provider = self.get_data_provider(data_id)
            np_array = data_provider.get_aligned_np_array(target_dates, target_codes, data_column)
            np_arrays.append(np_array)

            self.track_data[data_id] = {
                "data": np_array,
                "index": target_dates,
                "column": target_codes,
            }

        return np_arrays if len(np_arrays) > 1 else np_arrays[0]


    def get_data_by_code(
        self,
        market: Market,
        instrument: Instrument,
        data_category: DataCategory,
        data_column: DataColumn,
        codes: List[str]
    ):
        data_id = self.build_data_id(market, instrument, data_category, data_column)
        data_provider = self.get_data_provider(data_id)
        return data_provider.get_np_array_by_codes(data_column, codes)


    def get_data_item(
        self,
        market: Market,
        instrument: Instrument,
        data_category: DataCategory,
        data_column: DataColumn,
        code: str,
        trading_date: datetime,
    ):
        data_id = self.build_data_id(market, instrument, data_category, data_column)
        data_provider = self.get_data_provider(data_id)
        return data_provider.get_data_item(data_column, trading_date, code)


    def get_data_date(
        self,
        market: Market,
        instrument: Instrument,
        data_category: DataCategory,
        data_column: DataColumn,
    ):
        data_id = self.build_data_id(market, instrument, data_category, data_column)
        data_provider = self.get_data_provider(data_id)
        return data_provider.get_all_dates()


    def get_data_code(
        self,
        market: Market,
        instrument: Instrument,
        data_category: DataCategory,
        data_column: DataColumn,
    ):
        data_id = self.build_data_id(market, instrument, data_category, data_column)
        data_provider = self.get_data_provider(data_id)
        return data_provider.get_all_codes()
            

    def get_technical_indicator(self, indicator: TechnicalIndicator, *argv) -> None:
        if indicator == TechnicalIndicator.SMA:
            # source: (date, code)
            source, time_period = argv[0], argv[1]
            sma = sliding_window_view(source, time_period, axis=0).mean(axis=2)
            return np.concatenate([np.full((time_period - 1, source.shape[1]), np.nan), sma], axis=0)

        elif indicator == TechnicalIndicator.RS:
            pass


    def step_data(
        self,
        trading_date: datetime,
        unit: TimeFrame = TimeFrame.Daily,
    ) -> None:
        # move all data for 1 step
        pass


    def get_data_provider(self, data_id: str) -> BaseProvider:
        if data_id not in self.data_id_provider_dict:
            raise ValueError(f"Data id {data_id} not found in data store")

        return self.data_id_provider_dict[data_id]


    def build_data_id(
        self, market: Market, instrument: Instrument, data_category: DataCategory, data_column: DataColumn
    ):
        return f"{market.value}_{instrument.value}_{data_category.value}_{data_column.value}"


    def build_data_id_provider_dict(self, data_provider_config: List[Dict]) -> None:
        data_provider_configs = deepcopy(data_provider_config)

        data_id_dict = dict()
        for data_provider_config in data_provider_configs:
            data_provider_class = data_provider_config.pop("data_provider_class")
            data_provider_instance = data_provider_class(**data_provider_config)
            
            for column in data_provider_instance.column_names:
                data_id = self.build_data_id(
                    data_provider_instance.market,
                    data_provider_instance.instrument,
                    data_provider_instance.data_category,
                    column,
                )
                data_id_dict[data_id] = data_provider_instance
        
        return data_id_dict
