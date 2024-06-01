import csv
import os
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd

from src.data_provider.base_provider import BaseProvider
from src.utils.common import (
    DataCategory,
    Instrument,
    Market,
    TimeFrame,
    DataCategoryColumn,
)


class DailyPriceProvider(BaseProvider):
    def __init__(
        self,
        market: Market,
        instrument: Instrument,
        data_category: DataCategory,
        data_dir: Path,
        lazy_loading: bool = True,
        date_data_dir: Path = None,
        code_data_dir: Path = None,
        meta_data_path: Path = None,
    ):
        super().__init__(
            market=market,
            instrument=instrument,
            data_category=data_category,
            unit=TimeFrame.Daily,
            data_dir=data_dir,
            column_names=DataCategoryColumn.get_columns(data_category),

            date_data_dir=date_data_dir,
            code_data_dir=code_data_dir,
            meta_data_path=meta_data_path,
            lazy_loading=lazy_loading,
        )
        self.date_data_dir = date_data_dir if date_data_dir != None else data_dir / "date"
        self.code_data_dir = code_data_dir if code_data_dir != None else data_dir / "code"
        self.meta_data_path = meta_data_path if meta_data_path != None else data_dir / "meta.json"

    def get_date_data(self, date = None):
        if date != None:
            date = datetime.fromisoformat(date.date().isoformat())

        return super().get_date_data(date)
