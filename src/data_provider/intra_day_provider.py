import csv
import os
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List
from collections import defaultdict

import pandas as pd

from src.data_provider.base_provider import BaseProvider
from src.utils.common import (
    DataColumn,
    DataCategory,
    Instrument,
    Market,
    TimeFrame,
    DataCategoryColumn,
)
from src.utils.redis_client import RedisClient


class IntraDayProvider(BaseProvider):
    def __init__(
        self,
        market: Market,
        instrument: Instrument,
        data_category: DataCategory,
        data_dir: Path,
        lazy_loading: bool = True,
        meta_data_path: Path = None,
    ):
        super().__init__(
            market=market,
            instrument=instrument,
            data_category=data_category,
            unit=TimeFrame.Minute,
            data_dir=data_dir,
            column_names=DataCategoryColumn.get_columns(data_category),
            date_data_dir=data_dir,
            code_data_dir=None,
            meta_data_path=meta_data_path,
            lazy_loading=lazy_loading,
        )

    def load_date_data(self, start_year: int = None, end_year: int = None):
        cache_key = f"data_{self.get_data_provider_id()}"

        if RedisClient.has(cache_key):
            self.date_data = RedisClient.get_json(cache_key)
            return
        self.date_data = defaultdict(list)
        # intra day data path is like '{year}/{date}.csv'
        year_dirs = sorted(os.listdir(self.date_data_dir))
        if start_year != None:
            year_dirs = list(filter(lambda x: int(x) >= start_year, year_dirs))

        if end_year != None:
            year_dirs = list(filter(lambda x: int(x) <= end_year, year_dirs))

        for year_dir in year_dirs:
            year_dir_path = self.date_data_dir / year_dir
            file_names = os.listdir(year_dir_path)
            for file_name in file_names:
                file_path = f"{year_dir_path}/{file_name}"

                with open(file_path) as fp:
                    rows = list(csv.reader(fp))[1:]
                    for row in rows:
                        row_date = datetime.fromisoformat(row[1]).isoformat()
                        self.date_data[row_date].append(row)

        # store all_code
        print(f"load data from {start_year} to {end_year} done, loaded {len(self.date_data)} date data")
        RedisClient.set_json(cache_key, self.date_data)
