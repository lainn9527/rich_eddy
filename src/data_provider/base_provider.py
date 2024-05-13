import os
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from multiprocessing import Pool

import numpy as np

from src.utils.common import DataCategory, DataColumn, Instrument, Market
from src.utils.utils import split_payload

def task(payload):
    date_data_dir, year_dirs = payload
    date_data = dict()
    for year_dir in year_dirs:
        year_dir_path = date_data_dir / year_dir
        file_names = os.listdir(year_dir_path)
        for file_name in file_names:
            file_path = f"{year_dir_path}/{file_name}"

            # the format of file_name is 'yyyy-mm-dd.csv'
            file_date = datetime.strptime(file_name.split(".")[0], "%Y-%m-%d")
            with open(file_path) as fp:
                date_data[file_date.isoformat()] = list(csv.reader(fp))[1:]  # skip column row
    
    return date_data

class BaseProvider:
    market: Market
    instrument: Instrument
    data_category: DataCategory

    date_data: Dict[datetime, any]
    code_data: Dict[str, any]
    meta_data: Dict

    all_date: List[datetime]
    all_code: List[str]

    date_data_dir: Path
    date_data_path: Path
    code_data_dir: Path
    meta_data_path: Path
    column_np_array: Dict[str, np.array]
    column_names: List[DataColumn]
    unit: str
    num_of_cores: int
    lazy_loading: bool

    def __init__(
        self,
        market: Market,
        instrument: Instrument,
        data_category: DataCategory,
        unit: str,
        data_dir: Path,
        column_names: List[DataColumn],
        num_of_cores: int = 7,
        lazy_loading: bool = True,
        date_data_dir: Path = None,
        code_data_dir: Path = None,
        meta_data_path: Path = None,
    ):
        self.market = market
        self.instrument = instrument
        self.data_category = data_category
        self.unit = unit

        self.data_dir = data_dir
        self.date_data_dir = date_data_dir if date_data_dir != None else data_dir / "date"
        self.code_data_dir = code_data_dir if code_data_dir != None else data_dir / "code"
        self.meta_data_path = meta_data_path if meta_data_path != None else data_dir / "meta.json"

        self.column_names = column_names
        self.lazy_loading = lazy_loading
        self.num_of_cores = num_of_cores

        self.meta_data = dict()
        self.code_data = dict()
        self.date_data = dict()
        self.all_date = None
        self.all_code = None
        self.np_array_column = dict()
        self.aligned_np_array_column = dict()


    def load_date_data(self, start_year: int = None, end_year: int = None):
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

                # the format of file_name is 'yyyy-mm-dd.csv'
                file_date = datetime.strptime(file_name.split(".")[0], "%Y-%m-%d")
                with open(file_path) as fp:
                    self.date_data[file_date.isoformat()] = list(csv.reader(fp))[1:]  # skip column row

        print(
            f"load data from {start_year} to {end_year} done, loaded {len(self.date_data)} date data"
        )


    def load_date_data_parallel_task(self, payload):
        date_data_dir, year_dirs = payload
        date_data = dict()
        for year_dir in year_dirs:
            year_dir_path = date_data_dir / year_dir
            file_names = os.listdir(year_dir_path)
            for file_name in file_names:
                file_path = f"{year_dir_path}/{file_name}"

                # the format of file_name is 'yyyy-mm-dd.csv'
                file_date = datetime.strptime(file_name.split(".")[0], "%Y-%m-%d")
                with open(file_path) as fp:
                    date_data[file_date.isoformat()] = list(csv.reader(fp))[1:]  # skip column row
        
        return date_data

    def load_date_data_parallel(self, start_year: int = None, end_year: int = None):
        year_dirs = sorted(os.listdir(self.date_data_dir))

        if start_year != None:
            year_dirs = list(filter(lambda x: int(x) >= start_year, year_dirs))

        if end_year != None:
            year_dirs = list(filter(lambda x: int(x) <= end_year, year_dirs))


        split_year_dirs = split_payload(year_dirs, self.num_of_cores)
        splitted_payload = [(self.date_data_dir, year_dirs) for year_dirs in split_year_dirs]
        with Pool(self.num_of_cores) as p:
            results = p.map(self.load_date_data_parallel_task, splitted_payload)

        for result in results:
            self.date_data.update(result)

        print(f"load data from {start_year} to {end_year} done, loaded {len(self.date_data)} date data")
    

    def load_data_by_codes(self, codes = None):
        if codes is None:
            file_names = os.listdir(self.code_data_dir)
            codes = [file_name.split(".")[0] for file_name in file_names]

        code_data = dict()
        for code in codes:
            with open(self.code_data_dir / f"{code}.csv") as fp:
                pv = list(csv.reader(fp))[1:]
                code_data[code] = pv
        

        print(f"{len(codes)} code data loaded")
        self.date_data = self.code_data_to_date_data(code_data)


    def code_data_to_date_data(self, code_data: Dict[str, any]):
        date_dict = dict()
        for data in code_data.values():
            for row in data:
                date_str = datetime.strptime(row[1], "%Y%m%d").isoformat()
                if date_dict.get(date_str) == None:
                    date_dict[date_str] = []
                date_dict[date_str].append(row)
        
        # sort by code
        for date, data in date_dict.items():
            date_dict[date] = sorted(data, key=lambda x: x[0])
        
        return date_dict


    def load_meta_data(self):
        with open(self.meta_data_path, "r") as fp:
            lines = list(csv.reader(fp))
            for line in lines:
                if len(line) < 6:
                    continue
                code, name, market_type, industry = line[0], line[1], line[4], line[5]
                self.meta_data[code] = f"{name} {code} {market_type} {industry}"


    def build_column_np_array(self, codes: List[str] = None, start_date = None, end_date = None):
        if codes != None:
            self.load_data_by_codes(codes)

        if self.is_date_data_loaded() == False:
            start_date_year = start_date.year if start_date != None else None
            end_date_year = end_date.year if end_date != None else None
            self.load_date_data(start_date_year, end_date_year)

        all_trading_date = self.get_all_dates()
        all_codes = self.get_all_codes()

        # x: date, y: code
        build_column_idx = []
        code_idx = np.nan
        np_array_column = dict()
        for idx, column in enumerate(self.column_names):
            if column == DataColumn.Date:
                continue
            if column == DataColumn.Code:
                code_idx = idx
                continue

            np_array_column[column] = np.full(
                (len(all_trading_date), len(all_codes)), np.nan
            )
            build_column_idx.append(idx)

        # build code: idx map
        code_idx_map = {code: idx for idx, code in enumerate(all_codes)}

        for i, trading_date in enumerate(all_trading_date):
            date_data = self.get_date_data(trading_date)
            for row in date_data:
                # replace empty value with np.nan
                row = [np.nan if x == "" else x for x in row]
                code = row[0]
                for idx in build_column_idx:
                    value = row[idx]
                    column = self.column_names[idx]
                    code_idx = code_idx_map[code]
                    np_array_column[column][i][code_idx] = value

        self.np_array_column = np_array_column


    def get_date_data(self, date = None):
        if self.date_data == {}:
            self.load_date_data()

        if date != None:
            return self.date_data[date.isoformat()]

        return self.date_data
 
    
    def get_data_item(self, column: DataColumn, trading_date: datetime, code: str):
        column_np_array, dates, codes = self.get_np_array(column)
        code_idx, date_idx = codes.index(code), dates.index(trading_date)
        return column_np_array[date_idx, code_idx].item()


    def get_np_array_by_codes(self, column: DataColumn, codes: List[str]):
        if type(codes) == str:
            codes = [codes]

        if codes == None:
            return {}
        
        column_np_array, _, codes = self.get_np_array(column, codes)
        code_data = {}
        for code in codes:
            code_idx = codes.index(code)
            code_data[code] = column_np_array[:, code_idx]

        return code_data


    def get_np_array(self, column, codes = None, start_date = None, end_date = None):
        if self.np_array_column == {}:
            self.build_column_np_array(codes, start_date, end_date)

        np_array = self.np_array_column[column]
        dates = self.get_all_dates()
        codes = self.get_all_codes()

        if start_date != None:
            if start_date not in dates:
                start_date = next(date for date in dates if date >= start_date)
            start_idx = dates.index(start_date)
            np_array, dates = np_array[start_idx:], dates[start_idx:]

        if end_date != None:
            if dates[-1] <= end_date:
                end_date = dates[-1]
            if end_date not in dates:
                end_date = next(date for date in dates if date >= end_date)
            end_idx = dates.index(end_date) + 1
            np_array, dates = np_array[:end_idx], dates[:end_idx]

        return np_array, dates, codes


    def get_aligned_np_array(self, target_dates, target_codes, column):
        if column not in self.aligned_np_array_column:
            self.align_np_array(target_dates, target_codes, column)
        
        return self.aligned_np_array_column[column]


    def to_dataframe(self):
        pass


    def to_datetime_string(self, datetime: datetime):
        return datetime.isoformat()


    def get_all_dates(self):
        if self.all_date == None:
            date_data = self.get_date_data()
            self.all_date = sorted(
                [datetime.fromisoformat(date_string) for date_string in list(date_data.keys())]
            )

        return self.all_date


    def get_all_codes(self):
        if self.all_code == None:
            date_data = self.get_date_data()
            code_set = set()

            for data in date_data.values():
                for row in data:
                    code = row[0]
                    code_set.add(code)

            self.all_code = sorted(list(code_set))
        
        return self.all_code
  
    
    def is_date_data_loaded(self):
        return self.date_data != {}
    
    
    def align_np_array(self, target_dates, target_codes, column):
        np_array, current_dates, current_codes = self.get_np_array(column, start_date=target_dates[0], end_date=target_dates[-1])

        # align code
        code_idx_map = {code: idx for idx, code in enumerate(current_codes)}

        # set the idx of code not in current_codes to -1, which means the last line
        target_code_idx = [code_idx_map[code] if code in current_codes else -1 for code in target_codes]
        # add a column with full nan to the end of array, such that the value of the code not in current_codes will be nan
        np_array = np.append(np_array, [[np.nan]] * len(current_dates), axis=1)
        # use the target_code_idx to select the column
        np_array = np.take(np_array, target_code_idx, axis=1)

        # align date
        date_idx_map = {date: idx for idx, date in enumerate(current_dates)}
        target_np_array = np.full((len(target_dates), len(target_codes)), np.nan)

        # iterate through the target_dates and set new_row as the value of that date
        # if the date is in current_dates, use the value in current_dates to update the new_row
        new_row = np.full(len(target_codes), np.nan)
        for i, target_date in enumerate(target_dates):
            if target_date in date_idx_map:
                np_array_row = np_array[date_idx_map[target_date]]
                # use np_array_row to update row
                new_row = np.array([np_array_row[i] if ~np.isnan(np_array_row[i]) else new_row[i] for i in range(len(new_row))])

            target_np_array[i] = new_row

        self.aligned_np_array_column[column] = target_np_array


    def get_meta_data(self, codes: List[str]):
        if self.meta_data == {}:
            self.load_meta_data()

        if type(codes) == str:
            codes = [codes]

        return [self.meta_data[code] for code in codes]