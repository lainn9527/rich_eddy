import csv
import gc
import os

from datetime import datetime
from pathlib import Path
from typing import List

from .tej_data_processor import TejDataProcessor

class TejDailyDataProcessor(TejDataProcessor):
    def extract_code_and_date_from_line(line: List[str]):
        code_with_name, trading_date = line[0], datetime.strptime(line[1], "%Y%m%d").date()
        code, name = code_with_name.split(" ")[0], code_with_name.split(" ")[1]
        return code, name, trading_date, line[2:]

    def is_valid_stock(code: str):
        return (
            code.isdecimal()
            and int(code) >= 1000
            and int(code) <= 9999
            and len(code) == 4
        )


    # def transform_raw_data_to_date_data(
    #     data_category: str,
    #     raw_data_dir: Path,
    #     dest_data_dir: Path
    # ):
    #     """
    #     1. read tej raw data
    #     2. process null data
    #     3. write data with full columns
    #     4. write data with useful columns
    #     """
    #     year_to_daily_info = dict()
    #     raw_data_dir = raw_data_dir
    #     file_names = os.listdir(raw_data_dir)
    #     raw_column_names = TejDailyDataProcessor.get_column_names(data_category)

    #     # read tej raw data
    #     for file_name in file_names:
    #         file_path = raw_data_dir / file_name
    #         with open(file_path, "r") as fp:
    #             lines = TejDailyDataProcessor.remove_null_token(fp.readlines(), ["\x00", "-"])
    #             lines = list(csv.reader(lines))
    #             for line in lines:
    #                 # first field need to be code
    #                 code, trading_date, remaining_line = TejDailyDataProcessor.extract_code_and_date_from_line(line)
    #                 year = trading_date.year
    #                 date_str = trading_date.isoformat()
    #                 if year_to_daily_info.get(year) == None:
    #                     year_to_daily_info[year] = dict()

    #                 if year_to_daily_info.get(year).get(date_str) == None:
    #                     year_to_daily_info[year][date_str] = []
    #                 year_to_daily_info[year][date_str].append([code, date_str] + remaining_line)
    #     print("finish reading all lines")
    #     gc.collect()

    #     # write data with full columns
    #     full_data_dir = Path(dest_data_dir / "date_full")
    #     for year, daily_info_dict in year_to_daily_info.items():
    #         year_dir_path = full_data_dir / str(year)
    #         if not year_dir_path.exists():
    #             year_dir_path.mkdir(parents=True, exist_ok=True)

    #         for trading_date, daily_info in daily_info_dict.items():
    #             # sort by code
    #             daily_info.sort(key=lambda d: d[0])
    #             # append column names
    #             daily_info.insert(0, raw_column_names)
    #             # write to file
    #             with open(f"{year_dir_path}/{trading_date}.csv", "w") as fp:
    #                 csv.writer(fp).writerows(daily_info)
    #     print("finish writing full date data")

    #     # write picked column
    #     picked_data_dir = Path(dest_data_dir / "date")
    #     for year, daily_info_dict in year_to_daily_info.items():
    #         year_dir_path = picked_data_dir / str(year)
    #         if not year_dir_path.exists():
    #             year_dir_path.mkdir(parents=True, exist_ok=True)

    #         for trading_date, daily_info in daily_info_dict.items():
    #             columns = daily_info[0]
    #             valid_stock_lines = [line for line in daily_info[1:] if TejDailyDataProcessor.is_valid_stock(line[0])]
    #             valid_stock_info = TejDailyDataProcessor.pick_columns([columns] + valid_stock_lines)

    #             with open(f"{year_dir_path}/{trading_date}.csv", "w") as fp:
    #                 csv.writer(fp).writerows(valid_stock_info)
    #     print("finish writing useful date data")