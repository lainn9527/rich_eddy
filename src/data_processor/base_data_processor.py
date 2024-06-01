import csv
import os

from pathlib import Path
from typing import List
from ..utils.common import DataColumn

class BaseDataProcessor:
    @classmethod
    def transform_raw_data_to_date_data(
        cls,
        data_category: str,
        raw_data_dir: Path,
        dest_data_dir: Path
    ):
        raise NotImplementedError
        
    @classmethod
    def transform_date_data_to_code_data(
        cls,
        date_data_dir: Path,
        dest_data_dir: Path,
        start_year: int = None,
        with_column: bool = True,
    ):
        year_dirs = os.listdir(date_data_dir)
        if start_year != None:
            year_dirs = list(filter(lambda x: int(x) >= start_year, year_dirs))

        # read date data and construct code data
        code_info_dict = dict()
        for year_dir in year_dirs:
            year = year_dir
            year_dir_path = date_data_dir / year_dir
            file_names = os.listdir(year_dir_path)
            for file_name in file_names:
                file_path = f"{year_dir_path}/{file_name}"
                with open(file_path) as fp:
                    lines = list(csv.reader(fp))
                    column_names = lines[0]
                    for line in lines[1:]:
                        code = line[0]
                        if code_info_dict.get(code) == None:
                            code_info_dict[code] = []
                        code_info_dict[code].append(line)

            print(f"finish year {year}")

        # write code data
        if not dest_data_dir.exists():
            dest_data_dir.mkdir(parents=True, exist_ok=True)

        for code, daily_info in code_info_dict.items():
            sorted_daily_info = sorted(daily_info, key=lambda d: d[1])
            if with_column:
                sorted_daily_info = [column_names] + sorted_daily_info
            with open(dest_data_dir / f"{code}.csv", "w") as fp:
                csv.writer(fp).writerows(sorted_daily_info)

    @classmethod
    def pick_columns(cls, lines: List[str], picked_columns: List[DataColumn], column_mapper: dict):
        """
        Input: lines (with column), picked_columns, column_mapper
        Output: processed_lines (with column)
        """

        picked_columns = list(map(lambda x: x.value, picked_columns))
        origin_column_names = lines[0]
        picked_source_columns = list(map(lambda x: column_mapper[x], picked_columns))
        picked_source_columns_idx = list(map(lambda x: origin_column_names.index(x), picked_source_columns))
        processed_lines = [picked_columns]
        for line in lines[1:]:
            processed_lines.append([line[idx] for idx in picked_source_columns_idx])

        return processed_lines

    @classmethod
    def remove_null_token(cls, lines: List[str], null_tokens: List[str]) -> List[str]:
        if type(lines) == str:
            lines = [lines]

        if type(null_tokens) == str:
            null_tokens = [null_tokens]

        for idx, line in enumerate(lines):
            for token in null_tokens:
                line = line.replace(token, "")
            lines[idx] = line

        return lines
