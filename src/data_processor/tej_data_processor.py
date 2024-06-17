import csv
import gc
import json
import os
from pathlib import Path
from typing import List

from ..utils.common import DataCategoryColumn, DataCategory
from .base_data_processor import BaseDataProcessor

class TejDataProcessor(BaseDataProcessor):
    column_file_path = "data/tej_col_names.json"

    @classmethod
    def transform_raw_data_to_date_data(
        cls,
        data_category: DataCategory,
        raw_data_dir: Path,
        dest_data_dir: Path,
        encoding: str = None,
    ):
        """
        1. read tej raw data
        2. process null data
        3. write data with full columns
        4. write data with useful columns
        """
        year_to_daily_info = dict()
        raw_data_dir = raw_data_dir
        file_names = os.listdir(raw_data_dir)
        raw_column_names = cls.get_column_names(data_category)

        # read tej raw data
        for file_name in file_names:
            file_path = raw_data_dir / file_name
            with open(file_path, "r", encoding=encoding) as fp:
                lines = cls.remove_null_token(fp.readlines(), ["\x00", "-"])
                lines = list(csv.reader(lines))
                for line in lines[1:]:
                    # first field need to be code
                    code, name, trading_date, remaining_line = cls.extract_code_and_date_from_line(line)
                    if not cls.is_valid_stock(code):
                        continue

                    year = trading_date.year
                    date_str = trading_date.isoformat()
                    if year_to_daily_info.get(year) == None:
                        year_to_daily_info[year] = dict()

                    if year_to_daily_info.get(year).get(date_str) == None:
                        year_to_daily_info[year][date_str] = []

                    if name != None:
                        year_to_daily_info[year][date_str].append([code, name, trading_date.strftime("%Y%m%d")] + remaining_line)
                    else:
                        year_to_daily_info[year][date_str].append([code, trading_date.strftime("%Y%m%d")] + remaining_line)

        print("finish reading all lines")
        gc.collect()

        # write data with full columns
        full_data_dir = Path(dest_data_dir / "date_full")
        for year, daily_info_dict in year_to_daily_info.items():
            year_dir_path = full_data_dir / str(year)
            if not year_dir_path.exists():
                year_dir_path.mkdir(parents=True, exist_ok=True)

            for trading_date, daily_info in daily_info_dict.items():
                # sort by code
                daily_info.sort(key=lambda d: d[0])
                # append column names
                daily_info.insert(0, raw_column_names)
                # write to file
                with open(f"{year_dir_path}/{trading_date}.csv", "w") as fp:
                    csv.writer(fp).writerows(daily_info)
        print("finish writing full date data")

        # write picked column
        picked_data_dir = Path(dest_data_dir / "date")
        column_mapper = cls.tej_column_name_mapper[data_category.value]
        for year, daily_info_dict in year_to_daily_info.items():
            year_dir_path = picked_data_dir / str(year)
            if not year_dir_path.exists():
                year_dir_path.mkdir(parents=True, exist_ok=True)

            for trading_date, daily_info in daily_info_dict.items():
                daily_info = cls.pick_columns(daily_info, DataCategoryColumn.get_columns(data_category), column_mapper)

                with open(f"{year_dir_path}/{trading_date}.csv", "w") as fp:
                    csv.writer(fp).writerows(daily_info)
        print("finish writing useful date data")


    def extract_code_and_date_from_line(line: List[str]):
        raise NotImplementedError

    @classmethod
    def transform_useful_columns_and_stock(cls, date_data_dir: Path, dest_data_dir: Path, data_category: str):
        year_dirs = os.listdir(date_data_dir)
        for year_dir in year_dirs:
            year_dir_path = date_data_dir / year_dir
            new_year_dir_path = dest_data_dir / year_dir
            if not new_year_dir_path.exists():
                new_year_dir_path.mkdir(parents=True, exist_ok=True)

            file_names = os.listdir(year_dir_path)
            for file_name in file_names:
                file_path = f"{year_dir_path}/{file_name}"
                with open(file_path, "r") as fp:
                    lines = list(csv.reader(fp))
                    valid_stock_info = [lines[0]]
                    for line in lines[1:]:
                        code = line[0]
                        if TejDataProcessor.is_valid_stock(code):
                            valid_stock_info.append(line)

                    processed_lines = TejDataProcessor.pick_columns(
                        valid_stock_info,
                        DataCategoryColumn.get_columns(data_category),
                        cls.tej_column_name_mapper[data_category.value]
                    )

                new_file_path = f"{new_year_dir_path}/{file_name}"
                with open(new_file_path, "w") as fp:
                    writer = csv.writer(fp)
                    writer.writerows(processed_lines)


    def get_column_names(data_category: DataCategory):
        with open(TejDataProcessor.column_file_path) as fp:
            col_names = json.load(fp)[data_category.value]
        return col_names


    def filter_valid_stock(codes: List[str]):
        codes = list(
            filter(lambda code: TejDataProcessor.is_valid_stock(code), codes)
        )
        return codes


    def is_valid_stock(code: str):
        return True


    tej_column_name_mapper = {
        "daily_price": {
            "code": "股票代碼",
            "date": "日期",
            "open": "開盤價(元)",
            "high": "最高價(元)",
            "low": "最低價(元)",
            "close": "收盤價(元)",
            "volume": "成交量(千股)",
            "trading_value": "成交值(千元)",
            "total_stocks": "流通在外股數(千股)",
            "market_value": "市值(百萬元)",
        },
        "finance_report": {
            "code": "股票代碼",
            "date": "年月",
            "quarter": "季別",
            "eps": "每股盈餘",
            "recurring_eps": "常續性EPS",
        },
        "chip": {
            "code": "股票代碼",
            "date": "日期",

            # 法人資料
            "foreign_buy_volume": "外資買超(張)",
            "foreign_sell_volume": "外資賣超(張)",
            "foreign_net_volume": "外資買賣超(張)",
            "foreign_buy_amount": "外資買進金額(百萬)",
            "foreign_sell_amount": "外資賣出金額(百萬)",
            "local_investor_buy_volume": "投信買超(張)",
            "local_investor_sell_volume": "投信賣超(張)",
            "local_investor_net_volume": "投信買賣超(張)",
            "local_investor_buy_amount": "投信買進金額(百萬)",
            "local_investor_sell_amount": "投信賣出金額(百萬)",
            "local_self_buy_volume": "自營買超(張)",
            "local_self_sell_volume": "自營賣超(張)",
            "local_self_net_volume": "自營買賣超(張)",
            "local_self_buy_amount": "自營商買進金額(百萬)",
            "local_self_sell_amount": "自營商賣出金額(百萬)",
            "total_investor_buy_volume": "三大法人買超(張)",
            "total_investor_sell_volume": "三大法人賣超(張)",
            "total_investor_net_volume": "合計買賣超(張)",
            "foreign_trading_ratio": "外資成交比重",
            "local_investor_trading_ratio": "投信成交比重",
            "local_self_trading_ratio": "自營成交比重",
            "total_investor_trading_ratio": "法人成交比重",
            "foreign_total_holdings_ratio": "外資總持股率_不含董監%",
            "local_investor_holdings_ratio": "投信持股率％",
            "local_self_holdings_ratio": "自營持股率％",
            "foreign_total_holdings": "外資總持股數",
            "local_investor_holdings": "投信持股數(張)",
            "local_self_holdings": "自營持股數",
            "director_supervisor_holdings_ratio": "董監持股％",
            "director_supervisor_pledge_ratio": "董監質押％",
            "director_supervisor_holdings": "董監持股數",

            # 信用交易
            "margin_buy_volume": "融資增加(張)",
            "margin_sell_volume": "融資減少(張)",
            "short_sell_buy_volume": "融券增加(張)",
            "short_sell_sell_volume": "融券減少(張)",
            "credit_trading_ratio": "信用交易比重",
            "spot_trading_ratio": "一般現股成交比重",
            "offset_volume": "資券互抵(張)",
            "offset_ratio": "資券互抵比例",
            "turnover_rate": "實際週轉率％",
            "margin_balance": "融資餘額(張)",
            "margin_balance_amount": "融資餘額(千元)",
            "margin_usage_ratio": "融資使用率",
            "short_sell_balance": "融券餘額(張)",
            "short_sell_balance_amount": "融券餘額(千元)",
            "short_sell_usage_ratio": "融券使用率",
            "securities_to_capital_ratio": "券資比",
            "total_stocks": "流通在外股數(百萬股)"
        },
    }
