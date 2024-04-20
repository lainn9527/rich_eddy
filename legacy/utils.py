import copy
import csv
import json
import os
from datetime import datetime, timedelta
from enum import Enum, IntEnum
from functools import wraps
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from pyinstrument import Profiler


class TimePeriod(IntEnum):
    WEEK = 5
    HALF_MONTH = 10
    MONTH = 20
    QUARTER = 60
    YEAR = 250


class PriceType(Enum):
    ADJUSTED = "adjusted"
    NON_ADJUSTED = "non_adjusted"


declare_report_data = {1: "0515", 2: "0814", 3: "1114", 4: "0331"}


def read_tradable_stock():
    with open("tradable_stock/120_100_569.txt", "r") as file:
        tradable_stock_codes = file.read().splitlines()[0].split(",")
    valid_stock = list(code_to_info.keys())
    filtered_stock = sorted(
        list(set(valid_stock).intersection(set(tradable_stock_codes)))
    )
    return filtered_stock


def read_all_stock_codes(from_year=None):
    if from_year != None:
        data_dir = Path(f"data/daily_price/code_from_{from_year}")
    else:
        data_dir = Path("data/daily_price/code")
    file_names = os.listdir(data_dir)

    # filter out bonds(可轉債), special stocks(特別股), ETF(ETF)
    all_codes = [file_name.split(".")[0] for file_name in file_names]
    codes = list(
        filter(
            lambda code: code.isdecimal() and int(code) >= 1000 and int(code) <= 9999,
            all_codes,
        )
    )

    return codes


def read_daily_data_by_codes(codes: List[str], data_category: str, from_year: int=None):
    if type(codes) == str:
        codes = [codes]
    if from_year != None:
        data_dir = Path(f"data/{data_category}/code_from_{from_year}")
    else:
        data_dir = Path(f"data/{data_category}/code")
    code_to_pv = dict()
    for code in codes:
        if code not in code_to_info:
            code_to_info[code] = dict()

        if data_category not in code_to_info[code]:
            file_path = data_dir / f"{code}.csv"
            if not file_path.exists():
                print(f"file not found: {file_path}")
            with open(file_path) as fp:
                pv = list(csv.reader(replace_null_with_dash(fp)))
                # handle '-' as nan
                # pv = list(map(lambda row: list(map(lambda x: np.nan if x == '-' or x == '\x00' else x, row)), pv))
                code_to_info[code][data_category] = pv
            with open(file_path, "w") as fp:
                writer = csv.writer(fp)
                writer.writerows(pv)

        code_to_pv[code] = copy.deepcopy(code_to_info[code][data_category])
    return code_to_pv


def read_daily_data(data_category: str, from_year: int, all_stock: bool = True):
    data_dir = Path(f"data/{data_category}/code_from_{from_year}")

    # with all stock
    if all_stock:
        codes = read_all_stock_codes(from_year)
    else:
        # with tradable stock
        codes = read_tradable_stock()

    return read_daily_data_by_codes(codes, data_category, from_year)


def read_eps_data():
    with open("data/finance/eps/eps.json", "r") as fp:
        eps_data = json.load(fp)
    for code, eps in eps_data.items():
        if code not in code_to_info:
            code_to_info[code] = dict()
        code_to_info[code]["eps"] = eps

    return eps_data


def read_trading_record():
    """
    Return: [[stock_code, date, side, price, volume, amount]]
    """
    trading_record = []
    with open("data/trading_record/2024-03.csv", "r") as fp:
        for record in list(csv.reader(fp))[1:]:
            stock_name, date, volume, amount, price = (
                record[0],
                record[1],
                record[2],
                record[3],
                record[4],
            )
            price, volume, amount = (
                price.replace(",", ""),
                volume.replace(",", ""),
                amount.replace(",", ""),
            )
            side = "buy" if amount[0] == "-" else "sell"
            stock_code = get_stock_name_to_code(stock_name)
            date = datetime.strptime(date, "%Y/%m/%d")
            trading_record.append([stock_code, date, side, price, volume, amount])

    return trading_record


def get_column_names(data_category: str):
    column_file_path = "data/tej_col_names.json"
    with open(column_file_path) as fp:
        col_names = json.load(fp)[data_category]
    return col_names


def read_stock_meta():
    with open("data/stock_meta.csv", "r") as fp:
        lines = list(csv.reader(fp))
        for line in lines:
            if len(line) < 6:
                continue
            code, name, market_type, industry = line[0], line[1], line[4], line[5]
            code_to_info[code] = dict()
            code_to_info[code]["meta"] = f"{name} {code} {market_type} {industry}"


def get_stock_name_to_code(stock_name):
    if stock_name not in stock_name_to_code:
        with open("data/stock_meta.csv", "r") as fp:
            lines = list(csv.reader(fp))
            for line in lines:
                if len(line) < 6:
                    continue
                code, name = line[0], line[1]
                stock_name_to_code[name] = code

    return stock_name_to_code[stock_name]


def extend_quarterly_finance_data_to_daily(
    data,
    start_date=datetime(year=2010, month=1, day=1),
    end_date=datetime(year=2030, month=1, day=1),
):
    """
    Extend quarterly finance data to daily data to align with daily stock price data.
    """
    rows = []
    eps, recurring_eps = None, None
    for line, idx in zip(data, range(len(data))):
        if idx + 1 < len(data):
            next_eps, next_recurring_eps = data[idx + 1][3], data[idx + 1][4]
        else:
            next_eps, next_recurring_eps = 0, 0
        code, year, quarter, eps, recurring_eps = (
            line[0],
            line[1],
            line[2],
            line[3],
            line[4],
        )
        report_release_date, report_end_date = get_report_valid_date(year, quarter)
        if start_date > report_end_date or end_date < report_release_date:
            continue
        date_range = pd.date_range(
            max(report_release_date, start_date),
            min(report_end_date, end_date),
            freq="D",
        )

        for date in date_range:
            rows.append([code, date, eps, recurring_eps, next_eps, next_recurring_eps])
    return rows


def get_report_valid_date(year, quarter):
    """
    Given year and quarter, return the release date and the day before the next release date of the financial report.
    """

    # start of 4th quarter is next year
    year = str(int(year) + int(int(quarter) == 4))
    release_date = datetime.strptime(year + declare_report_data[int(quarter)], "%Y%m%d")
    next_quarter = int(quarter) % 4 + 1

    # end of 3th and 4th quarter is next year
    end_year = str(int(year) + int(int(quarter) == 3))
    end_date = datetime.strptime(
        end_year + declare_report_data[int(next_quarter)], "%Y%m%d"
    ) - timedelta(days=1)

    return release_date, end_date


def get_stock_meta(code: str):
    result = code
    try:
        result = code_to_info[code]["meta"]
    except:
        print(f"code {code} not found in code_to_info")
    return result


def get_daily_price_dataframe(code: str, from_year: int=None):
    data = read_daily_data_by_codes(code, "daily_price", from_year)
    df = pd.DataFrame(
        data[code][1:],
        columns=[
            "code",
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "trading_value",
            "total_stocks",
        ],
    ).replace(
        '', np.nan
    ).astype(
        {
            "code": "str",
            "date": "datetime64[ns]",
            "open": "float64",
            "high": "float64",
            "low": "float64",
            "close": "float64",
            "volume": "float64",
            "trading_value": "float64",
            "total_stocks": "int",
        }
    )
    df["date"] = pd.to_datetime(df["date"])
    return df


def get_eps_dataframe(code: str, daily=True):
    if code not in code_to_info:
        print(f"code {code} not found in code_to_info")

    if "eps" not in code_to_info[code]:
        read_eps_data()
    eps_data = code_to_info[code]["eps"][1:]  # skip column names
    if daily:
        eps_data = extend_quarterly_finance_data_to_daily(eps_data)

    return pd.DataFrame(
        eps_data,
        columns=[
            "code",
            "date",
            "eps",
            "recurring_eps",
            "next_eps",
            "next_recurring_eps",
        ],
    ).astype(
        {
            "code": "str",
            "date": "datetime64[ns]",
            "eps": "float64",
            "recurring_eps": "float64",
            "next_eps": "float64",
            "next_recurring_eps": "float64",
        }
    )


def get_trading_record_dataframe():
    trading_record = read_trading_record()
    return pd.DataFrame(
        trading_record, columns=["code", "date", "side", "price", "volume", "amount"]
    ).astype(
        {
            "code": "str",
            "date": "datetime64[ns]",
            "side": "str",
            "price": "float64",
            "volume": "float64",
            "amount": "float64",
        }
    )


def get_tw50_index_df():
    df = pd.read_csv("data/market_index/tw50_index.csv", header=0, parse_dates=[0])
    df.columns = ["date", "open", "high", "low", "close", "adj_close", "volume"]
    return df


def replace_null_with_dash(string):
    if type(string) == str:
        string = [string]
    return list(map(lambda x: x.replace("\x00", ""), string))


def time_profiler(func):
    # only for local development
    @wraps(func)
    def wrapper(*args, **kwargs):
        profiler = Profiler()
        profiler.start()

        result = func(*args, **kwargs)
        profiler.stop()
        print(profiler.output_text(unicode=True, color=True))

        return result

    return wrapper


code_to_info = dict()
stock_name_to_code = dict()
read_stock_meta()
