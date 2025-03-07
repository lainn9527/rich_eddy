import csv
import json
import os
from datetime import date
from pathlib import Path

import requests


def fetch_daily_data_from_twse(trading_date: str):
    api_url = (
        f"https://www.twse.com.tw/exchangeReport/STOCK_DAY_ALL?date={trading_date}"
    )
    response_data = requests.get(api_url).json()
    if response_data["stat"] != "OK":
        raise ValueError("fetch data from twse failed")

    if response_data["date"] != trading_date:
        raise ValueError(
            f"date not match, expected: {trading_date}, got: {response_data[0]['Date']}"
        )

    return response_data["data"]


def fetch_daily_data_from_tpex(trading_date: str):
    api_url = f"https://www.tpex.org.tw/openapi/v1/tpex_mainboard_quotes"
    response_data = requests.get(api_url).json()
    if response_data[0]["Date"][-4:] != trading_date[-4:]:
        raise ValueError(
            f"date not match, expected: {trading_date}, got: {response_data[0]['Date']}"
        )

    return response_data


# append daily price data from twse to current data set
def append_price_data_from_twse(trading_date: str = date.today().strftime("%Y%m%d")):
    stock_data = fetch_daily_data_from_twse(trading_date)

    written_codes = []
    for data in stock_data:
        code = data[0]
        open_price, high_price, low_price, close_price, volume, total_money = map(
            lambda x: float(x.replace(",", "")),
            [data[4], data[5], data[6], data[7], data[2], data[3]],
        )
        volume = volume / 1000
        total_money = total_money / 1000

        file_path = Path(f"data/daily_price/code_from_2020/{code}.csv")
        with open(file_path, "r") as fp:
            last_record = list(csv.reader(fp))[-1]
            latest_trading_date, total_stocks = last_record[1], last_record[-1]
            if latest_trading_date == trading_date:
                print(f"{code} already has data on {trading_date}")
                continue

        with open(file_path, "a") as fp:
            writer = csv.writer(fp)
            writer.writerow(
                [
                    code,
                    trading_date,
                    open_price,
                    high_price,
                    low_price,
                    close_price,
                    volume,
                    total_money,
                    total_stocks,
                ]
            )
            written_codes.append(code)

    print(f"append data to {len(written_codes)} stocks out of {len(stock_data)}")


def append_price_data_from_tpex(trading_date: str = date.today().strftime("%Y%m%d")):
    stock_data = fetch_daily_data_from_tpex(trading_date)

    original_dir = Path("data/daily_price/code_from_2020")
    written_codes = []
    for data in stock_data:
        try:
            code = data["SecuritiesCompanyCode"]
            trading_year = str(int(data["Date"][:3]) + 1911)
            trading_date = f"{trading_year}{data['Date'][3:]}"
            open_price = float(data["Open"])
            high_price = float(data["High"])
            low_price = float(data["Low"])
            close_price = float(data["Close"])
            volume = float(data["TradingShares"]) / 1000
            total_money = float(data["TransactionAmount"]) / 1000
        except:
            continue

        file_path = Path(f"{original_dir}/{code}.csv")
        with open(file_path, "r") as fp:
            last_record = list(csv.reader(fp))[-1]
            latest_trading_date, total_stocks = last_record[1], last_record[-1]
            if latest_trading_date == trading_date:
                continue

        with open(file_path, "a") as fp:
            writer = csv.writer(fp)
            writer.writerow(
                [
                    code,
                    trading_date,
                    open_price,
                    high_price,
                    low_price,
                    close_price,
                    volume,
                    total_money,
                    total_stocks,
                ]
            )
            written_codes.append(code)
    print(f"append data to {len(written_codes)} stocks out of {len(stock_data)}")


if __name__ == "__main__":
    append_price_data_from_twse()
    append_price_data_from_tpex()
