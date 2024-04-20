import gzip
import pickle
from enum import IntEnum

import numpy as np
import plotly.graph_objects as go
import talib

from utils import TimePeriod, sma


def strategy(
    datetime,
    open,
    high,
    low,
    close,
    week_ma,
    half_month_ma,
    month_ma,
    quarter_ma,
    half_year_ma,
    year_ma,
):
    signal, signal_datetime = [False, False, False], []
    diff_ratio = 0.01
    for i in range(3, len(close)):
        week_half_month_diff = (week_ma[i] - half_month_ma[i]) / half_month_ma[i]
        half_month_month_diff = (half_month_ma[i] - month_ma[i]) / month_ma[i]
        if (
            np.abs(week_half_month_diff) < diff_ratio
            and np.abs(half_month_month_diff) < diff_ratio
            and close[i] > week_ma[i]
            and close[i] > half_month_ma[i]
            and close[i] > month_ma[i]
            and close[i - 1] > week_ma[i - 1]
            and close[i - 1] > half_year_ma[i - 1]
            and close[i - 1] > month_ma[i - 1]
            and low[i - 2] < week_ma[i - 2]
            and low[i - 2] < half_month_ma[i - 2]
            and low[i - 2] < month_ma[i - 2]
            and close[i] > open[i]
        ):
            signal.append(True)
            signal_datetime.append(datetime[i])
        else:
            signal.append(False)
    return signal, signal_datetime


if __name__ == "__main__":
    stock_code = "2376"
    with gzip.open("s_code_to_info.gz", "rb") as fp:
        all_data: dict = pickle.load(fp)

    data = all_data[stock_code]

    datetime = np.array([d[0] for d in data], dtype="datetime64")
    open_prices, high_prices, low_prices, close_prices, volume = np.array(
        [d[1:6] for d in data], dtype=float
    ).transpose()

    week_sma, half_month_sma, month_sma, quarter_sma, half_year_sma, year_sma = sma(
        close_prices,
        [
            TimePeriod.WEEK,
            TimePeriod.HALF_MONTH,
            TimePeriod.MONTH,
            TimePeriod.QUARTER,
            TimePeriod.QUARTER * 2,
            TimePeriod.YEAR,
        ],
    )

    signals, signals_datetime = strategy(
        datetime,
        open_prices,
        high_prices,
        low_prices,
        close_prices,
        week_sma,
        half_month_sma,
        month_sma,
        quarter_sma,
        half_month_sma,
        year_sma,
    )

    for i in range(0, len(signals)):
        if signals[i]:
            print(datetime[i])
            print("\n")
