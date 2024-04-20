import csv
import datetime
import math
from typing import Dict, List

import numpy as np
import pandas as pd
import pandas_ta as ta
import talib

from .utils import (
    TimePeriod,
    get_daily_price_dataframe,
    get_stock_meta,
    get_tw50_index_df,
    read_daily_data,
    read_daily_data_by_codes,
    read_eps_data,
)


def filter_by_layers(filters: List[callable], codes: List[str], config: dict):
    for filter in filters:
        filter_name = filter.__name__
        prev_lens = len(codes)
        codes = filter(codes, config.get(filter_name))
        print(
            f"{filter_name}: {(1 - len(codes) / prev_lens) * 100:.2f}%, remaining: {len(codes)}"
        )

    return codes


def signal_summary(data):
    len_list = [len(v) for v in data.values()]
    sorted_len_list = sorted(len_list)
    print(
        f"min: {sorted_len_list[0]}, max: {sorted_len_list[-1]}, avg: {sum(len_list)/len(len_list)}"
    )
    print(
        f"p99: {sorted_len_list[int(len(sorted_len_list)*0.99)]}, \
p95: {sorted_len_list[int(len(sorted_len_list)*0.95)]}, \
p90: {sorted_len_list[int(len(sorted_len_list)*0.90)]}, \
p75: {sorted_len_list[int(len(sorted_len_list)*0.75)]}"
    )


def data_summary(name, data):
    sorted_data = sorted(data)
    print(f"Summary for {name}")
    print(
        f"len: {len(data)}, min: {sorted_data[0]:.2f}, max: {sorted_data[-1]:.2f}, avg: {float(sum(sorted_data)/len(sorted_data)):.2f}, sum: {float(sum(sorted_data)):.2f}"
    )
    print(
        f"p99: {sorted_data[int(len(sorted_data)*0.99)]:.2f}, \
p95: {sorted_data[int(len(sorted_data)*0.95)]:.2f}, \
p90: {sorted_data[int(len(sorted_data)*0.90)]:.2f}, \
p75: {sorted_data[int(len(sorted_data)*0.75)]:.2f}"
    )


# check if the avg volume of for last n day is greater than volume_threshold
def filter_stock_by_volume(volume_threshold, n):
    code_to_pv = read_daily_data("daily_price")
    valid_stock_codes = []
    for stock_code, data_list in code_to_pv.items():
        latest_date = datetime.date.fromisoformat(data_list[-1][1])
        week_ago = datetime.date.today() - datetime.timedelta(days=7)
        if latest_date < week_ago or len(data_list) < n:
            continue

        avg_volume = 0
        for i in range(1, n):
            avg_volume += float(data_list[-i][5])

        avg_volume /= n
        if avg_volume > volume_threshold:
            valid_stock_codes.append(stock_code)

    with open(
        f"tradable_stock/{n}_{volume_threshold}_{len(valid_stock_codes)}.txt", "w"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(valid_stock_codes)


# extract close price with given amplitude within given time window
def within_amplitude_with_time_window_and_high_end(
    time_window: int, amplitude: float, all_stock: bool = False
):
    code_to_pv = read_daily_data("daily_price", all_stock)
    code_to_signal = dict()
    date_to_signal = dict()
    for code, pv in code_to_pv.items():
        # transpose data
        pv = np.array(pv[1:]).transpose()
        date, close = pv[1].astype(np.datetime64), pv[5].astype(np.float64)

        i = time_window
        while i < len(close):
            date_slice, close_slice = (
                date[i - time_window : i],
                close[i - time_window : i],
            )
            max_close_idx, min_close_idx = close_slice.argmax(), close_slice.argmin()
            max_close, min_close = (
                close_slice[max_close_idx],
                close_slice[min_close_idx],
            )

            if (
                max_close == close_slice[-1]
                and max_close - min_close <= min_close * amplitude
            ):
                if code_to_signal.get(code) == None:
                    code_to_signal[code] = []
                if date_to_signal.get(date_slice[-1]) == None:
                    date_to_signal[date_slice[-1]] = []
                code_to_signal[code].append(
                    f"{code},{date_slice[-1]},{close_slice[-1]},{date_slice[min_close_idx]},{min_close}"
                )
                date_to_signal[date_slice[-1]].append(
                    f"{code},{date_slice[-1]},{close_slice[-1]},{date_slice[min_close_idx]},{min_close}"
                )
                i += 20
            else:
                i += 1
        if code in code_to_signal:
            print(f"finish {code}, got {len(code_to_signal.get(code))} signals")
        else:
            print(f"finish {code}, got 0 signals")
    # signal_summary(code_to_signal)
    # signal_summary(date_to_signal)
    # write data
    with open(f"code_signal_{time_window}_{amplitude}.csv", "w") as fp:
        for v in code_to_signal.values():
            fp.write("\n".join(v))
            fp.write("\n")


def within_amplitude_trading(
    codes: List[str],
    time_window: int,
    amplitude: float,
    start_date: datetime.datetime,
    end_date: datetime.datetime,
    write_file: bool = True,
):
    """
    Extract close price with given amplitude within given time window.
    尋找壓縮後突破
    1. 過去一段時間內高低價再一定區間內
    2. 當日價格突破
    """

    code_to_pv = read_daily_data_by_codes(codes, "daily_price")
    code_to_signal = dict()
    date_to_signal = dict()
    signal_codes = set()
    before_days = (end_date - start_date).days
    for code, pv in code_to_pv.items():
        # transpose data
        pv = np.array(pv[1:]).transpose()
        date, close = pv[1].astype(np.datetime64), pv[5].astype(np.float64)

        if date[-1].__str__() != end_date.strftime("%Y%m%d"):
            continue

        i = len(close) - before_days
        while i <= len(close):
            date_slice, close_slice = (
                date[i - time_window : i],
                close[i - time_window : i],
            )
            if len(date_slice) < time_window:
                break

            max_close_idx, min_close_idx = close_slice.argmax(), close_slice.argmin()
            max_close, min_close = (
                close_slice[max_close_idx],
                close_slice[min_close_idx],
            )

            if (
                max_close == close_slice[-1]
                and max_close - min_close <= min_close * amplitude
            ):
                if code_to_signal.get(code) == None:
                    code_to_signal[code] = []
                if date_to_signal.get(date_slice[-1]) == None:
                    date_to_signal[date_slice[-1]] = []
                code_to_signal[code].append(
                    f"{code},{date_slice[-1]},{close_slice[-1]},{date_slice[min_close_idx]},{min_close}"
                )
                date_to_signal[date_slice[-1]].append(
                    f"{code},{date_slice[-1]},{close_slice[-1]},{date_slice[min_close_idx]},{min_close}"
                )
                signal_codes.add(code)
            i += 1

    # write data
    if write_file:
        with open(
            f"trading_code_signal_{time_window}_{amplitude}_{end_date}.csv", "w"
        ) as fp:
            fp.write("code,date,close,min_date,min_close\n")
            for v in code_to_signal.values():
                fp.write("\n".join(v))
                fp.write("\n")
    return list(signal_codes)


# extract close price with given amplitude within given time window, and end at high price
def extract_amplitude_with_time_window_and_high_end(
    time_window: int, amplitude: float, all_stock: bool = False
):
    code_to_pv = read_daily_data("daily_price", all_stock)
    code_to_signal = dict()
    for code, pv in code_to_pv.items():
        # transpose data
        pv = np.array(pv[1:]).transpose()
        date, close = pv[1].astype(np.datetime64), pv[5].astype(np.float64)

        for i in range(time_window, len(close)):
            date_slice, close_slice = (
                date[i - time_window : i],
                close[i - time_window : i],
            )
            max_close_idx, min_close_idx = close_slice.argmax(), close_slice.argmin()
            max_close, min_close = (
                close_slice[max_close_idx],
                close_slice[min_close_idx],
            )

            if (
                max_close == close_slice[-1]
                and max_close - min_close > min_close * amplitude
            ):
                if code_to_signal.get(code) == None:
                    code_to_signal[code] = []
                code_to_signal[code].append(
                    f"{code},{date_slice[-1]},{close_slice[-1]},{date_slice[min_close_idx]},{min_close}"
                )
        if code in code_to_signal:
            print(f"finish {code}, got {len(code_to_signal.get(code))} signals")
        else:
            print(f"finish {code}, got 0 signals")
    # write data
    with open(f"code_signal_{time_window}_{amplitude}.csv", "w") as fp:
        for v in code_to_signal.values():
            fp.write("\n".join(v))
            fp.write("\n")


# find stock with max close surpassing min close in x% within given time window
def surpass_amplitude_within_time_window(
    time_window: int, amplitude: float, all_stock: bool = False
):
    code_to_pv = read_daily_data("daily_price", all_stock)
    code_to_signal = dict()
    date_to_signal = dict()
    for code, pv in code_to_pv.items():
        # transpose data
        pv = np.array(pv[1:]).transpose()
        date, close = pv[1].astype(np.datetime64), pv[5].astype(np.float64)

        i = time_window
        while i < len(close):
            date_slice, close_slice = (
                date[i - time_window : i],
                close[i - time_window : i],
            )
            max_close_idx, min_close_idx = close_slice.argmax(), close_slice.argmin()
            max_close, min_close = (
                close_slice[max_close_idx],
                close_slice[min_close_idx],
            )

            if max_close > min_close * amplitude:
                next_idx = min(i + 120, len(close))
                next_date_slice, next_close_slice = (
                    date[i - time_window : next_idx],
                    close[i - time_window : next_idx],
                )
                next_max_close_idx = next_close_slice.argmax()
                next_max_close = next_close_slice[next_max_close_idx]
                if next_max_close > max_close:
                    max_close_idx, max_close, date_slice, close_slice = (
                        next_max_close_idx,
                        next_max_close,
                        next_date_slice,
                        next_close_slice,
                    )

                if code_to_signal.get(code) == None:
                    code_to_signal[code] = []
                if date_to_signal.get(date_slice[-1]) == None:
                    date_to_signal[date_slice[-1]] = []
                code_to_signal[code].append(
                    f"{code},{date_slice[max_close_idx]},{max_close},{date_slice[min_close_idx]},{min_close}"
                )
                date_to_signal[date_slice[-1]].append(
                    f"{code},{date_slice[max_close_idx]},{max_close},{date_slice[min_close_idx]},{min_close}"
                )

                i += max_close_idx + 120
            else:
                i += time_window
        if code in code_to_signal:
            print(f"finish {code}, got {len(code_to_signal.get(code))} signals")
        else:
            print(f"finish {code}, got 0 signals")
    # signal_summary(code_to_signal)
    # signal_summary(date_to_signal)
    # write data
    with open(f"code_signal_{time_window}_{amplitude}.csv", "w") as fp:
        for v in code_to_signal.values():
            fp.write("\n".join(v))
            fp.write("\n")


# find stock with recent eps surpassing previous N eps in X%
# e.g. [0.05, 0.1] means the 1st last eps should be 5% greater than the 2nd last eps, and the 2nd last eps should be 10% greater than the 3rd last eps
def surpass_eps(config: dict, write_file: bool = False):
    amplitudes = config["amplitudes"]

    code_to_finance = read_eps_data()
    signal_arrays = [[] for _ in range(len(amplitudes))]
    recurring_signal_arrays = [[] for _ in range(len(amplitudes))]
    both_signal_array = [[] for _ in range(len(amplitudes))]

    signal = [
        [
            "code",
            "date",
            "signal_type",
            "level",
            "eps",
            "recurring_eps",
            "eps_change_ratio",
            "recurring_eps_change_ratio",
        ]
    ]
    for code, finance in code_to_finance.items():
        # transpose data
        finance = pd.DataFrame(finance[1:], columns=finance[0])
        finance.replace("-", np.nan, inplace=True)
        finance = finance.astype(
            {
                "code": "str",
                "year": "str",
                "quarter": "int",
                "eps": "float64",
                "recurring_eps": "float64",
            }
        )
        finance["eps_change_ratio"] = finance["eps"].pct_change(fill_method=None)
        finance["recurring_eps_change_ratio"] = finance["recurring_eps"].pct_change(
            fill_method=None
        )

        if len(finance["eps"]) < len(amplitudes):
            print(f"{code} has less than {len(amplitudes)} quarters of data")
            continue

        # when amplitude is positive/negative, eps should be positive/negative
        if (
            finance["eps"].iloc[-1] * amplitudes[0] < 0
            or finance["recurring_eps"].iloc[-1] * amplitudes[0] < 0
        ):
            continue

        (
            eps_continuous,
            recurring_continuous,
        ) = [True] * 2
        for i, amplitude in enumerate(amplitudes):
            idx = -(i + 1)
            if eps_continuous and finance["eps_change_ratio"].iloc[idx] > amplitude:
                signal_arrays[i].append(code)
                signal.append(
                    [
                        code,
                        f'{finance["year"].iloc[idx]}-{finance["quarter"].iloc[idx]}',
                        "eps",
                        i + 1,
                        finance["eps"].iloc[idx],
                        finance["recurring_eps"].iloc[idx],
                        finance["eps_change_ratio"].iloc[idx],
                        finance["recurring_eps_change_ratio"].iloc[idx],
                    ]
                )
            else:
                eps_continuous = False

            if (
                recurring_continuous
                and finance["recurring_eps_change_ratio"].iloc[idx] > amplitude
            ):
                recurring_signal_arrays[i].append(code)
                signal.append(
                    [
                        code,
                        f'{finance["year"].iloc[idx]}-{finance["quarter"].iloc[idx]}',
                        "recurring_eps",
                        i + 1,
                        finance["eps"].iloc[idx],
                        finance["recurring_eps"].iloc[idx],
                        finance["eps_change_ratio"].iloc[idx],
                        finance["recurring_eps_change_ratio"].iloc[idx],
                    ]
                )

                if code in signal_arrays[i]:
                    both_signal_array[i].append(code)
                    signal.append(
                        [
                            code,
                            f'{finance["year"].iloc[idx]}-{finance["quarter"].iloc[idx]}',
                            "both",
                            i + 1,
                            finance["eps"].iloc[idx],
                            finance["recurring_eps"].iloc[idx],
                            finance["eps_change_ratio"].iloc[idx],
                            finance["recurring_eps_change_ratio"].iloc[idx],
                        ]
                    )
            else:
                recurring_continuous = False

    # write data
    if write_file:
        with open(f"eps_signal_{amplitude*100}%.csv", "w") as fp:
            writer = csv.writer(fp)
            writer.writerows(signal)

    return signal_arrays, recurring_signal_arrays, both_signal_array


def corp_trade():
    code_to_pv = read_daily_data("daily_price", True)
    code_to_chip = read_daily_data("chip", True)

    for code, pv in code_to_pv.items():
        if code not in code_to_chip:
            continue
        chip = code_to_chip[code]


def find_local_max_min(codes: List[str], config: dict, remove_noise: bool = False):
    from_year = config["meta"]["from_year"]
    code_to_pv = read_daily_data_by_codes(codes, "daily_price", from_year)

    MIN_WINDOW = 1
    MAX_WINDOW = 1

    code_to_signal = dict()
    for code, pv in code_to_pv.items():
        local_signal = []
        # transpose data
        pv = np.array(pv[1:]).transpose()
        date, high, low, close = (
            pv[1].astype(np.datetime64),
            pv[3].astype(np.float64),
            pv[4].astype(np.float64),
            pv[5].astype(np.float64),
        )

        # find local minimum within time window
        i = MIN_WINDOW
        while i < len(low):
            low_slice = low[i - MIN_WINDOW : i + MIN_WINDOW + 1]
            min_low_idx = low_slice.argmin()
            if low_slice[min_low_idx] == low[i]:
                local_signal.append(
                    {
                        "type": "min",
                        "value": float(low[i]),
                        "index": i,
                        "date": datetime.datetime.strptime(date[i].__str__(), "%Y%m%d"),
                    }
                )
                i += MIN_WINDOW
            else:
                i += 1

        # find local maximum within time window
        i = MAX_WINDOW
        while i < len(high):
            high_slice = high[i - MAX_WINDOW : i + MAX_WINDOW + 1]
            max_high_idx = high_slice.argmax()
            if high_slice[max_high_idx] == high[i]:
                local_signal.append(
                    {
                        "type": "max",
                        "value": float(high[i]),
                        "index": i,
                        "date": datetime.datetime.strptime(date[i].__str__(), "%Y%m%d"),
                    }
                )
                i += MAX_WINDOW
            else:
                i += 1

        local_signal = sorted(local_signal, key=lambda x: x["index"])
        # remove the noise: combine the consecutive local maximum or local minimum
        if remove_noise:
            i = 1
            while i < len(local_signal):
                if local_signal[i]["type"] != local_signal[i - 1]["type"]:
                    i += 1
                    continue

                previous_value, current_value = (
                    local_signal[i - 1]["value"],
                    local_signal[i]["value"],
                )
                if local_signal[i]["type"] == "max":
                    if current_value >= previous_value:
                        local_signal.pop(i - 1)
                    else:
                        local_signal.pop(i)
                elif local_signal[i]["type"] == "min":
                    if current_value <= previous_value:
                        local_signal.pop(i - 1)
                    else:
                        local_signal.pop(i)

        code_to_signal[code] = local_signal

    return code_to_signal


def find_middle_max(codes: List[str], config: dict, level: int = 1):
    from_year = config["meta"]["from_year"]
    code_to_signal = find_local_max_min(codes, config)
    while level != 0:
        code_to_middle_max = {code: [] for code in codes}
        for code in codes:
            if code not in code_to_signal:
                print(f"{code} has no signal")
                continue
            signals = code_to_signal[code]
            local_min = list(filter(lambda x: x["type"] == "min", signals))
            local_max = list(filter(lambda x: x["type"] == "max", signals))
            # middle min
            i = 1
            while i < len(local_min):
                slice_min_signals = local_min[i - 1 : i + 2]
                middle_min = min(slice_min_signals, key=lambda x: x["value"])
                if local_min[i]["value"] == middle_min["value"]:
                    code_to_middle_max[code].append(middle_min)
                i += 1

            # middle max
            i = 1
            while i < len(local_max):
                slice_max_signals = local_max[i - 1 : i + 2]
                middle_max = max(slice_max_signals, key=lambda x: x["value"])
                if local_max[i]["value"] == middle_max["value"]:
                    code_to_middle_max[code].append(middle_max)
                i += 1

            code_to_middle_max[code] = sorted(
                code_to_middle_max[code], key=lambda x: x["index"]
            )
        code_to_signal = code_to_middle_max
        level -= 1
    return code_to_signal


def sma(close: np.ndarray, time_periods: List[int]):
    sma = [talib.SMA(close, timeperiod=tp) for tp in time_periods]

    return sma


def sam_filter(codes: List[str], config: dict):
    from_year = config["meta"]["from_year"]
    code_to_pv = read_daily_data_by_codes(codes, "daily_price", from_year)

    filtered_codes = []
    for code, pv in code_to_pv.items():
        # transpose data
        pv = np.array(pv[1:]).transpose()
        date, high, low, close = (
            pv[1].astype(np.datetime64),
            pv[3].astype(np.float64),
            pv[4].astype(np.float64),
            pv[5].astype(np.float64),
        )
        current_price = close[-1]
        week_sma, half_month_sma, month_sma, quarter_sma, half_year_sma, year_sma = sma(
            close,
            [
                TimePeriod.WEEK,
                TimePeriod.HALF_MONTH,
                TimePeriod.MONTH,
                TimePeriod.QUARTER,
                TimePeriod.QUARTER * 2,
                TimePeriod.YEAR,
            ],
        )
        if current_price < half_year_sma[-1] or current_price < year_sma[-1]:
            continue
        else:
            filtered_codes.append(code)
    return filtered_codes


def sma_breakthrough_alignment_filter(codes: List[str], config: dict):
    from_year = config["meta"]["from_year"]
    code_to_pv = read_daily_data_by_codes(codes, "daily_price", from_year)

    filtered_codes = []
    for code, pv in code_to_pv.items():
        # transpose data
        pv = np.array(pv[1:]).transpose()
        try:
            date, open_price, high, low, close = (
                pv[1].astype(np.datetime64),
                pv[2].astype(np.float64),
                pv[3].astype(np.float64),
                pv[4].astype(np.float64),
                pv[5].astype(np.float64),
            )
            current_price = close[-1]
            (
                week_sma,
                half_month_sma,
                month_sma,
                quarter_sma,
                half_year_sma,
                year_sma,
            ) = sma(
                close,
                [
                    TimePeriod.WEEK,
                    TimePeriod.HALF_MONTH,
                    TimePeriod.MONTH,
                    TimePeriod.QUARTER,
                    TimePeriod.QUARTER * 2,
                    TimePeriod.YEAR,
                ],
            )
        except:
            print(code)
            continue
        # 當日收盤價 > 年線, 半年線
        if current_price < half_year_sma[-1] or current_price < year_sma[-1]:
            continue

        # 週線-半月線 & 半月線-月線 < n%
        week_half_month_diff = (week_sma[-1] - half_month_sma[-1]) / half_month_sma[-1]
        half_month_month_diff = (half_month_sma[-1] - month_sma[-1]) / month_sma[-1]
        if (
            np.abs(week_half_month_diff) > config["week_half_month_diff_ratio"]
            or np.abs(half_month_month_diff) > config["half_month_month_diff_ratio"]
        ):
            continue

        # 當日收盤價 > 週線, 半月線, 月線
        if (
            current_price < week_sma[-1]
            or current_price < half_month_sma[-1]
            or current_price < month_sma[-1]
        ):
            continue

        # 前一日收盤價 > 週線, 半月線, 月線
        if (
            close[-2] < week_sma[-2]
            or close[-2] < half_month_sma[-2]
            or close[-2] < month_sma[-2]
        ):
            continue

        # 突破: 前五日低價 < 週線, 半月線, 月線
        min_low = np.min(low[-5:])
        if (
            min_low > week_sma[-1]
            or min_low > half_month_sma[-1]
            or min_low > month_sma[-1]
        ):
            continue

        # 當日收盤價 > 開盤價
        if current_price < open_price[-1]:
            continue

        filtered_codes.append(code)

    return filtered_codes


def trend_filter(codes: List[str], config: dict):
    from_year = config["meta"]["from_year"]
    code_to_signals = find_local_max_min(codes, config)
    code_to_pv = read_daily_data_by_codes(codes, "daily_price", from_year)
    filtered_codes = []

    for code, signals in code_to_signals.items():
        signals = signals["signal"]
        latest_signal = signals[-1]
        if latest_signal["type"] == "max":
            filtered_codes.append(code)

    return filtered_codes


def recurring_eps_filter(codes: List[str], config: dict):
    eps_signal, recurring_eps_signal, both_signal = surpass_eps(
        config
    )  # [last 1, last 2]...
    finance_code = recurring_eps_signal[-1]
    eps_filtered_codes = list(set(codes).intersection(finance_code))
    return eps_filtered_codes


def find_strategy_one_signal(codes: List[str], config: dict):
    """
    第一階段: 上升
    第二階段: 下降
    第三階段: 盤整
    第四階段: 突破
    """
    from_year = config["meta"]["from_year"]
    strategy_one_config = config["strategy_one"]
    start_days_before_current_date = strategy_one_config[
        "start_days_before_current_date"
    ]
    signal_before_days = strategy_one_config["signal_before_days"]
    up_min_ratio = strategy_one_config["up_min_ratio"]
    up_time_window = strategy_one_config["up_time_window"]
    down_max_ratio = strategy_one_config["down_max_ratio"]
    down_max_time_window = strategy_one_config["down_max_time_window"]
    consolidation_time_window = strategy_one_config["consolidation_time_window"]
    breakthrough_fuzzy = strategy_one_config["breakthrough_fuzzy"]

    current_date = datetime.datetime.now()
    start_date = current_date - datetime.timedelta(days=start_days_before_current_date)
    signal_date = current_date - datetime.timedelta(days=signal_before_days)
    code_to_local_extremums = find_middle_max(codes, config)
    code_to_signals = dict()
    code_to_ready_to_breakthrough = dict()
    not_enough_data_codes = []
    not_enough_consolidation_codes = []

    for code in codes:
        local_extremums = code_to_local_extremums[code]
        # start from start data
        for i in range(0, len(local_extremums)):
            if local_extremums[i]["date"] > start_date:
                break

        local_extremums = local_extremums[i:]
        df = get_daily_price_dataframe(code, from_year)
        for idx, local_extremum in enumerate(local_extremums):
            if idx == 113:
                print("debug")
            # filter by trading volume
            if (
                local_extremum["index"] < 120
                or df.iloc[local_extremum["index"] - 120 : local_extremum["index"]]["volume"].mean() < 200
            ):
                continue

            if local_extremum["type"] != "max":
                continue

            local_extremum_idx, local_extremum_value, local_extremum_date = (
                local_extremum["index"],
                local_extremum["value"],
                local_extremum["date"],
            )

            if local_extremum_idx - up_time_window < 0:
                not_enough_data_codes.append(code)
                continue

            prev_low_idx = df.iloc[local_extremum_idx - up_time_window : local_extremum_idx + 1]["low"].idxmin()
            prev_low = df.iloc[prev_low_idx]["low"]
            # 上升
            if (local_extremum_value - prev_low) / prev_low < up_min_ratio:
                continue

            # 下降
            while idx < len(local_extremums) and local_extremums[idx]["type"] != "min":
                idx += 1
            if idx != len(local_extremums):
                down_low_value_idx = local_extremums[idx]["index"]
                down_low_value = local_extremums[idx]["value"]
            else:
                down_low_value_idx = df.iloc[
                    local_extremum_idx : local_extremum_idx + down_max_time_window + 1
                ]["low"].idxmin()
                down_low_value = df.iloc[down_low_value_idx]["low"]
            if (local_extremum_value - down_low_value) / local_extremum_value > down_max_ratio:
                continue

            # 盤整 & 突破
            breakthrough_idx = down_low_value_idx
            while (
                breakthrough_idx < len(df)
                and df.iloc[breakthrough_idx]["close"] < local_extremum_value
            ):
                breakthrough_idx += 1

            if breakthrough_idx - local_extremum_idx < consolidation_time_window:
                not_enough_consolidation_codes.append(code)
                continue

            if breakthrough_idx == len(df):
                # find ready to breakthrough
                if (local_extremum_value - df.iloc[-3:]["close"].max()) / df.iloc[-3:][
                    "close"
                ].max() < breakthrough_fuzzy:
                    code_to_ready_to_breakthrough[code] = [
                        {
                            "up_date": local_extremum_date,
                            "up_index": local_extremum_idx,
                            "signal_index": breakthrough_idx - 1,
                            "signal_date": df.iloc[breakthrough_idx - 1]["date"],
                        }
                    ]
                break

            if signal_date > df.iloc[breakthrough_idx]["date"]:
                continue

            datetime.datetime.now().isoformat()
            if code not in code_to_signals:
                code_to_signals[code] = []

            code_to_signals[code].append(
                {
                    "up_date": local_extremum_date,
                    "up_index": local_extremum_idx,
                    "signal_index": breakthrough_idx,
                    "signal_date": df.iloc[breakthrough_idx]["date"],
                }
            )

    print(f"Not enough data: {len(not_enough_data_codes)}")
    print(f"Not enough consolidation: {len(not_enough_consolidation_codes)}")
    return code_to_signals, code_to_ready_to_breakthrough


def find_strategy_two_signal(codes: List[str], config: dict):
    """
    第一階段: 上升
    第二階段: 下降
    第三階段: 盤整
    第四階段: 突破(假)
    第五階段: 下跌
    第六階段: 突破(真)
    """
    start_days_before_current_date = config["start_days_before_current_date"]
    signal_before_days = config["signal_before_days"]
    up_min_ratio = config["up_min_ratio"]
    up_time_window = config["up_time_window"]
    down_max_ratio = config["down_max_ratio"]
    down_max_time_window = config["down_max_time_window"]
    consolidation_time_window = config["consolidation_time_window"]
    breakthrough_fuzzy = config["breakthrough_fuzzy"]

    current_date = datetime.datetime.now()
    start_date = current_date - datetime.timedelta(days=start_days_before_current_date)
    signal_date = current_date - datetime.timedelta(days=signal_before_days)
    code_to_local_extremums = find_middle_max(codes)
    code_to_signals = dict()
    code_to_ready_to_breakthrough = dict()
    not_enough_data_codes = []
    not_enough_consolidation_codes = []
    for code in codes:
        local_extremums = code_to_local_extremums[code]
        # start from start data
        for i in range(0, len(local_extremums)):
            if local_extremums[i]["date"] > start_date:
                break

        local_extremums = local_extremums[i:]
        df = get_daily_price_dataframe(code)
        for idx, local_extremum in enumerate(local_extremums):
            if local_extremum["type"] != "max":
                continue
            local_extremum_idx, local_extremum_value, local_extremum_date = (
                local_extremum["index"],
                local_extremum["value"],
                local_extremum["date"],
            )

            if local_extremum_idx - up_time_window < 0:
                not_enough_data_codes.append(code)
                continue

            prev_low_idx = df.iloc[
                local_extremum_idx - up_time_window : local_extremum_idx + 1
            ]["low"].idxmin()
            prev_low = df.iloc[prev_low_idx]["low"]
            # 上升
            if (local_extremum_value - prev_low) / prev_low < up_min_ratio:
                continue

            # 下降
            while idx < len(local_extremums) and local_extremums[idx]["type"] != "min":
                idx += 1
            if idx != len(local_extremums):
                down_low_value_idx = local_extremums[idx]["index"]
                down_low_value = local_extremums[idx]["value"]
            else:
                down_low_value_idx = df.iloc[
                    local_extremum_idx : local_extremum_idx + down_max_time_window + 1
                ]["low"].idxmin()
                down_low_value = df.iloc[down_low_value_idx]["low"]
            if (
                local_extremum_value - down_low_value
            ) / local_extremum_value > down_max_ratio:
                continue

            # 盤整 & 突破
            breakthrough_idx = down_low_value_idx
            while (
                breakthrough_idx < len(df)
                and df.iloc[breakthrough_idx]["close"] < local_extremum_value
            ):
                breakthrough_idx += 1

            if breakthrough_idx - local_extremum_idx < consolidation_time_window:
                not_enough_consolidation_codes.append(code)
                continue

            if breakthrough_idx == len(df):
                # find ready to breakthrough
                if (local_extremum_value - df.iloc[-3:]["close"].max()) / df.iloc[-3:][
                    "close"
                ].max() < breakthrough_fuzzy:
                    code_to_ready_to_breakthrough[code] = [
                        {
                            "up_date": local_extremum_date,
                            "up_index": local_extremum_idx,
                            "signal_index": breakthrough_idx - 1,
                            "signal_date": df.iloc[breakthrough_idx - 1]["date"],
                        }
                    ]
                break

            if signal_date > df.iloc[breakthrough_idx]["date"]:
                continue

            datetime.datetime.now().isoformat()
            if code not in code_to_signals:
                code_to_signals[code] = []

            code_to_signals[code].append(
                {
                    "up_date": local_extremum_date,
                    "up_index": local_extremum_idx,
                    "signal_index": breakthrough_idx,
                    "signal_date": df.iloc[breakthrough_idx]["date"],
                }
            )

    print(f"Not enough data: {len(not_enough_data_codes)}")
    print(f"Not enough consolidation: {len(not_enough_consolidation_codes)}")
    return code_to_signals, code_to_ready_to_breakthrough


def get_relative_strength(time_period: int = 1):
    prevous_n_days = 60
    code_to_pv = read_daily_data("daily_price", False)
    all_closes = np.array(
        [
            np.array(pv[-prevous_n_days - 1 :]).transpose()[5].astype(np.float64)
            for pv in code_to_pv.values()
        ]
    )
    all_closes_change = (
        np.diff(all_closes, n=time_period) / all_closes[:, :-time_period]
    )

    market_index_df = get_tw50_index_df()
    market_index_change = np.diff(market_index_df["close"].to_numpy(), n=time_period)[
        -prevous_n_days:
    ]
    relative_strength = all_closes_change / market_index_change
    relative_strength_sorted_idx = relative_strength.transpose().argsort(axis=1)

    all_codes = np.array(list(code_to_pv.keys()))
    relative_strength_sorted = []
    for idx in relative_strength_sorted_idx:
        relative_strength_sorted.append(all_codes[idx])

    code_to_rs = {code: [] for code in all_codes}
    total_code_number = len(code_to_pv)
    for daily_rs in relative_strength_sorted:
        for idx, code in enumerate(daily_rs):
            code_to_rs[code].append(math.floor(idx / total_code_number * 100))

    return code_to_rs


def analyze_signal(code_to_signals: Dict[str, List[Dict]], config):
    STOP_LOSS_THRESHOLD = -0.04
    from_year = config["meta"]["from_year"]
    signal_result = []
    for code, signals in code_to_signals.items():
        df = get_daily_price_dataframe(code)
        for signal in signals:
            up_date, up_index, signal_index, signal_date = (
                signal["up_date"],
                signal["up_index"],
                signal["signal_index"],
                signal["signal_date"],
            )
            slice_df = df.iloc[signal_index : signal_index + 10]

            signal_data_price = df.iloc[signal_index]["close"]
            last_price = slice_df.iloc[-1]["close"]
            min_in_2w, max_in_2w = slice_df["close"].min(), slice_df["close"].max()

            hold_in_2w_ratio = (last_price - signal_data_price) / signal_data_price
            min_in_2w_ratio = (min_in_2w - signal_data_price) / signal_data_price
            max_in_2w_ratio = (max_in_2w - signal_data_price) / signal_data_price

            if min_in_2w_ratio < STOP_LOSS_THRESHOLD:
                signal_result.append(
                    {
                        "code": code,
                        "start_date": signal_date,
                        "start_price": signal_data_price,
                        "end_date": slice_df.iloc[slice_df["close"].argmin()]["date"],
                        "end_price": min_in_2w,
                        "min_in_2w": STOP_LOSS_THRESHOLD,
                        "max_in_2w": 0,
                        "hold_in_2w": STOP_LOSS_THRESHOLD,
                    }
                )
                continue

            signal_result.append(
                {
                    "code": code,
                    "start_date": signal_date,
                    "start_price": signal_data_price,
                    "end_date": slice_df.iloc[-1]["date"],
                    "end_price": last_price,
                    "min_in_2w": min_in_2w_ratio,
                    "max_in_2w": max_in_2w_ratio,
                    "hold_in_2w": hold_in_2w_ratio,
                }
            )

    return signal_result
