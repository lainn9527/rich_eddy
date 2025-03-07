import pandas as pd
import os
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta, time, date
from typing import List


def get_all_kbar_time_idx():
    night_market_start_time = datetime(1900, 1, 1, 15, 0)
    night_market_end_time = datetime(1900, 1, 2, 5, 0)
    kbar_time_idx = {}
    while night_market_start_time != night_market_end_time:
        kbar_time_idx[night_market_start_time.time()] = len(kbar_time_idx)
        night_market_start_time = night_market_start_time + timedelta(minutes=1)
    day_market_start_time = datetime(1900, 1, 1, 8, 45)
    day_market_end_time = datetime(1900, 1, 1, 13, 45)
    while day_market_start_time != day_market_end_time:
        kbar_time_idx[day_market_start_time.time()] = len(kbar_time_idx)
        day_market_start_time = day_market_start_time + timedelta(minutes=1)

    return kbar_time_idx


class FutureRecordProcessor:
    @staticmethod
    def is_night_market(datetime_: datetime) -> bool:
        return datetime_.time() >= time(15, 0) or datetime_.time() <= time(5, 0)

    @staticmethod
    def is_twse_open(datetime_: datetime) -> bool:
        return datetime_.time() >= time(9, 0) and datetime_.time() <= time(13, 30)

    @staticmethod
    def is_twse_open_bar(datetime_: datetime) -> bool:
        return datetime_.time() == time(9, 0)

    @staticmethod
    def is_twse_close_bar(datetime_: datetime) -> bool:
        return datetime_.time() == time(13, 30)

    @staticmethod
    def correct_time(time_: str, date_: str) -> tuple[str, str]:
        if int(time_[:2]) < 24:
            return time_, date_
        else:
            new_time = str(int(time_[:2]) - 24).zfill(2) + time_[2:]
            return new_time, date_
    
    @staticmethod
    def correct_date(datetime_: datetime, valid_trading_dates: List[date]) -> str:
        if FutureRecordProcessor.is_night_market(datetime_):
            new_date = valid_trading_dates[
                valid_trading_dates.index(datetime_.date()) - 1
            ]
            if datetime_.time() >= time(0, 0) and datetime_.time() <= time(5, 0):
                new_date = new_date + timedelta(days=1)
            datetime_ = datetime_.replace(
                year=new_date.year, month=new_date.month, day=new_date.day
            )
        return datetime_

    @staticmethod
    def process_time(order_df: pd.DataFrame) -> str:
        for _, row in order_df.iterrows():
            processed_time, processed_date = FutureRecordProcessor.correct_time(
                row["成交時間"], row["成交日期"]
            )
            order_df.at[_, "成交時間"] = processed_time
            order_df.at[_, "成交日期"] = processed_date
        order_datetime = pd.to_datetime(
            order_df["成交日期"] + " " + order_df["成交時間"],
            format="%Y/%m/%d %H:%M:%S",
        )
        order_df["is_night_market"] = order_datetime.map(
            lambda x: FutureRecordProcessor.is_night_market(x)
        )
        valid_trading_dates = FutureRecordProcessor.get_trading_dates()
        order_df["executed_datetime"] = order_datetime.map(
            lambda x: FutureRecordProcessor.correct_date(x, valid_trading_dates)
        )
        order_df["order_date"] = order_datetime.map(lambda x: x.date())

        return order_df

    @staticmethod
    def rename_and_pick_columns(df: pd.DataFrame):
        df.columns = df.columns.map(lambda x: x.strip())
        side = "買賣別" if "買賣別" in df.columns else "買/賣"
        order_quantity = "委託數量" if "委託數量" in df.columns else "委託口數"
        executed_quantity = "成交數量" if "成交數量" in df.columns else "成交口數"
        executed_price = "成交均價" if "成交均價" in df.columns else "成交價格"
        order_type = "倉別"
        rename_mapper = {
            side: "side",
            order_quantity: "order_quantity",
            executed_quantity: "executed_quantity",
            executed_price: "executed_price",
            order_type: "order_type",
            "is_night_market": "is_night_market",
            "order_date": "order_date",
        }
        return df.rename(columns=rename_mapper)[
            list(rename_mapper.values()) + ["executed_datetime"]
        ]

    @staticmethod
    def process_columns(df: pd.DataFrame):
        # 買進 -> long, 賣出 -> short, else error
        def process_side(x: str) -> str:
            if x == "買進":
                return "long"
            elif x == "賣出":
                return "short"
            else:
                raise ValueError(f"side value error: {x}")
        df["side"] = df["side"].map(process_side)

        def process_order_type(x: str) -> str:
            if x == "新倉":
                return "open"
            elif x == "平倉":
                return "close"
            else:
                raise ValueError(f"order_type value error: {x}")
        df["order_type"] = df["order_type"].map(process_order_type)

        df["executed_price"] = df["executed_price"].map(
            lambda x: x.replace(",", "").replace(".00", "")
        )
        df["executed_price"] = df["executed_price"].astype(float)
        return df

    @staticmethod
    def get_trading_dates():
        twse_index_path = Path("intra_day_data") / "future"
        year_dirs = os.listdir(twse_index_path)
        trading_dates = []
        for year_dir in year_dirs:
            trading_dates += [
                datetime.strptime(file_name.split(".")[0], "%Y-%m-%d").date()
                for file_name in os.listdir(twse_index_path / year_dir)
            ]

        trading_dates.sort()
        return trading_dates

    def process_profit_loss(order_df: pd.DataFrame):
        new_order_rows = []
        profit_loss = pd.Series([np.nan] * len(order_df))
        uncovered_quantity_series = pd.Series([0.0] * len(order_df))

        order_df = order_df.reset_index(drop=True)
        for idx, row in order_df.iterrows():
            order_side = 1 if row["side"] == "long" else -1
            uncovered_quantity = uncovered_quantity_series[max(idx-1, 0)]

            if uncovered_quantity == 0 or uncovered_quantity * order_side > 0:
                # new order or add order (加倉)
                new_order_rows.append(row.copy())
                uncovered_quantity_series[idx] = order_side * row["executed_quantity"] + uncovered_quantity
            else:
                # opposite side 反向倉位
                cover_side = 1 if row["side"] == "short" else -1
                executed_quantity = row["executed_quantity"]
                while executed_quantity > 0 and len(new_order_rows) > 0:
                    covered_quantity = min(abs(new_order_rows[-1]["executed_quantity"]), executed_quantity)
                    if np.isnan(profit_loss[idx]):
                        profit_loss[idx] = 0
                    profit_loss[idx] = profit_loss[idx] + 200 * cover_side * covered_quantity * (
                        row["executed_price"] - new_order_rows[-1]["executed_price"]
                    )
                    new_order_rows[-1]["executed_quantity"] -= covered_quantity
                    executed_quantity -= covered_quantity

                    if new_order_rows[-1]["executed_quantity"] <= 0:
                        new_order_rows.pop()

                if executed_quantity > 0:
                    new_order_row = row.copy()
                    new_order_row["executed_quantity"] = executed_quantity
                    new_order_rows.append(new_order_row)

                uncovered_quantity_series[idx] = order_side * row["executed_quantity"] + uncovered_quantity

        order_df["profit_loss"] = profit_loss
        order_df["uncovered_quantity"] = uncovered_quantity_series
        return order_df

    @staticmethod
    def process_raw_data(data_path: Path):
        order_df = pd.read_csv(data_path, encoding="utf-8")
        order_df = FutureRecordProcessor.process_time(order_df)
        order_df = FutureRecordProcessor.rename_and_pick_columns(order_df)
        order_df = FutureRecordProcessor.process_columns(order_df)
        kbar_time_idx = get_all_kbar_time_idx()
        order_df["kbar_time_idx"] = order_df["executed_datetime"].map(
            lambda x: kbar_time_idx[x.time().replace(second=0)]
        )
        order_df = order_df.sort_values("executed_datetime", ascending=True)
        order_df = FutureRecordProcessor.process_profit_loss(order_df)

        return order_df
