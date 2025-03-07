import pandas as pd
import os
import csv
from typing import Dict, List, Callable
from pathlib import Path
from datetime import datetime, timedelta, time
from io import StringIO


class IntraDayDataProcessor:
    @staticmethod
    def transform_and_write_year_dir_data(
        data_dir: Path, transform_fn: Callable, output_dir: Path, codes: List[str], minute_k = 1,
    ):
        year_dirs = os.listdir(data_dir)
        for year_dir in year_dirs:
            year_dir_path = data_dir / year_dir
            if not year_dir_path.is_dir():
                continue
            file_path = os.listdir(year_dir_path)
            for file_name in file_path:
                file_path = year_dir_path / file_name
                if not file_path.is_file():
                    continue
                date_str = file_name.split(".")[0]
                output_file_path = output_dir / year_dir / f"{date_str}.csv"
                if output_file_path.exists():
                    continue
                output_file_path.parent.mkdir(parents=True, exist_ok=True)
                code_df_dict = transform_fn(
                    file_path, codes, minute_k, datetime.fromisoformat(date_str)
                )
                df = pd.concat([ code_df for code_df in code_df_dict.values()])
                df.to_csv(output_file_path, index=False)

    @staticmethod
    def transform_csv_and_write_dir_data(
        twse_future_data: Dict[datetime, List[List[any]]],
        transform_fn: Callable,
        output_dir: Path,
        codes,
    ):
        for date, data in twse_future_data.items():
            # remove , in data
            data = [[d.replace(",", "") for d in row] for row in data]
            csv_str = "\n".join([",".join(d) for d in data])
            sio = StringIO(csv_str)
            code_df_dict = transform_fn(sio, codes, 1, date)

            for code, code_df in code_df_dict.items():
                year = date.year
                date_str = date.strftime("%Y-%m-%d")
                output_file_path = output_dir / code / str(year) / f"{date_str}.csv"
                output_file_path.parent.mkdir(parents=True, exist_ok=True)
                code_df.to_csv(output_file_path)

    @staticmethod
    def transform_twse_future_to_kbar(file_path: Path, codes: List[str], minute_k: int, trading_date: datetime):
        try:
            df = pd.read_csv(file_path, encoding="big5", dtype=str)
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding="utf-8", dtype=str)
        df[["商品代號", "到期月份(週別)"]] = df[["商品代號", "到期月份(週別)"]].map(
            str.strip
        )
        df["date"] = pd.to_datetime(
            df["成交日期"] + " " + df["成交時間"], format="%Y%m%d %H%M%S"
        )
        df["成交日期"] = pd.to_datetime(df["成交日期"])
        df[["成交價格", "成交數量(B+S)"]] = df[["成交價格", "成交數量(B+S)"]].astype(
            float
        )

        code_df_dict = {}
        for code in codes:
            # filter by code
            code_df = df[df["商品代號"] == code]
            # filter 近月期貨
            current_month = code_df["成交日期"].iloc[0].strftime("%Y%m")
            next_month = (
                datetime.strptime(current_month, "%Y%m") + pd.DateOffset(months=1)
            ).strftime("%Y%m")
            if (code_df["到期月份(週別)"] == current_month).any():
                code_df = code_df[code_df["到期月份(週別)"] == current_month]
            else:
                code_df = code_df[code_df["到期月份(週別)"] == next_month]
            code_df = (
                code_df[["date", "成交價格", "成交數量(B+S)"]]
                .resample(f"{minute_k}min", on="date")
                .agg(
                    open=("成交價格", "first"),
                    high=("成交價格", "max"),
                    low=("成交價格", "min"),
                    close=("成交價格", "last"),
                    volume=("成交數量(B+S)", "sum"),
                )
            )
            try:
                # find last 8:45 data since there might be holiday
                day_market = code_df.loc[
                    code_df[
                        code_df.index.map(lambda x: x.time()) == time(8, 45, 0)
                    ].index[-1] :
                ]
            except:
                day_market = code_df[0:0]
            night_market = code_df[
                (code_df.index < (code_df.index[0] + timedelta(hours=14)))
            ]
            code_df = pd.concat([night_market, day_market]).reset_index()
            code_df.insert(loc=0, column="code", value=code)
            code_df_dict[code] = code_df
        return code_df_dict

    @staticmethod
    def transform_twse_index_to_kbar(file_path: Path, codes: List[str], minute_k: int, trading_date: datetime):
        try:
            df = pd.read_csv(file_path, encoding="big5", dtype=str)
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding="utf-8", dtype=str)
        # remove all , in df cell
        df = df.map(lambda x: x.replace(",", ""))
        code_mapper = {
            "twse": "發行量加權股價指數",
            "ee": "電子類指數",
            "fn": "金融保險類指數",
        }
        df["date"] = pd.to_datetime(trading_date.date().isoformat() + ' ' + df["時間"], format='%Y-%m-%d %H:%M:%S')
        df = df.iloc[1:]
        code_df_dict = {}
        for code in codes:
            df["value"] = df[code_mapper[code]].astype(float)
            code_df = df[["date", "value"]].copy()
            code_df = code_df.resample(f"{minute_k}min", on="date").agg(
                open=("value", "first"),
                high=("value", "max"),
                low=("value", "min"),
                close=("value", "last"),
            ).reset_index()
            code_df.insert(loc=0, column="code", value=code)
            code_df_dict[code] = code_df
        return code_df_dict
    
    @staticmethod
    def transform_kbar_and_write_year_dir_data(
        data_dir: Path, output_dir: Path, codes=["TX"], minute_k=1
    ):
        for code in codes:
            code_data_dir = data_dir / code
            code_output_dir = output_dir / code

        year_dirs = os.listdir(code_data_dir)
        for year_dir in year_dirs:
            year_dir_path = code_data_dir / year_dir
            if not year_dir_path.is_dir():
                continue
            file_path = os.listdir(year_dir_path)
            for file_name in file_path:
                file_path = year_dir_path / file_name
                if not file_path.is_file():
                    continue

                df = IntraDayDataProcessor.transform_kbar(
                    file_path, minute_k
                )
                date_str = file_name.split(".")[0]
                output_file_path = code_output_dir / year_dir / f"{date_str}.csv"
                output_file_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(output_file_path)

    @staticmethod
    def transform_kbar(file_path: Path, minute_k = 1):
        try:
            df = pd.read_csv(file_path, encoding="big5", dtype=str)
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding="utf-8", dtype=str)

        df["date"] = pd.to_datetime(df["date"])
        df[["open", "high", "low", "close", "volume"]] = df[
            ["open", "high", "low", "close", "volume"]
        ].astype(float)
        df = (
            df.resample(f"{minute_k}min", on="date")
            .agg(
                open=("open", "first"),
                high=("high", "max"),
                low=("low", "min"),
                close=("close", "last"),
                volume=("volume", "sum"),
            )
        )

        return df
