import pandas as pd
import numpy as np
import os
import json
import plotly.graph_objects as go
from datetime import datetime, time
from pathlib import Path

from src.data_processor.future_record_processor import FutureRecordProcessor
from src.data_visualizer.data_visualizer import DataVisualizer
from src.data_store.data_store import DataStore
from src.utils.common import (
    DataCategory,
    Instrument,
    Market,
    DataColumn,
    OrderSide
)

class FutureOrderVisualizer:
    @staticmethod
    def visualize_future_record(
        record_path: Path,
        data_dir: Path,
        start_date: datetime = datetime(2000, 1, 1),
        end_date: datetime = datetime(2050, 1, 1),
    ):
        record_df = pd.read_csv(
            record_path, parse_dates=["executed_datetime", "order_date"]
        )
        trading_dates = record_df["order_date"].map(lambda x: x.date()).unique()
        trading_dates = trading_dates[
            (trading_dates >= start_date.date()) & (trading_dates <= end_date.date())
        ]
        for trading_date in trading_dates[-10:]:
            day_data_path = (
                data_dir
                / str(trading_date.year)
                / f"{trading_date.strftime('%Y-%m-%d')}.csv"
            )
            day_trading_df = record_df[
                record_df["order_date"].map(lambda x: x.date()) == trading_date
            ]
            data_df = pd.read_csv(day_data_path)

            data_df["date"] = pd.to_datetime(data_df["date"])
            data_df[["open", "high", "low", "close", "volume"]] = data_df[
                ["open", "high", "low", "close", "volume"]
            ].astype(float)
            data_df = (
                data_df.resample(f"5min", on="date")
                .agg(
                    open=("open", "first"),
                    high=("high", "max"),
                    low=("low", "min"),
                    close=("close", "last"),
                    volume=("volume", "sum"),
                )
            ).reset_index()

            # get actual order execute time
            fig = FutureOrderVisualizer.plot_future_record(data_df, day_trading_df)
            fig.show()

    @staticmethod
    def visualize_intra_strategy(
        record_path: Path,
        data_dir: Path,
        start_date: datetime = datetime(2000, 1, 1),
        end_date: datetime = datetime(2050, 1, 1),
    ):
        # process record
        order_record_df = pd.read_csv(record_path)
        future_records = []
        for row in order_record_df.iterrows():
          row = row[1]
          gap = json.loads(row["gap"])
          new_order = {
            "side": "long" if row["side"] == "OrderSide.Buy" else "short",
            "order_quantity": row["volume"],
            "executed_quantity": row["volume"],
            "executed_price": row["buy_price"],
            "order_type": "open",
            "order_date": row["buy_date"],
            "executed_datetime": row["buy_date"],
            "profit_loss": 0,
            "uncovered_quantity": row["volume"]
          }
          cover_order = {
            "side": "long" if row["side"] == "OrderSide.Sell" else "short",
            "order_quantity": row["volume"],
            "executed_quantity": row["volume"],
            "executed_price": row["cover_price"],
            "order_type": "close",
            "order_date": row["cover_date"],
            "executed_datetime": row["cover_date"],
            "profit_loss": row["profit_loss"],
            "uncovered_quantity": 0
          }
          future_records.append(new_order)
          future_records.append(cover_order)

        record_df = pd.DataFrame(future_records)
        record_df["executed_datetime"] = pd.to_datetime(record_df["executed_datetime"])
        record_df["order_date"] = record_df["executed_datetime"].map(
            lambda x: FutureOrderVisualizer.align_tx_date(x, FutureRecordProcessor.get_trading_dates())
        )
        record_df.sort_values("executed_datetime", inplace=True)
        
        trading_dates = record_df["order_date"].map(lambda x: x.date()).unique()
        trading_dates = trading_dates[(trading_dates >= start_date.date()) & (trading_dates <= end_date.date())]

        # prepare data
        tx_df, twse_df = FutureOrderVisualizer.prepare_data()
        valid_trading_dates = FutureRecordProcessor.get_trading_dates()

        for trading_date in trading_dates[-10:]:
            plot_start_dates = valid_trading_dates[valid_trading_dates.index(trading_date)-3]
            plot_end_dates = valid_trading_dates[valid_trading_dates.index(trading_date)+1]
            plot_tx_df = tx_df[(tx_df["date"].map(lambda x: x.date()) >= plot_start_dates) & (tx_df["date"].map(lambda x: x.date()) <= plot_end_dates)]
            day_trading_df = record_df[record_df["order_date"].map(lambda x: x.date()) == trading_date]
            fig = FutureOrderVisualizer.plot_future_record(plot_tx_df, day_trading_df)
            fig.show()

    @staticmethod
    def visualize_pair_intra_strategy(record_path: Path):
        # 一張圖包含新倉, 平倉, 缺口
        # prepare data
        tx_df, twse_df = FutureOrderVisualizer.prepare_data()

        # process record
        order_record_df = pd.read_csv(record_path)
        for row in order_record_df.iterrows():
            row = row[1]
            new_order = {
              "side": "long" if row["side"] == "OrderSide.Buy" else "short",
              "order_quantity": row["volume"],
              "executed_quantity": row["volume"],
              "executed_price": row["buy_price"],
              "order_type": "open",
              "order_date": row["buy_date"],
              "executed_datetime": row["buy_date"],
              "profit_loss": 0,
              "uncovered_quantity": row["volume"]
            }
            cover_order = {
              "side": "long" if row["side"] == "OrderSide.Sell" else "short",
              "order_quantity": row["volume"],
              "executed_quantity": row["volume"],
              "executed_price": row["cover_price"],
              "order_type": "close",
              "order_date": row["cover_date"],
              "executed_datetime": row["cover_date"],
              "profit_loss": row["profit_loss"],
              "uncovered_quantity": 0
            }

            record_df = pd.DataFrame([new_order, cover_order])
            record_df["executed_datetime"] = pd.to_datetime(record_df["executed_datetime"])
            record_df["order_date"] = record_df["executed_datetime"].map(
                lambda x: FutureOrderVisualizer.align_tx_date(x, FutureRecordProcessor.get_trading_dates())
            )
            record_df.sort_values("executed_datetime", inplace=True)
            
            gap = json.loads(row["gap"])
            gap = {
                **gap,
                "open_date": datetime.fromisoformat(gap['open_date']),
                "close_date": datetime.fromisoformat(gap['close_date'])
            }
            start_trading_date = gap['open_date'].date()
            end_trading_date = max(gap['close_date'], record_df["order_date"][1]).date()

            valid_trading_dates = FutureRecordProcessor.get_trading_dates()
            plot_start_dates = valid_trading_dates[valid_trading_dates.index(start_trading_date)-1]
            plot_end_dates = valid_trading_dates[min(valid_trading_dates.index(end_trading_date)+1, len(valid_trading_dates)-1)]
            plot_tx_df = tx_df[(tx_df["date"].map(lambda x: x.date()) >= plot_start_dates) & (tx_df["date"].map(lambda x: x.date()) <= plot_end_dates)]
            fig = FutureOrderVisualizer.plot_future_record(plot_tx_df, record_df)
            fig = FutureOrderVisualizer.plot_gap(fig, gap)
            fig.show()

    @staticmethod
    def visualize_gaps(
        data_dir: Path,
        start_date: datetime = datetime(2000, 1, 1),
        end_date: datetime = datetime(2050, 1, 1),
    ):
        tx_df, twse_df = FutureOrderVisualizer.prepare_data()
        gaps = json.load(open(data_dir / "gap.json"))
        gaps


    @staticmethod
    def plot_future_record(data_df: pd.DataFrame, record_df: pd.DataFrame):
        fig = DataVisualizer.basic_plot_stock("TX", data_df)
        long_record_df = record_df[record_df["side"] == "long"]

        fig.add_trace(
            go.Scatter(
                x=long_record_df["executed_datetime"],
                y=long_record_df["executed_price"],
                name="buy",
                mode="markers+text",
                text=long_record_df["profit_loss"],
                textfont=dict(weight="bold"),
                textposition="top center",
                marker=dict(color="green", size=15),
            ),
            secondary_y=False,
            row=1,
            col=1,
        )

        short_record_df = record_df[record_df["side"] == "short"]
        fig.add_trace(
            go.Scatter(
                x=short_record_df["executed_datetime"],
                y=short_record_df["executed_price"],
                name="sell",
                mode="markers+text",
                text=short_record_df["profit_loss"],
                textfont=dict(weight="bold"),
                textposition="top center",
                marker=dict(color="red", size=15),
            ),
            secondary_y=False,
            row=1,
            col=1,
        )
        daily_profit = record_df["profit_loss"].sum()
        fig.update_layout(title=f"Date: {str(data_df['date'].iloc[-1].date())}, Profit: {daily_profit}")

        return fig
    

    @staticmethod
    def plot_gap(fig: go.Figure, gap: dict):
         # plot gap as box with filled color on the chart
        color = "red" if gap["side"] == "long" else "green"
        fig.add_shape(
            # Line Horizontal
            type="rect",
            x0=gap["open_date"],
            y0=gap["lower"],
            x1=gap["close_date"],
            y1=gap["upper"],
            line=dict(color=color, width=2),
            fillcolor=color,
            opacity=0.2
        )

        return fig
    @staticmethod
    def prepare_data():
        data_store = DataStore()
        tx_data, tx_dates, tx_codes = data_store.get_data(
            Market.TW,
            Instrument.Future,
            DataCategory.Minute_Price,
            [DataColumn.Open, DataColumn.High, DataColumn.Low, DataColumn.Close, DataColumn.Volume],
        )
        twse_data, twse_dates, twse_codes = data_store.get_data(
            Market.TW,
            Instrument.StockIndex,
            DataCategory.Minute_Index,
            [DataColumn.Open, DataColumn.High, DataColumn.Low, DataColumn.Close],
        )
        tx_df = pd.DataFrame(np.stack([data[:, 0].T for data in tx_data]).T, columns=['open', 'high', 'low', 'close', 'volume'], index=tx_dates)
        twse_df = pd.DataFrame(np.stack([data[:, twse_codes.index('twse')].T for data in twse_data]).T, columns=['open', 'high', 'low', 'close'], index=twse_dates)
        tx_df = tx_df.resample('5min').agg(
          open=("open", "first"),
          high=("high", "max"),
          low=("low", "min"),
          close=("close", "last"),
          volume=("volume", "sum")
        )
        twse_df = twse_df.resample('5min').agg(
          open=("open", "first"),
          high=("high", "max"),
          low=("low", "min"),
          close=("close", "last")
        )
        tx_df["is_night"] = tx_df.index.map(FutureRecordProcessor.is_night_market)
        tx_df["date"] = tx_df.index
        
        return tx_df, twse_df
    
    @staticmethod
    def align_tx_date(datetime_: datetime, valid_trading_dates: list) -> datetime:
        if not FutureRecordProcessor.is_night_market(datetime_):
            return datetime_
        # the date of order in night market should be next trading date
        # if the order time has crossed 00:00, the date should back 1 day
        if datetime_.time() == time(10, 20):
            datetime_
        if datetime_.time() >= time(0, 0) and datetime_.time() <= time(5, 0):
            datetime_ = datetime_ + pd.DateOffset(days=-1)
        new_date = valid_trading_dates[valid_trading_dates.index(datetime_.date()) + 1]
        if new_date is None:
            new_date
        return datetime(new_date.year, new_date.month, new_date.day)