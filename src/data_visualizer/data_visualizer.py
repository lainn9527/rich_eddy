from datetime import datetime
from typing import Dict, List
from pathlib import Path
from plotly.subplots import make_subplots
import csv
import numpy as np
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
import plotly.express as px
import json
import os

from src.strategy.trend_strategy import TrendStrategy
from src.data_store.data_store import DataStore
from src.utils.common import (
    DataCategory,
    Instrument,
    Market,
    OrderSide,
    DataColumn,
    TechnicalIndicator,
)
from src.data_transformer.data_transformer import DataTransformer
from src.config.default import config

plotly_config = dict(
    {
        "scrollZoom": True,
        "modeBarButtonsToRemove": ["select"],
    }
)
pd.options.plotting.backend = "plotly"


class DataVisualizer:
    def __init__(self):
        pass

    def visualize_signal_one(codes: List[str]):
        start_date = datetime(2021, 5, 18)
        end_date = datetime(2024, 3, 1)

        data_store = DataStore(codes=codes)
        global trading_codes, trading_dates
        [open_, high_, low_, close_, volume_], trading_dates, trading_codes = (
            data_store.get_data(
                market=Market.TW,
                instrument=Instrument.Stock,
                data_category=DataCategory.Daily_Price,
                data_columns=[
                    DataColumn.Open,
                    DataColumn.High,
                    DataColumn.Low,
                    DataColumn.Close,
                    DataColumn.Volume,
                ],
                start_date=start_date,
                end_date=end_date,
                # selected_codes=codes,
            )
        )

        market_index_, c, t = data_store.get_data(
            market=Market.TW,
            instrument=Instrument.StockIndex,
            data_category=DataCategory.Market_Index,
            data_columns=[DataColumn.Close],
            selected_codes=["Y9999"],
            start_date=trading_dates[0],
            end_date=trading_dates[-1],
        )

        relative_strength_ = DataTransformer.get_relative_strength(
            codes, close_, market_index_
        )
        relative_strength_sma_ = data_store.get_technical_indicator(
            TechnicalIndicator.SMA,
            relative_strength_,
            config["parameter"]["strategy_one"]["rs_sma_period"],
        )

        signal_one_, mark_ = DataTransformer.get_signal_one(
            config["parameter"],
            close_,
            high_,
            low_,
            volume_,
            relative_strength_sma_,
            trading_dates,
        )
        for code in codes:
            code_idx = trading_codes.index(code)
            data = {
                "date": trading_dates,
                "open": open_[:, code_idx],
                "high": high_[:, code_idx],
                "low": low_[:, code_idx],
                "close": close_[:, code_idx],
                "volume": volume_[:, code_idx],
                "signal": signal_one_[:, code_idx],
                "mark": mark_[:, code_idx],
            }
            fig = DataVisualizer.plot_strategy_one_signal(code, data)
            fig.show(config=plotly_config)

    def visualize_local_min_max(codes: List[str]):
        data_store = DataStore(codes=codes)
        global trading_codes, trading_dates
        [open_, high_, low_, close_, volume_], trading_dates, trading_codes = (
            data_store.get_data(
                market=Market.TW,
                instrument=Instrument.Stock,
                data_category=DataCategory.Daily_Price,
                data_columns=[
                    DataColumn.Open,
                    DataColumn.High,
                    DataColumn.Low,
                    DataColumn.Close,
                    DataColumn.Volume,
                ],
                selected_codes=codes,
            )
        )
        local_min_, local_max_ = DataTransformer.get_local_ex(low_, high_)
        middle_min_, middle_max_ = DataTransformer.get_middle_ex(low_, high_)
        for code in codes:
            code_idx = trading_codes.index(code)
            data = {
                "date": trading_dates,
                "open": open_[:, code_idx],
                "high": high_[:, code_idx],
                "low": low_[:, code_idx],
                "close": close_[:, code_idx],
                "volume": volume_[:, code_idx],
                "local_min": local_min_[:, code_idx],
                "local_max": local_max_[:, code_idx],
            }
            fig = DataVisualizer.plot_local_min_max(code, data)
            data["local_max"] = middle_max_[:, code_idx]
            data["local_min"] = middle_min_[:, code_idx]
            fig = DataVisualizer.plot_local_min_max(code, data, fig)
            fig.show(config=plotly_config)

    def group_signal_by_codes(signal_objects, codes):
        code_signal_dict = {}
        for signal_object in signal_objects:
            code = codes[signal_object["code_idx"]]
            if code not in code_signal_dict:
                code_signal_dict[code] = []
            code_signal_dict[code].append(signal_object)
        return code_signal_dict

    def visualize_trend_strategy(drawing_codes: List[str], result_dir: Path):
        with open(result_dir / "analyze_material.json", "r") as fp:
            analyze_material = json.load(fp)

        order_record_df = pd.read_csv(
            result_dir / "order_record.csv", header=0, parse_dates=[1, 6, 7]
        )
        order_record_df["code"] = order_record_df["code"].astype(str)

        start_date = datetime.fromisoformat(analyze_material["start_date"])
        end_date = datetime.fromisoformat(analyze_material["end_date"])
        local_min = np.array(analyze_material["local_min"])
        local_max = np.array(analyze_material["local_max"])
        filtered_reason = np.array(analyze_material["filtered_reason"])
        filtered_reason_mapper = analyze_material["filtered_reason_mapper"]
        signal_objects = analyze_material["signal_objects"]
        failure_breakthrough_objects = analyze_material["failure_breakthrough_objects"]

        data_store = DataStore()
        [open_, high_, low_, close_, volume_], dates, codes = data_store.get_data(
            market=Market.TW,
            instrument=Instrument.Stock,
            data_category=DataCategory.Daily_Price,
            data_columns=[
                DataColumn.Open,
                DataColumn.High,
                DataColumn.Low,
                DataColumn.Close,
                DataColumn.Volume,
            ],
            start_date=start_date,
            end_date=end_date,
        )
        close_5_sma = data_store.get_technical_indicator(
            TechnicalIndicator.SMA, close_, 5
        )
        close_10_sma = data_store.get_technical_indicator(
            TechnicalIndicator.SMA, close_, 10
        )
        close_20_sma = data_store.get_technical_indicator(
            TechnicalIndicator.SMA, close_, 20
        )
        close_60_sma = data_store.get_technical_indicator(
            TechnicalIndicator.SMA, close_, 60
        )
        close_120_sma = data_store.get_technical_indicator(
            TechnicalIndicator.SMA, close_, 120
        )

        # group signal by codes
        code_signal_dict = DataVisualizer.group_signal_by_codes(signal_objects, codes)
        code_failure_breakthrough_dict = DataVisualizer.group_signal_by_codes(
            failure_breakthrough_objects, codes
        )

        for code in drawing_codes:
            code_idx = codes.index(code)
            code_order_record_df = order_record_df[order_record_df["code"] == code]
            data = {
                "date": dates,
                "open": open_[:, code_idx],
                "high": high_[:, code_idx],
                "low": low_[:, code_idx],
                "close": close_[:, code_idx],
                "volume": volume_[:, code_idx],
                "local_min": local_min[:, code_idx],
                "local_max": local_max[:, code_idx],
                "filtered_reason": filtered_reason[:, code_idx],
                "filtered_reason_mapper": filtered_reason_mapper,
                "signal": code_signal_dict[code] if code in code_signal_dict else [],
                "failure_breakthrough": (
                    code_failure_breakthrough_dict[code]
                    if code in code_failure_breakthrough_dict
                    else []
                ),
                "sma_5": close_5_sma[:, code_idx],
                "sma_10": close_10_sma[:, code_idx],
                "sma_20": close_20_sma[:, code_idx],
                "sma_60": close_60_sma[:, code_idx],
                "sma_120": close_120_sma[:, code_idx],
            }

            fig = DataVisualizer.basic_plot_stock(code, data)
            fig = DataVisualizer.plot_trend_strategy(code, data, fig)
            # fig = DataVisualizer.plot_local_min_max(code, data, fig)
            fig = DataVisualizer.plot_correct_local_min_max(code, data, fig)
            fig = DataVisualizer.plot_filtered_reason(code, data, fig)
            fig = DataVisualizer.plot_order_record(
                code, data, code_order_record_df, fig
            )
            fig = DataVisualizer.plot_sma(code, data, fig)
            fig.show(config=plotly_config)

    def visualize_book_strategy(drawing_codes: List[str], result_dir: Path):
        with open(result_dir / "analyze_material.json", "r") as fp:
            analyze_material = json.load(fp)

        order_record_df = pd.read_csv(
            result_dir / "order_record.csv", header=0, parse_dates=[1, 6, 7]
        )
        order_record_df["code"] = order_record_df["code"].astype(str)

        start_date = datetime.fromisoformat(analyze_material["start_date"])
        end_date = datetime.fromisoformat(analyze_material["end_date"])
        local_min = np.array(analyze_material["local_min"])
        local_max = np.array(analyze_material["local_max"])
        filtered_reason = np.array(analyze_material["filtered_reason"])
        filtered_reason_mapper = analyze_material["filtered_reason_mapper"]
        signal_objects = analyze_material["signal_objects"]
        failure_breakthrough_objects = analyze_material["failure_breakthrough_objects"]

        data_store = DataStore()
        [open_, high_, low_, close_, volume_], dates, codes = data_store.get_data(
            market=Market.TW,
            instrument=Instrument.Stock,
            data_category=DataCategory.Daily_Price,
            data_columns=[
                DataColumn.Open,
                DataColumn.High,
                DataColumn.Low,
                DataColumn.Close,
                DataColumn.Volume,
            ],
            start_date=start_date,
            end_date=end_date,
        )
        close_5_sma = data_store.get_technical_indicator(
            TechnicalIndicator.SMA, close_, 5
        )
        close_10_sma = data_store.get_technical_indicator(
            TechnicalIndicator.SMA, close_, 10
        )
        close_20_sma = data_store.get_technical_indicator(
            TechnicalIndicator.SMA, close_, 20
        )
        close_60_sma = data_store.get_technical_indicator(
            TechnicalIndicator.SMA, close_, 60
        )
        close_120_sma = data_store.get_technical_indicator(
            TechnicalIndicator.SMA, close_, 120
        )

        # group signal by codes
        code_signal_dict = DataVisualizer.group_signal_by_codes(signal_objects, codes)
        code_failure_breakthrough_dict = DataVisualizer.group_signal_by_codes(
            failure_breakthrough_objects, codes
        )

        for code in drawing_codes:
            code_idx = codes.index(code)
            code_order_record_df = order_record_df[order_record_df["code"] == code]
            data = {
                "date": dates,
                "open": open_[:, code_idx],
                "high": high_[:, code_idx],
                "low": low_[:, code_idx],
                "close": close_[:, code_idx],
                "volume": volume_[:, code_idx],
                "local_min": local_min[:, code_idx],
                "local_max": local_max[:, code_idx],
                "filtered_reason": filtered_reason[:, code_idx],
                "filtered_reason_mapper": filtered_reason_mapper,
                "signal": code_signal_dict[code] if code in code_signal_dict else [],
                "failure_breakthrough": (
                    code_failure_breakthrough_dict[code]
                    if code in code_failure_breakthrough_dict
                    else []
                ),
                "sma_5": close_5_sma[:, code_idx],
                "sma_10": close_10_sma[:, code_idx],
                "sma_20": close_20_sma[:, code_idx],
                "sma_60": close_60_sma[:, code_idx],
                "sma_120": close_120_sma[:, code_idx],
            }

            fig = DataVisualizer.basic_plot_stock(code, data)
            fig = DataVisualizer.plot_book_strategy(code, data, fig)
            fig = DataVisualizer.plot_correct_local_min_max(code, data, fig)
            fig = DataVisualizer.plot_filtered_reason(code, data, fig)
            fig = DataVisualizer.plot_order_record(
                code, data, code_order_record_df, fig
            )
            fig = DataVisualizer.plot_sma(code, data, fig)
            fig.show(config=plotly_config)

    def visualize_trend_strategy_tune_result(result_dir: Path):
        summary_result = pd.read_csv(result_dir / "result.csv", header=0, index_col=0)
        summary_result[["final_return", "annualized_return", "max_draw_down"]] = (
            summary_result[["final_return", "annualized_return", "max_draw_down"]].map(
                lambda x: pd.to_numeric(x[:-1])
            )
        )
        tune_result_dirs = os.listdir(result_dir)

        tune_result_dict = {}
        for tune_result_dir in tune_result_dirs:
            if (result_dir / tune_result_dir).is_dir() == False:
                continue
            with open(result_dir / tune_result_dir / "config.json", "r") as fp:
                tune_result_dict[tune_result_dir] = json.load(fp)

        config_names = []
        tune_result_series = {}
        for name in summary_result.index:
            tune_config = tune_result_dict[name]
            for _, sub_config_value in tune_config.items():
                for config_name, config_value in sub_config_value.items():
                    if config_name not in tune_result_series:
                        tune_result_series[config_name] = []
                        config_names.append(config_name)
                    tune_result_series[config_name].append(config_value)

        tune_result_df = pd.DataFrame(tune_result_series, index=summary_result.index)
        result_df = pd.concat([summary_result, tune_result_df], axis=1)
        for config_name in config_names:
            fig = result_df[[config_name, "final_return"]].plot.box(
                x=config_name, y="final_return", title=f"{config_name} vs final_return"
            )
            fig.show(config=plotly_config)
            # agg_result = result_df.groupby(config_name).aggregate(lambda tdf: tdf.tolist())["final_return"]
            # # px.box(x, title=f"{config_name} vs final_return").show(config=plotly_config)
            # x = list(agg_result.index)
            # for i in agg_result.index:
            #     fig = go.scatter(x=i, y=agg_result[i], mode="markers")
            #     fig.show(config=plotly_config)

    def visualize_order_record(record_path: Path):
        order_record_df = pd.read_csv(record_path, header=0, parse_dates=[1, 5, 6])
        order_record_df["code"] = order_record_df["code"].astype(str)
        max_return_codes = (
            order_record_df["return_rate"]
            .groupby(order_record_df["code"])
            .max()
            .sort_values(ascending=False)
            .index.to_list()
        )
        # draw top 10 code with high return rate
        drawing_codes = max_return_codes[:10]

        # draw random 10 code
        drawing_codes = np.random.choice(order_record_df["code"].unique(), 10).tolist()

        data_store = DataStore()
        [open_, high_, low_, close_, volume_], trading_dates, trading_codes = (
            data_store.get_data(
                market=Market.TW,
                instrument=Instrument.Stock,
                data_category=DataCategory.Daily_Price,
                data_columns=[
                    DataColumn.Open,
                    DataColumn.High,
                    DataColumn.Low,
                    DataColumn.Close,
                    DataColumn.Volume,
                ],
                selected_codes=drawing_codes,
            )
        )

        for code in drawing_codes:
            code_idx = trading_codes.index(code)
            data_store = DataStore()
            code_order_record_df = order_record_df[order_record_df["code"] == code]
            not_nan_idx = np.argwhere(np.isnan(open_[:, code_idx]) == False).reshape(-1)
            price_df = {
                "date": trading_dates[not_nan_idx[0] : not_nan_idx[-1]],
                "open": open_[not_nan_idx[0] : not_nan_idx[-1], code_idx],
                "high": high_[not_nan_idx[0] : not_nan_idx[-1], code_idx],
                "low": low_[not_nan_idx[0] : not_nan_idx[-1], code_idx],
                "close": close_[not_nan_idx[0] : not_nan_idx[-1], code_idx],
                "volume": volume_[not_nan_idx[0] : not_nan_idx[-1], code_idx],
            }
            fig = DataVisualizer.plot_order_record(code, price_df, code_order_record_df)
            fig.show(config=plotly_config)

    def basic_plot_stock(code, draw_df):
        candlestick = go.Candlestick(
            x=draw_df["date"],
            open=draw_df["open"],
            high=draw_df["high"],
            low=draw_df["low"],
            close=draw_df["close"],
            name=code,
        )
        trace_close = go.Scatter(
            x=draw_df["date"],
            y=draw_df["close"],
            name="close",
            line=dict(color="black", width=1),
        )
        volume_bars = go.Bar(
            x=draw_df["date"],
            y=draw_df["volume"],
            showlegend=False,
            marker={
                "color": "black",
            },
        )
        fig = make_subplots(
            rows=1,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=("OHLC", "Volume"),
            specs=[[{"secondary_y": True}]],
        )

        fig.add_trace(candlestick, secondary_y=False, row=1, col=1)
        # fig.add_trace(trace_close, secondary_y=False, row=1, col=1)
        # fig.add_trace(volume_bars, secondary_y=True, row=1, col=1)
        fig.update_layout(
            xaxis_rangeslider_visible=False,
        )
        fig.update_xaxes(matches="x")
        fig.update_yaxes(
            tickangle=90,
            secondary_y=False,
            showgrid=True,
        )
        return fig

    def plot_strategy_one_signal(code, draw_df: Dict[str, np.ndarray], fig=None):
        if fig is None:
            fig = DataVisualizer.basic_plot_stock(code, draw_df)

        start_idx_list = np.argwhere(draw_df["mark"] > 0).reshape(-1)
        end_idx_list = draw_df["mark"][draw_df["mark"] > 0].reshape(-1)

        for start_idx, end_idx in zip(start_idx_list, end_idx_list):
            fig.add_trace(
                go.Scatter(
                    x=[draw_df["date"][start_idx], draw_df["date"][end_idx]],
                    y=[draw_df["high"][start_idx], draw_df["close"][end_idx]],
                    mode="lines",
                    line=dict(color="blue", width=3),
                ),
                secondary_y=False,
                row=1,
                col=1,
            )

        return fig

    def plot_trend_strategy(code, draw_df: Dict[str, any], fig):
        signals = draw_df["signal"]
        failure_breakthrough = draw_df["failure_breakthrough"]

        for signal in signals:
            signal_idx = signal["signal_idx"]
            start_max_idx = signal["start_max_idx"]
            middle_min_idx = signal["middle_min_idx"]

            fig.add_trace(
                go.Scatter(
                    x=[
                        draw_df["date"][start_max_idx],
                        draw_df["date"][middle_min_idx],
                        draw_df["date"][signal_idx],
                    ],
                    y=[
                        draw_df["high"][start_max_idx],
                        draw_df["low"][middle_min_idx],
                        draw_df["close"][signal_idx],
                    ],
                    mode="lines",
                    line=dict(color="yellow", width=3),
                ),
                secondary_y=False,
                row=1,
                col=1,
            )

        for signal in failure_breakthrough:
            signal_idx = signal["signal_idx"]
            start_max_idx = signal["start_max_idx"]
            middle_min_idx = signal["middle_min_idx"]

            fig.add_trace(
                go.Scatter(
                    x=[draw_df["date"][start_max_idx], draw_df["date"][signal_idx]],
                    y=[draw_df["high"][start_max_idx], draw_df["close"][signal_idx]],
                    mode="lines",
                    name="failure_breakthrough",
                    line=dict(color="red", width=3),
                ),
                secondary_y=False,
                row=1,
                col=1,
            )
        return fig

    def plot_book_strategy(code, draw_df: Dict[str, any], fig):
        signals = draw_df["signal"]
        failure_breakthrough = draw_df["failure_breakthrough"]

        for signal in signals:
            signal_idx = signal["signal_idx"]
            start_max_idx = signal["start_max_idx"]
            middle_min_idx = signal["middle_min_idx"]
            signal_detected_idx = signal["signal_detected_idx"]
            fig.add_trace(
                go.Scatter(
                    x=[
                        draw_df["date"][start_max_idx],
                        draw_df["date"][middle_min_idx],
                        draw_df["date"][signal_idx],
                    ],
                    y=[
                        draw_df["high"][start_max_idx],
                        draw_df["low"][middle_min_idx],
                        draw_df["close"][signal_idx],
                    ],
                    mode="lines",
                    line=dict(color="yellow", width=3),
                ),
                secondary_y=False,
                row=1,
                col=1,
            )

        signal_detected_idxes = [signal["signal_detected_idx"] for signal in signals]
        fig.add_trace(
            go.Scatter(
                x=np.array(draw_df["date"])[signal_detected_idxes],
                y=draw_df["close"][signal_detected_idxes],
                name="signal_detected",
                mode="markers",
                marker=dict(size=10),
            ),
            secondary_y=False,
            row=1,
            col=1,
        )

        for signal in failure_breakthrough:
            signal_idx = signal["signal_idx"]
            start_max_idx = signal["start_max_idx"]
            middle_min_idx = signal["middle_min_idx"]

            fig.add_trace(
                go.Scatter(
                    x=[draw_df["date"][start_max_idx], draw_df["date"][signal_idx]],
                    y=[draw_df["high"][start_max_idx], draw_df["close"][signal_idx]],
                    mode="lines",
                    name="failure_breakthrough",
                    line=dict(color="red", width=3),
                ),
                secondary_y=False,
                row=1,
                col=1,
            )
        return fig

    def plot_local_min_max(code, draw_df: Dict[str, np.ndarray], fig=None):
        if fig == None:
            fig = DataVisualizer.basic_plot_stock(code, draw_df)
        fig.add_trace(
            go.Scatter(
                x=np.array(draw_df["date"])[draw_df["local_min"]],
                y=draw_df["low"][draw_df["local_min"]],
                name="local_min",
                mode="markers",
                marker=dict(size=10),
            ),
            secondary_y=False,
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=np.array(draw_df["date"])[draw_df["local_max"]],
                y=draw_df["high"][draw_df["local_max"]],
                name="local_max",
                mode="markers",
                marker=dict(size=10),
                text=[i for i in range(draw_df["local_max"].sum())],
            ),
            secondary_y=False,
            row=1,
            col=1,
        )

        return fig

    def plot_correct_local_min_max(code, draw_df: Dict[str, np.ndarray], fig=None):
        if fig == None:
            fig = DataVisualizer.basic_plot_stock(code, draw_df)
        fig.add_trace(
            go.Scatter(
                x=np.array(draw_df["date"])[draw_df["local_min"] != -1],
                y=draw_df["low"][draw_df["local_min"] != -1],
                name="local_min",
                mode="markers",
                marker=dict(size=10, color="blue"),
            ),
            secondary_y=False,
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=np.array(draw_df["date"])[draw_df["local_max"] != -1],
                y=draw_df["high"][draw_df["local_max"] != -1],
                name="local_max",
                mode="markers",
                marker=dict(size=10, color="purple"),
            ),
            secondary_y=False,
            row=1,
            col=1,
        )

        return fig

    def plot_filtered_reason(code, draw_df: Dict[str, np.ndarray], fig=None):
        filtered_reason_mapper = draw_df["filtered_reason_mapper"]
        filtered_reason_df = pd.DataFrame(
            draw_df["filtered_reason"], dtype=str
        ).replace(filtered_reason_mapper)

        fig.add_trace(
            go.Scatter(
                x=np.array(draw_df["date"])[draw_df["filtered_reason"] != 0],
                y=draw_df["high"][draw_df["filtered_reason"] != 0],
                name="filtered_reason",
                mode="markers",
                marker=dict(color="purple", size=10),
                text=filtered_reason_df[filtered_reason_df != "0"].dropna()[0].tolist(),
            ),
            secondary_y=False,
            row=1,
            col=1,
        )

        return fig

    def plot_order_record(
        code, price_df: Dict[str, np.ndarray], order_record_df: pd.DataFrame, fig=None
    ):
        if fig == None:
            fig = DataVisualizer.basic_plot_stock(code, price_df)

        fig.add_trace(
            go.Scatter(
                x=order_record_df["buy_date"],
                y=order_record_df["buy_price"],
                name="buy",
                mode="markers",
                marker=dict(color="green", size=15),
            ),
            secondary_y=False,
            row=1,
            col=1,
        )
        cover_order_texts = [
            f'{row["return_rate"]:.2f}_{row["profit_loss"]:.2f}_{row["holding_days"]}_{row["cover_reason"]}'
            for _, row in order_record_df.iterrows()
        ]
        fig.add_trace(
            go.Scatter(
                x=order_record_df["cover_date"],
                y=order_record_df["cover_price"],
                name="sell",
                mode="markers",
                marker=dict(color="red", size=15),
                text=cover_order_texts,
            ),
            secondary_y=False,
            row=1,
            col=1,
        )
        return fig

    def plot_sma(code, draw_df: Dict[str, np.ndarray], fig=None):
        if fig == None:
            fig = DataVisualizer.basic_plot_stock(code, draw_df)
        for i in [5, 10, 20, 60, 120]:
            fig.add_trace(
                go.Scatter(
                    x=draw_df["date"],
                    y=draw_df[f"sma_{i}"],
                    name=f"sma_{i}",
                    line=dict(width=2),
                ),
                secondary_y=False,
                row=1,
                col=1,
            )
        return fig
