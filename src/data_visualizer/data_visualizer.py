from datetime import datetime
from typing import Dict, List
import numpy as np
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.strategy.trend_strategy import TrendStrategy
from src.data_store.data_store import DataStore
from src.utils.common import DataCategory, Instrument, Market, OrderSide, DataColumn, TechnicalIndicator
from src.data_transformer.data_transformer import DataTransformer
from src.config.default import config

plotly_config = dict(
    {
        "scrollZoom": True,
        "modeBarButtonsToRemove": ["select"],
    }
)

class DataVisualizer:
    def __init__(self):
        pass

    def visualize_signal_one(codes: List[str]):
        data_store = DataStore(codes=codes)
        global trading_codes, trading_dates
        [open_, high_, low_, close_, volume_], trading_dates, trading_codes = data_store.get_data(
            market=Market.TW,
            instrument=Instrument.Stock,
            data_category=DataCategory.Daily_Price,
            data_columns=[DataColumn.Open, DataColumn.High, DataColumn.Low, DataColumn.Close, DataColumn.Volume],
        )
        signal_one_, mark_ = DataTransformer.get_signal_one(config["parameter"], close_, high_, low_, volume_)
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
            DataVisualizer.plot_strategy_one_signal(code, data)


    def visualize_local_min_max(codes: List[str]):
        data_store = DataStore(codes=codes)
        global trading_codes, trading_dates

        [open_, high_, low_, close_, volume_], trading_dates, trading_codes = data_store.get_data(
            market=Market.TW,
            instrument=Instrument.Stock,
            data_category=DataCategory.Daily_Price,
            data_columns=[DataColumn.Open, DataColumn.High, DataColumn.Low, DataColumn.Close, DataColumn.Volume],
        )
        local_min_, local_max_ = DataTransformer.get_middle_ex(low_, high_)
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
            DataVisualizer.plot_local_min_max(code, data)


    def basic_plot_stock(code, price_df):
        draw_df = price_df
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
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=("OHLC", "Volume"),
            row_width=[0.2, 0.7],
            specs=[[{"secondary_y": True}], [{}]],
        )
        fig.add_trace(candlestick, secondary_y=False, row=1, col=1)
        fig.add_trace(trace_close, secondary_y=False, row=1, col=1)
        fig.add_trace(volume_bars, row=2, col=1)
        fig.update_yaxes(
            secondary_y=False,
            showgrid=True,
        )
        fig.update_layout(
            # title=get_stock_meta(code),
            xaxis_rangeslider_visible=False,
            xaxis=dict(type="category"),
        )
        fig.update_xaxes(matches="x")
        return fig


    def plot_strategy_one_signal(code, draw_df: Dict[str, np.ndarray]):
        fig = DataVisualizer.basic_plot_stock(code, draw_df)
        start_idx_list = np.argwhere(draw_df['mark']>0).reshape(-1)
        end_idx_list = draw_df['mark'][draw_df['mark']>0].reshape(-1)

        for start_idx, end_idx in zip(start_idx_list, end_idx_list):
            fig.add_trace(
                go.Scatter(
                    x=[draw_df['date'][start_idx], draw_df['date'][end_idx]],
                    y=[draw_df["high"][start_idx], draw_df["close"][end_idx]],
                    mode="lines",
                    line=dict(color="blue", width=3),
                ),
                secondary_y=False,
                row=1,
                col=1,
            )

        fig.show(config=plotly_config)


    def plot_local_min_max(code, draw_df: Dict[str, np.ndarray]):
        fig = DataVisualizer.basic_plot_stock(code, draw_df)
        fig.add_trace(
            go.Scatter(
                x=np.array(draw_df['date'])[draw_df['local_min']],
                y=draw_df['low'][draw_df['local_min']],
                name="local_min",
                mode="markers",
                marker=dict(color="blue", size=10),
            ),
            secondary_y=False,
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=np.array(draw_df['date'])[draw_df['local_max']],
                y=draw_df['high'][draw_df['local_max']],
                name="local_max",
                mode="markers",
                marker=dict(color="purple", size=10),
            ),
            secondary_y=False,
            row=1,
            col=1,
        )

        fig.show(config=plotly_config)
