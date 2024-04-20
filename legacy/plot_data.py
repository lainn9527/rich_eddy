from datetime import datetime

import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .analyze_data import find_local_max_min, find_middle_max, find_strategy_one_signal
from .utils import (
    get_daily_price_dataframe,
    get_eps_dataframe,
    get_stock_meta,
    get_trading_record_dataframe,
)


plotly_config = dict(
    {
        "scrollZoom": True,
        "modeBarButtonsToRemove": ["select"],
    }
)


def basic_plot_stock(code, start_date: datetime):
    price_df = get_daily_price_dataframe(code, start_date.year)
    if start_date:
        price_df = price_df[price_df["date"] >= start_date]

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
        title=get_stock_meta(code),
        xaxis_rangeslider_visible=False,
        xaxis=dict(type="category"),
    )
    fig.update_xaxes(matches="x")
    return fig


def plot_stock_with_signal(code, config):
    from_year = config["meta"]["from_year"]

    fig = basic_plot_stock(code, datetime(from_year, 1, 1))
    price_df = get_daily_price_dataframe(code, from_year)
    code_to_signal = find_middle_max([code], config)
    signal = code_to_signal[code]
    draw_df = price_df

    local_min_idx = (
        list(map(lambda x: x["index"], filter(lambda x: x["type"] == "min", signal))),
    )
    trace_local_min = go.Scatter(
        x=draw_df["date"].iloc[local_min_idx],
        y=draw_df["low"].iloc[local_min_idx],
        name="local_min",
        mode="markers",
        marker=dict(color="blue", size=10),
    )
    local_max_idx = (
        list(map(lambda x: x["index"], filter(lambda x: x["type"] == "max", signal))),
    )
    trace_local_max = go.Scatter(
        x=draw_df["date"].iloc[local_max_idx],
        y=draw_df["high"].iloc[local_max_idx],
        name="local_max",
        mode="markers",
        marker=dict(color="purple", size=10),
    )

    fig.add_trace(trace_local_min, secondary_y=False, row=1, col=1)
    fig.add_trace(trace_local_max, secondary_y=False, row=1, col=1)

    fig.show(config=plotly_config)


def plot_stock_with_eps(code):
    fig = basic_plot_stock(code)
    price_df = get_daily_price_dataframe(code)
    eps_df = get_eps_dataframe(code)
    # left join price_df and finance_df by date
    df = price_df.merge(eps_df, on="date", how="left", validate="one_to_one")

    draw_df = df
    eps_traces = [
        go.Scatter(
            x=list(draw_df["date"]),
            y=list(draw_df[name]),
            name=name,
        )
        for name in ["next_eps", "next_recurring_eps"]
    ]
    for eps_trace in eps_traces:
        fig.add_trace(eps_trace, secondary_y=True)
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list(
                [
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all"),
                ]
            )
        ),
    )

    fig.show()


def plot_stock_with_chip(code):
    CHIP_FILE_PATH = f"data/chip/code/{code}.csv"
    price_df = get_daily_price_dataframe(code)
    chip_df = pd.read_csv(
        CHIP_FILE_PATH,
        header=0,
        parse_dates=[1],
    )
    df = price_df.merge(
        chip_df, left_on="date", right_on="日期", how="left", validate="one_to_one"
    )
    print(df)

    # for i in range(0, len(df), len(df)):
    #     draw_df = df
    #     candlestick = go.Candlestick(
    #         x=draw_df['date'],
    #         open=draw_df['open'],
    #         high=draw_df['high'],
    #         low=draw_df['low'],
    #         close=draw_df['close'],
    #         name=code,
    #     )
    #     trace_close = go.Scatter(
    #         x=list(draw_df['date']),
    #         y=list(draw_df['close']),
    #         name='close',
    #     )
    #     eps_traces = [
    #         go.Scatter(
    #             x=list(draw_df['date']),
    #             y=list(draw_df[name]),
    #             name=name,
    #         ) for name in ['next_eps', 'next_recurring_eps']
    #     ]
    #     fig = make_subplots(specs=[[{'secondary_y': True}]])
    #     fig.add_trace(candlestick, secondary_y=False)
    #     for eps_trace in eps_traces:
    #         fig.add_trace(eps_trace, secondary_y=True)
    #     fig.update_xaxes(
    #         rangeslider_visible=True,
    #         rangeselector=dict(
    #             buttons=list(
    #                 [
    #                     dict(count=1, label='1m', step='month', stepmode='backward'),
    #                     dict(count=6, label='6m', step='month', stepmode='backward'),
    #                     dict(count=1, label='YTD', step='year', stepmode='todate'),
    #                     dict(count=1, label='1y', step='year', stepmode='backward'),
    #                     dict(step='all'),
    #                 ]
    #             )
    #         ),
    #     )

    #     fig.show()


def plot_stock_with_squeeze(code):
    price_df = get_daily_price_dataframe(code)
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
        x=list(draw_df["date"]),
        y=list(draw_df["close"]),
        name="close",
        line=dict(color="black", width=1),
    )
    volume_bars = go.Bar(
        x=draw_df["date"],
        y=draw_df["volume"],
        showlegend=False,
        marker={
            "color": "rgba(128,128,128,0.5)",
        },
    )

    bbands = ta.bbands(price_df["close"], length=20, std=2)
    trace_bbands_upper = go.Scatter(
        x=list(draw_df["date"]),
        y=list(bbands["BBU_20_2.0"]),
        name="BBU",
        line=dict(color="rgba(255, 165, 0, 1)", width=1),
    )
    trace_bbands_lower = go.Scatter(
        x=list(draw_df["date"]),
        y=list(bbands["BBL_20_2.0"]),
        name="BBL",
        line=dict(color="rgba(255, 165, 0, 1)", width=1),
    )

    kc = ta.kc(
        price_df["high"], price_df["low"], price_df["close"], length=20, scalar=2
    )
    trace_kc_upper = go.Scatter(
        x=list(draw_df["date"]),
        y=list(kc["KCUe_20_2.0"]),
        name="KCU",
        line=dict(color="blue", width=1),
    )
    trace_kc_lower = go.Scatter(
        x=list(draw_df["date"]),
        y=list(kc["KCLe_20_2.0"]),
        name="KCL",
        line=dict(color="blue", width=1),
    )

    squeeze_df = ta.squeeze(
        price_df["high"],
        price_df["low"],
        price_df["close"],
        bb_length=20,
        bb_std=2,
        kc_length=20,
        kc_scalar=2,
        mom_length=12,
        mom_smooth=3,
        use_tr=True,
        mamode="sma",
    )
    squeeze_df["date"] = draw_df["date"]
    squeeze_on_date = squeeze_df[squeeze_df["SQZ_ON"] == 1]["date"]
    squeeze_off_date = squeeze_df[squeeze_df["SQZ_ON"] == 0]["date"]
    trace_squeeze_on = go.Scatter(
        x=list(squeeze_on_date),
        y=[0] * len(squeeze_on_date),
        name="squeeze_on",
        mode="markers",
        marker=dict(color="black", size=5),
    )
    trace_squeeze_off = go.Scatter(
        x=list(squeeze_off_date),
        y=[0] * len(squeeze_off_date),
        name="squeeze_off",
        mode="markers",
        marker=dict(color="grey", size=5),
    )
    trace_squeeze = go.Bar(
        x=list(draw_df["date"]),
        y=list(squeeze_df["SQZ_20_2.0_20_2.0"]),
        name="squeeze",
        marker={
            "color": "yellow",
        },
    )

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=("OHLC", "Volume", "Squeeze"),
        row_width=[0.2, 0.2, 0.6],
        specs=[
            [{"secondary_y": True}],
            [{"secondary_y": False}],
            [{"secondary_y": False}],
        ],
    )
    fig.update_layout(title=code, xaxis_rangeslider_visible=False)
    fig.add_trace(candlestick, secondary_y=False, row=1, col=1)
    fig.add_trace(trace_close, secondary_y=False, row=1, col=1)
    fig.add_trace(trace_bbands_lower, secondary_y=False, row=1, col=1)
    fig.add_trace(trace_bbands_upper, secondary_y=False, row=1, col=1)
    fig.add_trace(trace_kc_lower, secondary_y=False, row=1, col=1)
    fig.add_trace(trace_kc_upper, secondary_y=False, row=1, col=1)
    fig.update_yaxes(secondary_y=False, showgrid=True, row=1, col=1)

    fig.add_trace(volume_bars, row=2, col=1)

    fig.add_trace(trace_squeeze_on, row=3, col=1)
    fig.add_trace(trace_squeeze_off, row=3, col=1)
    fig.add_trace(trace_squeeze, row=3, col=1)

    fig.show(config=plotly_config)


def plot_trading_record(start_date=None, end_date=None):
    trading_records = get_trading_record_dataframe()
    if start_date:
        trading_records = trading_records[trading_records["date"] >= start_date]
    if end_date:
        trading_records = trading_records[trading_records["date"] <= end_date]

    trading_records = trading_records.sort_values(by="date", ascending=False)
    for code in trading_records["code"].unique():
        fig = basic_plot_stock(code, start_date=datetime(2023, 10, 1))
        record = trading_records[trading_records["code"] == code]
        DOT_SIZE_BASE = 10
        # filter out buy & sell records
        buy_records = record[record["side"] == "buy"]
        trace_buy = go.Scatter(
            x=buy_records["date"],
            y=buy_records["price"],
            name=code,
            mode="markers+text",
            marker=dict(color="blue", symbol="arrow-up"),
            marker_size=DOT_SIZE_BASE,
            text=buy_records["volume"] / 1000,
            textposition="top left",
            textfont=dict(color="black"),
        )

        sell_records = record[record["side"] == "sell"]
        trace_sell = go.Scatter(
            x=sell_records["date"],
            y=sell_records["price"],
            name=code,
            mode="markers+text",
            marker=dict(color="red", symbol="arrow-down"),
            marker_size=DOT_SIZE_BASE,
            text=sell_records["volume"] / 1000,
            textposition="top right",
            textfont=dict(color="black"),
        )

        fig.add_trace(trace_buy, secondary_y=False, row=1, col=1)
        fig.add_trace(trace_sell, secondary_y=False, row=1, col=1)
        fig.show(config=plotly_config)

    print(trading_records)


def plot_strategy_one_signal(code, signals, start_date):
    fig = basic_plot_stock(code, start_date)
    price_df = get_daily_price_dataframe(code, start_date.year)
    draw_df = price_df

    for signal in signals:
        up_date, up_idx, signal_date, signal_idx = (
            signal["up_date"],
            signal["up_index"],
            signal["signal_date"],
            signal["signal_index"],
        )
        fig.add_trace(
            go.Scatter(
                x=[up_date, signal_date],
                y=[draw_df["high"].iloc[up_idx], draw_df["close"].iloc[signal_idx]],
                mode="lines",
                line=dict(color="blue", width=3),
            ),
            secondary_y=False,
            row=1,
            col=1,
        )

    fig.show(config=plotly_config)
