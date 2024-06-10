from pathlib import Path
from plotly.subplots import make_subplots
from typing import List

import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
import csv
import pandas as pd
import numpy as np

plots = []
class TradingRecordAnalyzer:
    def load_trading_record_df(result_dir: Path) -> list[pd.DataFrame, pd.DataFrame]:
        order_record_df =  pd.read_csv(result_dir / "order_record.csv", header=0, parse_dates=[1, 5, 6])
        account_record_df =  pd.read_csv(result_dir / "account_record.csv", header=0, parse_dates=[0])
        return order_record_df, account_record_df

    def to_trading_record_df(order_rows: List[any], account_rows: List[any]) -> list[pd.DataFrame, pd.DataFrame]:
        # order_column = ["code", "date", "volume", "amount", "profit_loss", "side", "buy_date", "cover_date", "buy_price", "cover_price", "return_rate", "holding_days", "avg_return_rate", "cover_reason"]
        order_dtype_mapper = ['object', 'datetime64[ns]', 'float64', 'float64', 'float64', 'object', 'datetime64[ns]', 'datetime64[ns]', 'float64', 'float64', 'float64', 'int64', 'float64', 'object']
        order_record_df = pd.DataFrame(order_rows[1:], columns=order_rows[0])
        order_record_df = order_record_df.astype(dict(zip(order_record_df.columns, order_dtype_mapper)))

        # acount_column = ["date", "cash", "holding_value", "realized_profit_loss", "book_account_profit_loss", "book_account_profit_loss_rate"]
        account_dtype_mapper = ['datetime64[ns]', 'float64', 'float64', 'float64', 'float64', 'float64']
        account_record_df = pd.DataFrame(account_rows[1:], columns=account_rows[0])
        account_record_df = account_record_df.astype(dict(zip(account_record_df.columns, account_dtype_mapper)))

        return order_record_df, account_record_df

    def analyze(result_dir: Path, order_rows: List[any] = None, account_rows: List[any] = None):
        if order_rows and account_rows:
            # read from input rows first
            order_record_df, account_record_df = TradingRecordAnalyzer.to_trading_record_df(order_rows=order_rows, account_rows=account_rows)
        elif result_dir:
            order_record_df, account_record_df = TradingRecordAnalyzer.load_trading_record_df(result_dir)
        else:
            raise ValueError("No result dir or rows provided")

        init_cash = account_record_df["cash"][0]
        account_result = TradingRecordAnalyzer.analyze_account_record(account_record_df)
        order_result = TradingRecordAnalyzer.analyze_order_record(order_record_df)

        with open(result_dir / "fig.html", "w") as f:
            for plot in plots:
                f.write(plot.to_html(full_html=False, include_plotlyjs='cdn'))
            plots.clear()

        return sum([account_result, order_result], [])

    
    def analyze_order_record(order_record_df: pd.DataFrame):
        cover_reason_counter = order_record_df.groupby("cover_reason").count().iloc[:, 0]
        cover_reason_counter_ratio = (cover_reason_counter / cover_reason_counter.sum()) * 100
        plot_cover_reason = px.pie(names=cover_reason_counter.index, values=cover_reason_counter_ratio, title="Cover Reason Ratio")

        # trading frequency by month
        order_record_count_by_month = order_record_df["buy_date"].groupby(order_record_df["buy_date"].dt.to_period("M")).count()
        plot_trading_freq = px.bar(x=order_record_count_by_month.index.to_timestamp(), y=order_record_count_by_month, title="Trading Frequency by Month")
        
        # avg profit loss
        avg_profit_loss = order_record_df["profit_loss"].mean()

        # plot distributions of return rate
        plot_result_distribution = ff.create_distplot([order_record_df["return_rate"]], ["return rate"], bin_size=0.2, show_rug=False)

        # group by order code and rank by profit loss
        total_profit_loss = order_record_df["profit_loss"].sum()
        order_record_df_by_code = order_record_df.groupby("code")
        order_record_df_by_code["profit_loss"].sum().sort_values(ascending=False)

        plots.extend([plot_cover_reason, plot_trading_freq, plot_result_distribution])
        return [f"{avg_profit_loss:.2f}"]



    def analyze_account_record(account_record_df: pd.DataFrame):
        # plot the realized profit loss
        plot_realized_profit_loss = px.line(
            x=account_record_df["date"],
            y=account_record_df["book_account_profit_loss"],
            title="Realized Profit Loss"
        )

        # count max draw down
        lowest_idx = np.argmax(np.maximum.accumulate(account_record_df["book_account_profit_loss"]) - account_record_df["book_account_profit_loss"])
        highest_idx = np.argmax(account_record_df["book_account_profit_loss"][:lowest_idx]) if lowest_idx > 0 else 0
        max_draw_down_rate = (account_record_df["book_account_profit_loss"][lowest_idx] - account_record_df["book_account_profit_loss"][highest_idx]) / account_record_df["book_account_profit_loss"][highest_idx] * 100
        
        # append max_draw_down on figure
        plot_realized_profit_loss.add_trace(
            go.Scatter(
                x=[account_record_df["date"][highest_idx], account_record_df["date"][lowest_idx]],
                y=[account_record_df["book_account_profit_loss"][highest_idx], account_record_df["book_account_profit_loss"][lowest_idx]],
                mode="lines",
                line=dict(color="red"),
            )
        )
        print(f"Max Draw Down: {max_draw_down_rate:.2f}%")

        # count return for every year
        account_record_df["realized_profit_loss_by_day"] = account_record_df["realized_profit_loss"].diff()
        account_record_df.loc[0, "realized_profit_loss_by_day"] = 0 # set first day profit loss to 0 to avoid nan
        account_record_df_by_year = account_record_df.groupby(account_record_df["date"].dt.year)
        init_account_value_of_year = account_record_df_by_year.first()["cash"] + account_record_df_by_year.first()["holding_value"]
        return_by_year = account_record_df_by_year["realized_profit_loss_by_day"].sum() / init_account_value_of_year * 100
        plot_return_rate = px.line(x=return_by_year.index, y=return_by_year, title="Return Rate by Year")

        plots.extend([plot_realized_profit_loss, plot_return_rate])
        return [f"{max_draw_down_rate:.2f}%"]


    def compare_trading_record(origin_result_dir: Path, target_result_dir: Path):
        origin_order_record_df, origin_account_record_df = TradingRecordAnalyzer.load_trading_record_df(origin_result_dir)
        target_order_record_df, target_account_record_df = TradingRecordAnalyzer.load_trading_record_df(target_result_dir)

        TradingRecordAnalyzer.compare_account_record(origin_account_record_df, target_account_record_df)
        TradingRecordAnalyzer.compare_order_record(origin_order_record_df, target_order_record_df)


    def compare_account_record(origin_account_record_df: pd.DataFrame, target_account_record_df: pd.DataFrame):
        # plot the realized profit loss
        plot_realized_profit_loss = make_subplots(rows=2, cols=1, subplot_titles=("Realized Profit Loss", "Realized Profit Loss"))
        plot_realized_profit_loss.add_trace(go.Scatter(x=origin_account_record_df["date"], y=origin_account_record_df["realized_profit_loss"], name="origin", mode="lines"), row=1, col=1)
    

    def compare_order_record(origin_order_record_df: pd.DataFrame, target_order_record_df: pd.DataFrame):
        origin_order_record_df['order_id'] = origin_order_record_df['code'].astype(str) + '-' + origin_order_record_df['date'].astype(str)
        target_order_record_df['order_id'] = target_order_record_df['code'].astype(str) + '-' + target_order_record_df['date'].astype(str)
        origin_set, target_set = set(origin_order_record_df['order_id']), set(target_order_record_df['order_id'])
        added, deleted = target_set - origin_set, origin_set - target_set
        added = target_order_record_df.loc[target_order_record_df["order_id"].isin(added)].drop(columns=["order_id"])
        deleted = origin_order_record_df.loc[origin_order_record_df["order_id"].isin(deleted)].drop(columns=["order_id"])