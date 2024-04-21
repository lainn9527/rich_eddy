from pathlib import Path
from plotly.subplots import make_subplots

import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
import csv
import pandas as pd


plots = []
class TradingRecordAnalyzer:
    def load_trading_record_df(result_dir: Path) -> list[pd.DataFrame, pd.DataFrame]:
        order_record_df =  pd.read_csv(result_dir / "order_record.csv", header=0, parse_dates=[1, 5, 6])
        account_record_df =  pd.read_csv(result_dir / "account_record.csv", header=0, parse_dates=[0])
        return order_record_df, account_record_df

    def analyze(result_dir: Path):
        order_record_df, account_record_df = TradingRecordAnalyzer.load_trading_record_df(result_dir)
        init_cash = account_record_df["cash"][0]

        TradingRecordAnalyzer.analyze_account_record(account_record_df)
        TradingRecordAnalyzer.analyze_order_record(order_record_df)

        with open(result_dir / "fig.html", "w") as f:
            for plot in plots:
                f.write(plot.to_html(full_html=False, include_plotlyjs='cdn'))
            plots.clear()

    
    def analyze_order_record(order_record_df: pd.DataFrame):
        cover_reason_counter = order_record_df.groupby("cover_reason").count().iloc[:, 0]
        cover_reason_counter_ratio = (cover_reason_counter / cover_reason_counter.sum()) * 100
        plot_cover_reason = px.pie(names=cover_reason_counter.index, values=cover_reason_counter_ratio, title="Cover Reason Ratio")

        # trading frequency by month
        order_record_count_by_month = order_record_df["buy_date"].groupby(order_record_df["buy_date"].dt.to_period("M")).count()
        plot_trading_freq = px.bar(x=order_record_count_by_month.index.to_timestamp(), y=order_record_count_by_month, title="Trading Frequency by Month")
        
        # plot distributions of return rate
        plot_result_distribution = ff.create_distplot([order_record_df["return_rate"]], ["return rate"], bin_size=0.2, show_rug=False)

        plots.extend([plot_cover_reason, plot_trading_freq, plot_result_distribution])


    def analyze_account_record(account_record_df: pd.DataFrame):
        # count return for every year
        account_record_df["realized_profit_loss_by_day"] = account_record_df["realized_profit_loss"].diff()
        account_record_df.loc[0, "realized_profit_loss_by_day"] = 0 # set first day profit loss to 0 to avoid nan
        account_record_df_by_year = account_record_df.groupby(account_record_df["date"].dt.year)
        init_account_value_of_year = account_record_df_by_year.first()["cash"] + account_record_df_by_year.first()["holding_value"]
        return_by_year = account_record_df_by_year["realized_profit_loss_by_day"].sum() / init_account_value_of_year * 100
        plot_return_rate = px.line(x=return_by_year.index, y=return_by_year, title="Return Rate by Year")

        # count max draw down
        max_draw_down = account_record_df["book_account_profit_loss_rate"].min()

        print(f"Max Draw Down: {max_draw_down:.2f}%")

        plots.append(plot_return_rate)
