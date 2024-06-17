from datetime import datetime, timedelta
from typing import Dict, List
from collections import OrderedDict
from pathlib import Path
from functools import wraps
import csv
import numpy as np
import json

from src.data_provider.base_provider import BaseProvider
from src.data_store.data_store import DataStore
from src.platform.platform import Platform
from src.utils.common import DataCategory, Instrument, Market, OrderSide, DataColumn, TechnicalIndicator, TradingResultColumn
from src.utils.order import OrderRecord, Order
from src.data_analyzer.trading_record_analyzer import TradingRecordAnalyzer

def filter_decorator(filter_func):
    @wraps(filter_func)
    def wrapper(*args, **kwargs):
        self = args[0]
        if filter_func.__name__ not in self.activated_filters.keys() or not self.activated_filters[filter_func.__name__]:
            # not filtered
            return False
        
        # filter function is the condition to pass the filter
        # return value should be "is filtered or not", so add "not"
        pass_filter = filter_func(*args, **kwargs)
        if not pass_filter:
            self.filtered_signal_count[filter_func.__name__] += 1
            if "x" in kwargs and "y" in kwargs:
                self.filtered_reason[kwargs["x"], kwargs["y"]] = self.filtered_reason_mapper[filter_func.__name__]
        
        return not pass_filter

    return wrapper

class Strategy:
    data_provider_dict: Dict[str, BaseProvider]
    data_store: DataStore
    platform: Platform
    log_level: str
    config: Dict[str, any]
    preserved_data: Dict[str, any]

    cash: float
    account_history: List[dict]
    holdings: Dict[str, OrderRecord]
    realized_profit_loss: float

    order_record_dict: Dict[str, OrderRecord]
    history_order_by_day: Dict[datetime, List[OrderRecord]]

    data_dates: List[datetime]
    trading_dates: List[datetime]
    trading_codes: List[str]
    current_trading_date: datetime
    
    activated_filters: Dict[str, bool] = {}
    filtered_signal_count: Dict[str, int] = {}
    filtered_reason: np.ndarray
    filtered_reason_mapper = {}

    def __init__(
        self,
        platform: Platform,
        data_store: DataStore,
        cash: float,
        config: Dict[str, any],
        log_level: str = "INFO",
    ):
        self.platform = platform
        self.data_store = data_store
        self.preserved_data = dict()
        self.config = config
        self.log_level = log_level
        
        self.initial_cash = cash
        self.cash = cash
        self.account_history = list()
        self.current_trading_date = None

        self.holdings = dict()
        self.order_record_dict = dict()
        self.history_order_by_day = dict()
        self.realized_profit_loss = 0

        self.trading_dates = None
        self.trading_codes = None

        self.activated_filters = config.get("activated_filters", {})
        self.decorate_filter()


    def prepare_data(self, start_date: datetime, end_date: datetime):
        pass


    def ensure_data(self):
        data_list = vars(self)
        for key, value in data_list.items():
            if not key.endswith("_"):
                continue
            try:
                assert len(value) == len(self.get_trading_dates())
            except:
                print(f"Data shape mismatch: {key} with shape {len(value)} vs {len(self.get_trading_dates())}")
                exit(1)
            # store original data
            self.preserved_data[key] = value
        
        assert self.get_trading_dates() != None
        assert self.get_trading_codes() != None


    def slice_data(self, all_dates: List[datetime], start_date: datetime, end_date: datetime):
        if end_date > all_dates[-1]:
            end_date = all_dates[-1]

        start_trading_date = next(trading_date for trading_date in all_dates if trading_date >= start_date)
        end_trading_date = next(trading_date for trading_date in all_dates if trading_date >= end_date)
        start_idx = all_dates.index(start_trading_date)
        end_idx = all_dates.index(end_trading_date)

        data_list = vars(self)
        for key, value in data_list.items():
            if not key.endswith("_"):
                continue
            data_list[key] = value[start_idx:end_idx+1]
        
        return all_dates[start_idx:end_idx+1]
        

    def step_data(self, trading_date: datetime):
        data_list = vars(self)
        trading_date_idx = self.trading_dates.index(trading_date)
        for key in data_list:
            if not key.endswith("_"):
                continue
            data_list[key] = self.preserved_data[key][:trading_date_idx+1]


    def step(self, trading_date: datetime):
        # for each iteration, the data will be truncated to the :trading_date
        self.current_trading_date = trading_date

        # remove out of market holding
        for order_record in list(self.holdings.values()):
            code = order_record.order.code
            code_idx = self.get_trading_codes().index(code)
            if np.isnan(self.close_[-1, code_idx]):
                self.cover_order(
                    order_record.order.order_id,
                    self.close_[-2, code_idx],
                    order_record.order.volume,
                    'out_of_market'
                )
        self.update_book_profit_loss(trading_date)

    def step_end(self, trading_date: datetime):
        # update for new orders
        self.update_book_profit_loss(trading_date)

        # calculate book profit loss
        book_holding_value = sum([order_record.book_value for order_record in self.holdings.values()])
        book_account_profit_loss = (self.cash + book_holding_value) - self.initial_cash
        book_account_profit_loss_rate = (book_account_profit_loss / self.initial_cash) * 100

        self.account_history.append({
            "date": trading_date,
            "cash": self.cash,
            "realized_profit_loss": self.realized_profit_loss,
            "book_holding_value": book_holding_value,
            "book_account_profit_loss": book_account_profit_loss,
            "book_account_profit_loss_rate": book_account_profit_loss_rate,
        })


    def cover_order(self, order_id, price, volume, reason=None):
        order_record = self.order_record_dict[order_id]
        source_order = order_record.order

        if source_order.side == OrderSide.Buy:
            order_side = OrderSide.Sell
            position = -volume
        else:
            order_side = OrderSide.Buy
            position = volume

        cover_order: Order = self.platform.place_order(
            source_order.market,
            source_order.instrument,
            source_order.code,
            self.current_trading_date,
            price,
            volume,
            order_side,
            True
        )
        net_profit_loss = cover_order.execute_value - cover_order.transaction_cost
        if np.isnan(net_profit_loss):
            raise Exception("Invalid order")
        self.cash += net_profit_loss

        # update order record
        order_record.net_profit_loss += net_profit_loss
        order_record.open_position += position
        order_record.cover_order.append(cover_order)
        order_record.cover_reason = reason

        if order_record.open_position == 0:
            net_profit_loss_rate = (order_record.net_profit_loss / order_record.cost) * 100
            order_record.is_covered = True
            order_record.net_profit_loss_rate = net_profit_loss_rate
            self.holdings.pop(order_id)

        # record to realized profit loss
        self.realized_profit_loss += net_profit_loss - (order_record.cost / source_order.volume) * volume

        if self.log_level == "DETAIL":
            print((
                f"{self.current_trading_date.date()} cover order: "
                f"{order_side.value} {source_order.volume} {source_order.code} "
                f"with ${source_order.execute_price}, "
                f"earn ${order_record.net_profit_loss:.2f}, "
                f"{(order_record.net_profit_loss/order_record.cost)*100:.2f}%, "
                f"reason: {reason}"
            ))


    def place_order(
        self,
        market: Market,
        instrument: Instrument,
        code: str,
        price: float,
        volume: float,
        side: OrderSide,
        info: Dict = None,
    ) -> None:
        order: Order = self.platform.place_order(
            market,
            instrument,
            code,
            self.current_trading_date,
            price,
            volume,
            side,
            False,
        )
        net_profit_loss = -(order.execute_value + order.transaction_cost)
        if np.isnan(net_profit_loss):
            raise Exception("Invalid order")

        if self.cash + net_profit_loss < 0:
            raise Exception("Insufficient fund")

        self.cash += net_profit_loss

        # add order record
        open_position = order.volume if order.side == OrderSide.Buy else -order.volume
        order_record = OrderRecord(
            order=order,
            open_position=open_position,
            cost=order.execute_value + order.transaction_cost,
            net_profit_loss=net_profit_loss,
            info=info,
            book_price=price,
        )

        self.holdings[order.order_id] = order_record
        self.order_record_dict[order.order_id] = order_record

        # add daily order
        self.append_daily_order_record(self.current_trading_date, order_record)

        if self.log_level == "DETAIL":
            print(f"{self.current_trading_date.date()} place order: {order.side.value} {order.volume} {order.code} with ${order.execute_price}")


    def update_book_profit_loss(self, trading_date: datetime):
        # update book profit loss
        for order_record in self.holdings.values():
            if order_record.is_covered:
                continue

            source_order = order_record.order
            # get current price
            close_price = self.close_[-1][self.trading_codes.index(order_record.order.code)]
            cover_order = self.platform.place_order(
                source_order.market,
                source_order.instrument,
                source_order.code,
                trading_date,
                close_price,
                order_record.open_position,
                OrderSide.Sell if source_order.side == OrderSide.Buy else OrderSide.Buy,
                True,
            )
            net_profit_loss = cover_order.execute_value - cover_order.transaction_cost
            book_value = net_profit_loss
            book_profit_loss = order_record.net_profit_loss + net_profit_loss
            book_profit_loss_rate = (book_profit_loss / order_record.cost) * 100
            
            order_record.book_price = close_price
            order_record.book_value = book_value
            order_record.book_profit_loss = book_profit_loss
            order_record.book_profit_loss_rate = book_profit_loss_rate


    def append_daily_order_record(self, trading_date: datetime, order_record: OrderRecord):
        if self.history_order_by_day.get(trading_date) is None:
            self.history_order_by_day[trading_date] = []

        self.history_order_by_day[trading_date].append(order_record)


    def get_trading_dates(self):
        return self.trading_dates


    def get_trading_codes(self):
        return self.trading_codes


    def clear_holdings(self):
        for order_record_id in list(self.holdings):
            order_record = self.holdings[order_record_id]
            self.cover_order(
                order_record_id,
                order_record.book_price,
                order_record.open_position,
                'end'
            )


    def end(self, result_dir: Path, full_record):
        used_cash = self.initial_cash - min([account["cash"] for account in self.account_history])
        if used_cash == 0:
            print("Zero trading record")
            return

        # since we don't know how much cash our strategy need, we set a large number on initial cash
        # we have to use our real used cash to calculate the return
        for account in self.account_history:
            account["cash"] = account["cash"] - self.initial_cash + used_cash
            account["book_account_profit_loss_rate"] = (account["book_account_profit_loss"] / used_cash) * 100

        final_account_info = self.account_history[-1]
        final_profit_loss, final_return = final_account_info["book_account_profit_loss"], final_account_info["book_account_profit_loss_rate"]
        annualized_return = ((1 + final_profit_loss/used_cash) ** (1/len(self.trading_dates)) - 1) * 250 * 100

        print(f"Used cash: {used_cash}")
        print(f"Net Profit Loss: {final_profit_loss:2f}")
        print(f"Total Return: {final_return:.2f}%")
        print(f"Annualized Return: {annualized_return:.2f}%")
        print(f"Trading records: {len(self.order_record_dict)}")

        # clear order
        self.clear_holdings()

        # write order record
        order_records = list(self.order_record_dict.values())
        sorted_order_records = sorted(order_records, key=lambda x: x.order.execute_time)
        order_record_rows = [TradingResultColumn.trading_record]
        for order_record in sorted_order_records:
            for cover_order in order_record.cover_order:
                order_record_rows.append([
                    order_record.order.code,
                    cover_order.execute_time.date(),
                    order_record.order.volume,
                    order_record.cost,
                    f"{order_record.net_profit_loss:.2f}",
                    order_record.order.side,
                    order_record.order.execute_time.date(),
                    cover_order.execute_time.date(),
                    order_record.order.execute_price,
                    f"{cover_order.execute_price:.2f}",
                    f"{order_record.net_profit_loss_rate:.2f}",
                    (cover_order.execute_time - order_record.order.execute_time).days,
                    f"{order_record.net_profit_loss_rate / ((cover_order.execute_time - order_record.order.execute_time).days+1):.2f}",
                    order_record.cover_reason,
                ])

        # account history
        account_record_rows = [
            TradingResultColumn.account_record,
            [(self.trading_dates[0]-timedelta(days=1)).date(), used_cash, 0, 0, 0, 0], # add an initial record
        ]
        for account in self.account_history:
            account_record_rows.append([
                account["date"].date(),
                f"{account['cash']:.2f}",
                f"{account['book_holding_value']:.2f}",
                f"{account['realized_profit_loss']:.2f}",
                f"{account['book_account_profit_loss']:.2f}",
                f"{account['book_account_profit_loss_rate']:.2f}",
            ])

        summary_result = [result_dir.parts[-1], int(used_cash), int(final_profit_loss), f"{final_return:.2f}%", f"{annualized_return:.2f}%", len(self.order_record_dict)]
        result_dir.mkdir(parents=True, exist_ok=True)
        with open(result_dir / "config.json", "w") as fp:
            json.dump(self.config, fp)

        if full_record:
            with open(result_dir / "order_record.csv", "w") as fp:
                writer = csv.writer(fp)
                writer.writerows(order_record_rows)
            
            with open(result_dir / "account_record.csv", "w") as fp:
                writer = csv.writer(fp)
                writer.writerows(account_record_rows)

        summary_result.extend(TradingRecordAnalyzer.analyze(result_dir=result_dir, order_rows=order_record_rows, account_rows=account_record_rows))
        summary_result_path = result_dir.parent / "result.csv"
        if not summary_result_path.exists():
            with open(summary_result_path, "w") as fp:
                writer = csv.writer(fp)
                writer.writerow(TradingResultColumn.summary_result)

        with open(summary_result_path, "a+") as fp:
            writer = csv.writer(fp)
            writer.writerow(summary_result)


    def decorate_filter(self):
        filter_prefix = "filter_"
        for i, filter_func in enumerate(self.__class__.__dict__.values()):
            if callable(filter_func) and filter_func.__name__.startswith(filter_prefix):
                setattr(self.__class__, filter_func.__name__, filter_decorator(filter_func))
                self.filtered_signal_count[filter_func.__name__] = 0
                self.filtered_reason_mapper[filter_func.__name__] = i