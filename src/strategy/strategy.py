from datetime import datetime
from typing import Dict, List
from collections import OrderedDict
import csv
import numpy as np

from src.data_provider.base_provider import BaseProvider
from src.data_store.data_store import DataStore
from src.platform.platform import Platform
from src.utils.common import DataCategory, Instrument, Market, OrderSide, DataColumn, TechnicalIndicator
from src.utils.order import OrderRecord, Order

class Strategy:
    data_provider_dict: Dict[str, BaseProvider]
    data_store: DataStore
    platform: Platform
    log_level: str
    config: Dict[str, any]
    preserved_data: Dict[str, any]

    cash: float
    cash_history: List[float]
    holdings: Dict[str, OrderRecord]

    order_record_dict: Dict[str, OrderRecord]
    history_order_by_day: Dict[datetime, List[OrderRecord]]

    data_dates: List[datetime]
    trading_dates: List[datetime]
    trading_codes: List[str]
    current_trading_date: datetime
    
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

        self.cash = cash
        self.cash_history = list()
        self.current_trading_date = None

        self.holdings = dict()
        self.order_record_dict = dict()
        self.history_order_by_day = dict()

        self.trading_dates = None
        self.trading_codes = None


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
        self.cash_history.append(self.cash)

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


    
    def update_book_value(self):
        for order_id, order_record in self.holdings.items():
            order = order_record.order
            current_price = self.data_store.get_data_item(order.market, order.instrument, DataCategory.Daily_Price, DataColumn.Close, order.code, self.current_trading_date)
            order_record.book_profit_loss = current_price * order_record.open_position
            order_record.book_profit_loss_rate = order_record.book_profit_loss / (order.execute_price * order.volume)


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
        )

        self.holdings[order.order_id] = order_record
        self.order_record_dict[order.order_id] = order_record

        # add daily order
        self.append_daily_order_record(self.current_trading_date, order_record)

        if self.log_level == "DETAIL":
            print(f"{self.current_trading_date.date()} place order: {order.side.value} {order.volume} {order.code} with ${order.execute_price}")


    def append_daily_order_record(
        self, trading_date: datetime, order_record: OrderRecord
    ):
        if self.history_order_by_day.get(trading_date) is None:
            self.history_order_by_day[trading_date] = []

        self.history_order_by_day[trading_date].append(order_record)


    def get_trading_dates(self):
        return self.trading_dates


    def get_trading_codes(self):
        return self.trading_codes


    def end(self):
        # clear remaining holdings
        for order_id in list(self.holdings):
            order_record = self.holdings[order_id]
            self.cover_order(
                order_id,
                order_record.order.execute_price,
                order_record.open_position,
                'end'
            )
        used_cash = self.cash_history[0] - min(self.cash_history)
        net_profit_loss = self.cash_history[-1] - self.cash_history[0]
        total_return = net_profit_loss / used_cash * 100
        annualized_return = ((1 + net_profit_loss/used_cash) ** (1/len(self.trading_dates)) - 1) * 250 * 100

        print(f"Lowest cash: {min(self.cash_history)}")
        print(f"Net Profit Loss: {net_profit_loss:2f}")
        print(f"Total Return: {total_return:.2f}%")
        print(f"Annualized Return: {annualized_return:.2f}%")

        # write to csv
        order_records = list(self.order_record_dict.values())
        sorted_order_records = sorted(order_records, key=lambda x: x.order.execute_time)
        data_rows = [["股票", "日期", "股數", "損益", "交易別", "買進日期", "賣出日期", "買進單價", "賣出單價", "報酬率", "持有天數", "平均報酬", "出場理由"]]
        for order_record in sorted_order_records:
            for cover_order in order_record.cover_order:
                data_rows.append([
                    order_record.order.code,
                    cover_order.execute_time.date(),
                    order_record.order.volume,
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
        with open("result.csv", "w") as fp:
            writer = csv.writer(fp)
            writer.writerows(data_rows)
            
