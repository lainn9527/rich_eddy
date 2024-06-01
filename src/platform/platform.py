from datetime import datetime
from enum import Enum
from typing import Dict
from pathlib import Path
import json


from src.broker.broker import Broker
from src.data_provider.base_provider import BaseProvider
from src.utils.common import Instrument, Market, OrderSide
from src.utils.order import Order
from src.utils.utils import NumpyEncoder

class DataProvider(Enum):
    OpenPrice = "open_price"
    HighPrice = "high_price"
    LowPrice = "low_price"
    ClosePrice = "close_price"


class Platform:
    broker_dict = dict()
    def __init__(
        self,
        broker_dict: Dict[str, BaseProvider],
    ) -> None:
        self.broker_dict = broker_dict

    def run(
        self,
        strategy,
        start_date: datetime = None,
        end_date: datetime = None,
        result_path: Path = Path("result") / datetime.now().strftime("%Y%m%d_%H%M%S"),
        write_analyze_material: bool = False,
    ) -> None:
        analyze_material = strategy.prepare_data(start_date, end_date)
        strategy.ensure_data()
        trading_dates = strategy.get_trading_dates()

        for trading_date in trading_dates:
            strategy.step_data(trading_date)
            strategy.step(trading_date)
            strategy.step_end(trading_date)

        strategy.end(result_path)

        if write_analyze_material and analyze_material is not None:
            with open(result_path / "analyze_material.json", "w") as fp:
                json.dump(analyze_material, fp, cls=NumpyEncoder)


    def place_order(
        self,
        market: Market,
        instrument: Instrument,
        code: str,
        time: datetime,
        price: float,
        volume: float,
        side: OrderSide,
        is_cover: bool,
    ) -> Order:
        broker = self.get_broker(market, instrument)
        return broker.create_order(market, instrument, code, time, price, volume, side, is_cover)


    def get_broker(self, market: Market, instrument: Instrument) -> Broker:
        return self.broker_dict['broker']
