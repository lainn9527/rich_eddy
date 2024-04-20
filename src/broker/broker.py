from dataclasses import dataclass
from datetime import datetime

from src.utils.order import Order
from src.utils.common import OrderSide, Market, Instrument

class Broker:
    def __init__(self) -> None:
        self.transaction_tax_rate = 0.003
        self.transaction_fee_rate = 0.001425 # original
        self.transaction_fee_discount = 0.3
        self.broker_fee = 40
        self.order_counter = 0

    def create_order(
            self,
            market: Market,
            instrument: Instrument,
            code: str,
            time: datetime,
            price: float,
            volume: float,
            side: OrderSide,
            is_cover: bool
        ) -> Order:

        product_cost = volume * price
        transaction_fee = volume * price * self.transaction_fee_rate * self.transaction_fee_discount
        transaction_tax = volume * price * self.transaction_tax_rate if is_cover else 0
        transaction_cost = transaction_fee + transaction_tax
        
        self.order_counter += 1
        return Order(
            order_id=f"{market}-{instrument}-{self.order_counter:05}",
            market=market,
            instrument=instrument,
            code=code,
            side=side,
            place_price=price,
            place_time=time,
            volume=volume,
            execute_time=time,
            execute_price=price,
            execute_value=product_cost,
            transaction_cost=transaction_cost,
        )
