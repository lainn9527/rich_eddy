from dataclasses import dataclass, field
from datetime import datetime
from typing import List
from src.utils.common import Instrument, Market, OrderSide


@dataclass
class Order:
    order_id: str
    market: Market
    instrument: Instrument
    code: str
    
    side: OrderSide
    place_price: float
    place_time: datetime
    volume: float

    execute_time: datetime = None
    execute_price: float = None
    execute_value: float = None
    transaction_cost: float = None


@dataclass
class OrderRecord:
    order: Order
    open_position: float = None

    cover_order: List[Order] = field(default_factory=list) 
    is_covered: bool = False
    cover_reason: str = None

    cost: float = 0
    book_profit_loss: float = 0
    book_profit_loss_rate: float = 0
    net_profit_loss: float = 0
    net_profit_loss_rate: float = 0
    info: dict = None


@dataclass
class CoverOrderRecord:
    order: Order
    covered_order: Order
