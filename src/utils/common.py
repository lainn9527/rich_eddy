from enum import Enum
from typing import List

class DataProvider(Enum):
    DailyPriceProvider = "daily_price_provider"


class Market(Enum):
    TW = "tw"
    US = "us"


class Instrument(Enum):
    Stock = "stock"
    StockIndex = "stock_index"
    StockFuture = "stock_future"
    StockOption = "stock_option"
    CommodityFuture = "commodity_future"
    Index = "index"


class TimeFrame(Enum):
    Tick = "tick"
    Minute = "minute"
    Hourly = "hourly"
    Daily = "daily"
    Weekly = "weekly"
    Monthly = "monthly"
    Quarterly = "quarterly"
    Yearly = "yearly"


class DataCategory(Enum):
    # daily price
    Daily_Price = "daily_price"

    # infra price
    Hourly_Price = "hourly_price"
    Minute_Price = "minute_price"
    
    # financial data
    Finance_Report = "finance_report"
    
    # chip
    Chip = "chip"

    # market index
    Market_Index = "market_index"

class DataColumn(Enum):
    # meta
    Code = "code"
    Date = "date"
    Quarter = "quarter"

    # daily price
    Open = "open"
    High = "high"
    Low = "low"
    Close = "close"
    Volume = "volume"
    Trading_Value = "trading_value"
    Total_Stocks = "total_stocks"

    # financial data
    EPS = "eps"
    Recurring_EPS = "recurring_eps"

    # chip
    Foreign_Volume = "foreign_volume"
    Local_Investor_Volume = "local_investor_volume"
    Local_Self_Volume = "local_self_volume"

class DataCategoryColumn:
    daily_price = [
        DataColumn.Code,
        DataColumn.Date,
        DataColumn.Open,
        DataColumn.High,
        DataColumn.Low,
        DataColumn.Close,
        DataColumn.Volume,
        DataColumn.Trading_Value,
        DataColumn.Total_Stocks,
    ]

    finance_report = [
        DataColumn.Code,
        DataColumn.Date,
        DataColumn.Quarter,
        DataColumn.EPS,
        DataColumn.Recurring_EPS,
    ]

    chip = [
        DataColumn.Code,
        DataColumn.Date,
        DataColumn.Foreign_Volume,
        DataColumn.Local_Investor_Volume,
        DataColumn.Local_Self_Volume,
    ]

    market_index = [
        DataColumn.Code,
        DataColumn.Date,
        DataColumn.Open,
        DataColumn.High,
        DataColumn.Low,
        DataColumn.Close,
        DataColumn.Volume,
    ]

    @staticmethod
    def get_columns(data_category: DataCategory) -> List[DataColumn]:
        return DataCategoryColumn.__dict__[data_category.value]

class OrderSide(Enum):
    Buy = "buy"
    Sell = "sell"

class TechnicalIndicator(Enum):
    SMA = "sma"
    RS = "relative_strength"

class MarketIndexCode(Enum):
    TW50 = "tw50"
    TPEX = "tpex"
    SP500 = "sp500"
    NASDAQ = "nasdaq"
    DJI = "dji"
