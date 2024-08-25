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
    MARKET_TYPE = "market_type"

    # financial data
    EPS = "eps"
    Recurring_EPS = "recurring_eps"

    # chip
    Foreign_Buy_Volume="foreign_buy_volume"
    Foreign_Sell_Volume="foreign_sell_volume"
    Foreign_Net_Volume="foreign_net_volume"
    Foreign_Buy_Amount="foreign_buy_amount"
    Foreign_Sell_Amount="foreign_sell_amount"
    Local_Investor_Buy_Volume="local_investor_buy_volume"
    Local_Investor_Sell_Volume="local_investor_sell_volume"
    Local_Investor_Net_Volume="local_investor_net_volume"
    Local_Investor_Buy_Amount="local_investor_buy_amount"
    Local_Investor_Sell_Amount="local_investor_sell_amount"
    Local_Self_Buy_Volume="local_self_buy_volume"
    Local_Self_Sell_Volume="local_self_sell_volume"
    Local_Self_Net_Volume="local_self_net_volume"
    Local_Self_Buy_Amount="local_self_buy_amount"
    Local_Self_Sell_Amount="local_self_sell_amount"
    Total_Investor_Buy_Volume="total_investor_buy_volume"
    Total_Investor_Sell_Volume="total_investor_sell_volume"
    Total_Investor_Net_Volume="total_investor_net_volume"
    Foreign_Trading_Ratio="foreign_trading_ratio"
    Local_Investor_Trading_Ratio="local_investor_trading_ratio"
    Local_Self_Trading_Ratio="local_self_trading_ratio"
    Total_Investor_Trading_Ratio="total_investor_trading_ratio"
    Foreign_Total_Holdings_Ratio="foreign_total_holdings_ratio"
    Local_Investor_Holdings_Ratio="local_investor_holdings_ratio"
    Local_Self_Holdings_Ratio="local_self_holdings_ratio"
    Foreign_Total_Holdings="foreign_total_holdings"
    Local_Investor_Holdings="local_investor_holdings"
    Local_Self_Holdings="local_self_holdings"
    Director_Supervisor_Holdings_Ratio="director_supervisor_holdings_ratio"
    Director_Supervisor_Pledge_Ratio="director_supervisor_pledge_ratio"
    Director_Supervisor_Holdings="director_supervisor_holdings"

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
        DataColumn.MARKET_TYPE,
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
        DataColumn.Foreign_Buy_Volume,
        DataColumn.Foreign_Sell_Volume,
        DataColumn.Foreign_Net_Volume,
        DataColumn.Foreign_Buy_Amount,
        DataColumn.Foreign_Sell_Amount,
        DataColumn.Local_Investor_Buy_Volume,
        DataColumn.Local_Investor_Sell_Volume,
        DataColumn.Local_Investor_Net_Volume,
        DataColumn.Local_Investor_Buy_Amount,
        DataColumn.Local_Investor_Sell_Amount,
        DataColumn.Local_Self_Buy_Volume,
        DataColumn.Local_Self_Sell_Volume,
        DataColumn.Local_Self_Net_Volume,
        DataColumn.Local_Self_Buy_Amount,
        DataColumn.Local_Self_Sell_Amount,
        DataColumn.Total_Investor_Buy_Volume,
        DataColumn.Total_Investor_Sell_Volume,
        DataColumn.Total_Investor_Net_Volume,
        DataColumn.Foreign_Trading_Ratio,
        DataColumn.Local_Investor_Trading_Ratio,
        DataColumn.Local_Self_Trading_Ratio,
        DataColumn.Total_Investor_Trading_Ratio,
        DataColumn.Foreign_Total_Holdings_Ratio,
        DataColumn.Local_Investor_Holdings_Ratio,
        DataColumn.Local_Self_Holdings_Ratio,
        DataColumn.Foreign_Total_Holdings,
        DataColumn.Local_Investor_Holdings,
        DataColumn.Local_Self_Holdings,
        DataColumn.Director_Supervisor_Holdings_Ratio,
        DataColumn.Director_Supervisor_Pledge_Ratio,
        DataColumn.Director_Supervisor_Holdings,
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


class TradingResultColumn:
    summary_result = ['name', 'used_cash', 'final_profit_loss', 'final_return', 'annualized_return', '#_trading_records']
    trading_record = ["code", "date", "volume", "amount", "profit_loss", "side", "buy_date", "cover_date", "buy_price", "cover_price", "return_rate", "holding_days", "avg_return_rate", "cover_reason"]
    account_record = ["date", "cash", "holding_value", "realized_profit_loss", "book_account_profit_loss", "book_account_profit_loss_rate"]

class ColumnValueMapper:
    mapper = {
        DataColumn.MARKET_TYPE: {
            "TSE": 0,
            "OTC": 1,
            "REG": 2,
            "PSB": 3,
            "TIB": 4
        }
    }
    @staticmethod
    def get_column_value_mapper(column_name) -> dict:
        if column_name == DataColumn.MARKET_TYPE:
            return ColumnValueMapper.mapper[DataColumn.MARKET_TYPE]
        return None
    @staticmethod
    def get_reversed_column_value_mapper(column_name) -> dict:
        return {v: k for k, v in ColumnValueMapper.get_column_value_mapper(column_name).items()}
        

FILTER_PARAMETER_MAPPER = {
    "filter_up_min_ratio": ["up_min_ratio"],
    "filter_down_max_ratio": True,
    "filter_breakthrough_point": True,
    "filter_consolidation_time_window": True,
    "filter_relative_strength": True,
    "filter_market_index": True,
    "filter_chip": True,
    "filter_volume": True,
    "filter_signal_threshold": True,
}
