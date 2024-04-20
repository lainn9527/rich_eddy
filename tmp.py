from pathlib import Path
import talib
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
# from src.data_provider.daily_price_provider import DailyPriceProvider
from src.data_processor.tej_daily_data_processor import TejDailyDataProcessor
from src.data_processor.tej_market_index_processor import TejMarketIndexProcessor
# from src.utils import time_profiler
from src.data_store.data_store import DataStore, DataColumn
from src.utils.common import DataCategory, Market, Instrument, TimeFrame, TechnicalIndicator, DataCategoryColumn
from src.data_processor.tej_quarter_finance_processor import TejQuarterFinanceProcessor
from src.strategy.strategy import Strategy
from src.data_transformer.data_transformer import DataTransformer


date_data_dir = Path("data/daily_price/date_from_2024")
code_data_dir = Path("data/daily_price/code_from_2024")
meta_data_path = Path("data/stock_meta.csv")


# def main():
#     dp = DailyPriceProvider(
#         date_data_dir=date_data_dir,
#         code_data_dir=code_data_dir,
#         meta_data_path=meta_data_path,
#         unit="day",
#         lazy_loading=True,
#     )
#     dp.load_date_data()
#     dp.load_code_data()
#     dp.build_column_np_array()


# wrapped = time_profiler(main)
# wrapped()


# def main():
#     data_store = DataStore()
#     # tw50 = data_store.get_data(market=Market.TW, instrument=Instrument.StockIndex, data_category=DataCategory.Market_Index, data_columns=[DataColumn.Close])
#     close = data_store.get_data(market=Market.TW, instrument=Instrument.Stock, data_category=DataCategory.Daily_Price, data_columns=[DataColumn.Close])
#     # eps = data_store.get_data(market=Market.TW, instrument=Instrument.Stock, data_category=DataCategory.Finance_Report, data_columns=[DataColumn.EPS])
#     # dates = data_store.get_data_date(market=Market.TW, instrument=Instrument.Stock, data_category=DataCategory.Daily_Price, data_column=DataColumn.Close)
#     # codes = data_store.get_data_code(market=Market.TW, instrument=Instrument.Stock, data_category=DataCategory.Daily_Price, data_column=DataColumn.Close)
#     # f = data_store.get_aligned_data(target_dates=dates, target_codes=codes, market=Market.TW, instrument=Instrument.Stock, data_category=DataCategory.Finance_Report, data_columns=[DataColumn.EPS])    
    
#     # print(data_store.get_technical_indicator(TechnicalIndicator.SMA, close, 5))
#     # close: (date, code)
#     # code_data = data_store.get_data_by_code(market=Market.TW, instrument=Instrument.Stock, data_category=DataCategory.Daily_Price, data_column=DataColumn.Close, codes=["2330"])
#     # print(code_data)
#     # # print(DataCategoryColumn.daily_price[0] == DataColumn.Code.value)
#     print(close)
#     data_store
    


def main():
    TejMarketIndexProcessor.transform_raw_data_to_date_data(
        DataCategory.Daily_Price,
        Path("data/market_index/raw"),
        Path("data/market_index/")
    )
    TejMarketIndexProcessor.transform_date_data_to_code_data(
        Path("data/market_index/date"),
        Path("data/market_index/code")
    )
    # TejDailyDataProcessor.transform_date_data_to_code_data(
    #     DataCategory.Daily_Price,
    #     Path("data/market_index/date_from_2024"),
    #     Path("data/daily_price/test_code")
    # )
    # TejDailyDataProcessor.transform_date_data_to_code_data(
    #     Path("data/daily_price/date"),
    #     Path("data/daily_price/code")
    # )

def main():
    np_array = np.array([[1, 4, 7, 4, 5], [6, 2, 8, 9, 10], [3, 12, 13, 5, 3], [5, 2, 1, 6, 2]])
    np_array = np.array([[6, 2, 8, 9, 10], [3, 12, 13, 5, 3], [1, 4, 7, 4, 5], [5, 2, 1, 6, 2]])


main()