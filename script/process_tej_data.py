from pathlib import Path
import talib
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
# from src.data_provider.daily_price_provider import DailyPriceProvider
from src.data_processor.tej_daily_data_processor import TejDailyDataProcessor
from src.data_processor.tej_market_index_processor import TejMarketIndexProcessor
from src.data_processor.tej_chip_data_processor import TejChipDataProcessor

# from src.utils import time_profiler
from src.data_store.data_store import DataStore, DataColumn
from src.utils.common import DataCategory, Market, Instrument, TimeFrame, TechnicalIndicator, DataCategoryColumn
from src.data_processor.tej_quarter_finance_processor import TejQuarterFinanceProcessor
from src.strategy.strategy import Strategy
from src.data_transformer.data_transformer import DataTransformer
from src.data_analyzer.trading_record_analyzer import TradingRecordAnalyzer

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


def main():
    TejQuarterFinanceProcessor.transform_raw_data_to_date_data(
        DataCategory.Finance_Report,
        Path("raw"),
        Path("data/finance/quarter_report")
    )
   


# def main():
    # TejMarketIndexProcessor.transform_date_data_to_code_data(
    #     Path("data/market_index/date"),
    #     Path("data/market_index/code")
    # )
    # TejDailyDataProcessor.transform_date_data_to_code_data(
    #     DataCategory.Daily_Price,
    #     Path("data/market_index/date_from_2024"),
    #     Path("data/daily_price/test_code")
    # )
    # TejDailyDataProcessor.transform_date_data_to_code_data(
    #     Path("data/daily_price/date"),
    #     Path("data/daily_price/code")
    # )
    # TejChipDataProcessor.transform_useful_columns_and_stock(
    #     Path("data/chip/date_raw"),
    #     Path("data/chip/date"),
    #     DataCategory.Chip,
    # )
    # TejChipDataProcessor.transform_date_data_to_code_data(
    #     Path("data/chip/date"),
    #     Path("data/chip/code"),
    # )
    # TejDailyDataProcessor.transform_useful_columns_and_stock(
    #     Path("data/daily_price/date_full"),
    #     Path("data/daily_price/date_partial"),
    #     DataCategory.Daily_Price
    # )


if __name__ == "__main__":
    TejDailyDataProcessor.transform_raw_data_to_date_data(
        DataCategory.Daily_Price,
        Path("raw/daily_price.txt"),
        Path("data/daily_price/"),
        encoding="utf-8"
    )
    TejMarketIndexProcessor.transform_raw_data_to_date_data(
        DataCategory.Daily_Price,
        Path("raw/daily_price.txt"),
        Path("data/market_index/"),
        encoding="utf-8"
    )
    TejMarketIndexProcessor.transform_raw_data_to_date_data(
        DataCategory.Chip,
        Path("raw/chip.txt"),
        Path("data/chip/"),
        encoding="utf-8"
    )