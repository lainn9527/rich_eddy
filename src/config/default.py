from pathlib import Path

from src.data_provider import DailyPriceProvider, FinanceProvider
from src.utils.common import DataCategory, Instrument, Market, TimeFrame


config = {
    "data_provider": [
        {
            "market": Market.TW,
            "instrument": Instrument.Stock,
            "data_category": DataCategory.Daily_Price,
            "data_provider_class": DailyPriceProvider,
            "data_dir": Path("data/daily_price"),
            "code_data_dir": Path("data/daily_price/code_from_2020"),
        },
        {
            "market": Market.TW,
            "instrument": Instrument.Stock,
            "data_category": DataCategory.Finance_Report,
            "data_provider_class": FinanceProvider,
            "data_dir": Path("data/finance/quarter_report"),
        },
        {
            "market": Market.TW,
            "instrument": Instrument.StockIndex,
            "data_category": DataCategory.Market_Index,
            "data_provider_class": DailyPriceProvider,
            "data_dir": Path("data/market_index"),
        },
    ],
    "parameter": {
        "meta": {
            "from_year": 2010,
        },
        "sma_breakthrough_alignment_filter": {
            "week_half_month_diff_ratio": 0.03,
            "half_month_month_diff_ratio": 0.03,
        },
        "recurring_eps_filter": {"amplitudes": [0.05]},
        "strategy_one": {
            "up_min_ratio": 0.45,
            "up_time_window": 60,
            "down_max_ratio": 0.25,
            "down_max_time_window": 30,
            "consolidation_time_window": 10,
            "breakthrough_fuzzy": 0.2,

            "volume_avg_time_window": 120,
            "volume_avg_threshold": 200,

            "holding_days": 10,
            "stop_loss_ratio": 0.037,

            "rs_threshold": 0.99,
        },
    }
}
