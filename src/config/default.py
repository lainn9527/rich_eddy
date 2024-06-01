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
            "code_data_dir": Path("data/daily_price/code"),
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
        {
            "market": Market.TW,
            "instrument": Instrument.Stock,
            "data_category": DataCategory.Chip,
            "data_provider_class": DailyPriceProvider,
            "data_dir": Path("data/chip"),
        },
    ],
    "parameter": {
        "sma_breakthrough_alignment_filter": {
            "week_half_month_diff_ratio": 0.03,
            "half_month_month_diff_ratio": 0.03,
        },
        # "recurring_eps_filter": {"amplitudes": [0.05]},
        "strategy_one": {
            "up_min_ratio": 0.45,
            "up_time_window": 60,
            "down_max_ratio": 0.25,
            "down_max_time_window": 30,
            "consolidation_time_window": 10,
            "breakthrough_fuzzy": 0.2,

            "volume_avg_time_window": 120,
            "volume_avg_threshold": 200,

            "holding_days": 5,
            "stop_loss_ratio": 0.037,

            "rs_threshold": 90,
            "rs_sma_period": 3,

            "market_index_sma_period": 30,
        },
        "chip_strategy": {
            "foreign_total_holdings_ratio_sma_period": 5,
            "local_self_holdings_ratio_sma_period": 5,
            "local_investor_holdings_ratio_sma_period": 5,
        }
    }
}

tuned_config = {
    "sma_breakthrough_alignment_filter": {
        "week_half_month_diff_ratio": 0.03,
        "half_month_month_diff_ratio": 0.03
    },
    "strategy_one": {
        "up_min_ratio": 0.32,
        "up_time_window": 70,
        "down_max_ratio": 0.21,
        "down_max_time_window": 22,
        "consolidation_time_window": 13,
        "breakthrough_fuzzy": 0.2,
        "volume_avg_time_window": 120,
        "volume_avg_threshold": 200,
        "holding_days": 4,
        "stop_loss_ratio": 0.05,
        "rs_threshold": 94,
        "rs_sma_period": 3,
        "market_index_sma_period": 3,
        "signal_threshold": 2,
    },
    "chip_strategy": {
        "foreign_total_holdings_ratio_sma_period": 60,
        "local_self_holdings_ratio_sma_period": 5,
        "local_investor_holdings_ratio_sma_period": 5,
    }
}
