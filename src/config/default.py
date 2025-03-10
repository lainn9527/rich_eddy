from pathlib import Path

from src.data_provider import DailyPriceProvider, FinanceProvider, IntraDayProvider
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
        {
            "market": Market.TW,
            "instrument": Instrument.Future,
            "data_category": DataCategory.Minute_Price,
            "data_provider_class": IntraDayProvider,
            "data_dir": Path("intra_day_data/future_kbar"),
        },
        {
            "market": Market.TW,
            "instrument": Instrument.StockIndex,
            "data_category": DataCategory.Minute_Index,
            "data_provider_class": IntraDayProvider,
            "data_dir": Path("intra_day_data/twse_index_kbar"),
        },
    ],
    "parameter": {
        "activated_filters": {
            "filter_up_min_ratio": True,
            "filter_down_max_ratio": True,
            "filter_breakthrough_point": True,
            "filter_consolidation_time_window": True,
            "filter_relative_strength": True,
            "filter_market_index": True,
            "filter_chip": True,
            "filter_volume": True,
            "filter_signal_threshold": True,
            "filter_close_above_sma": True,
        },
        "strategy_one": {
            "close_above_sma_period": 60,
            "up_min_ratio": 0.32,
            "up_time_window": 70,
            "down_max_ratio": 0.21,
            "down_max_time_window": 22,
            "consolidation_time_window": 13,
            "breakthrough_fuzzy": 0.03,
            "volume_avg_time_window": 120,
            "volume_avg_threshold": 200,
            "holding_days": 4,
            "stop_loss_ratio": 0.05,
            "rs_threshold": 94,
            "rs_sma_period": 3,
            "market_index_sma_period": 3,
            "signal_threshold": 2,
            "volume_short_sma_period": 5,
            "volume_long_sma_period": 20,
        },
        "chip_strategy": {
            "foreign_total_holdings_ratio_sma_period": 10,
            "local_self_holdings_ratio_sma_period": 10,
            "local_investor_holdings_ratio_sma_period": 10,
        },
    },
}

tune_config = {
    "activated_filters": {
        "filter_up_min_ratio": True,
        "filter_down_max_ratio": True,
        "filter_breakthrough_point": True,
        "filter_consolidation_time_window": False,
        "filter_relative_strength": True,
        "filter_market_index": True,
        "filter_chip": True,
        "filter_volume": False,
        "filter_signal_threshold": True,
    },
    "strategy_one": {
        "up_min_ratio": 0.35,
        "up_time_window": 60,
        "down_max_ratio": 0.26,
        "down_max_time_window": 5,
        "consolidation_time_window": 10,
        "breakthrough_fuzzy": 0.03,
        "volume_avg_time_window": 120,
        "volume_avg_threshold": 200,
        "holding_days": 5,
        "stop_loss_ratio": 0.05,
        "rs_threshold": 70,
        "rs_sma_period": 2,
        "market_index_sma_period": 3,
        "volume_short_sma_period": 5,
        "volume_long_sma_period": 20,
        "signal_threshold": 3,
    },
    "chip_strategy": {
        "foreign_total_holdings_ratio_sma_period": 10,
        "local_self_holdings_ratio_sma_period": 10,
        "local_investor_holdings_ratio_sma_period": 10,
    },
}

large_record_config = {
    "activated_filters": {
        "filter_up_min_ratio": True,
        "filter_down_max_ratio": True,
        "filter_breakthrough_point": True,
        "filter_consolidation_time_window": True,
        "filter_relative_strength": True,
        "filter_market_index": True,
        "filter_chip": True,
        "filter_volume": False,
        "filter_signal_threshold": True,
    },
    "strategy_one": {
        "up_min_ratio": 0.32,
        "up_time_window": 77,
        "down_max_ratio": 0.23,
        "down_max_time_window": 21,
        "consolidation_time_window": 9,
        "breakthrough_fuzzy": 0.03,
        "volume_avg_time_window": 120,
        "volume_avg_threshold": 200,
        "holding_days": 3,
        "stop_loss_ratio": 0.05,
        "rs_threshold": 95,
        "rs_sma_period": 1,
        "market_index_sma_period": 3,
        "signal_threshold": 2,
        "volume_short_sma_period": 5,
        "volume_long_sma_period": 20,
    },
    "chip_strategy": {
        "foreign_total_holdings_ratio_sma_period": 10,
        "local_self_holdings_ratio_sma_period": 10,
        "local_investor_holdings_ratio_sma_period": 10,
    },
}

small_record_config = {
  "activated_filters": {
    "filter_up_min_ratio": True,
    "filter_down_max_ratio": True,
    "filter_breakthrough_point": True,
    "filter_consolidation_time_window": False,
    "filter_relative_strength": True,
    "filter_market_index": True,
    "filter_chip": True,
    "filter_volume": False,
    "filter_signal_threshold": True,
  },
  "strategy_one": {
    "up_min_ratio": 0.33,
    "up_time_window": 69,
    "down_max_ratio": 0.25,
    "down_max_time_window": 20,
    "consolidation_time_window": 5,
    "breakthrough_fuzzy": 0.03,
    "volume_avg_time_window": 120,
    "volume_avg_threshold": 200,
    "holding_days": 4,
    "stop_loss_ratio": 0.04,
    "rs_threshold": 93,
    "rs_sma_period": 3,
    "market_index_sma_period": 3,
    "signal_threshold": 2,
    "volume_short_sma_period": 5,
    "volume_long_sma_period": 20
  },
  "chip_strategy": {
    "foreign_total_holdings_ratio_sma_period": 10,
    "local_self_holdings_ratio_sma_period": 10,
    "local_investor_holdings_ratio_sma_period": 10
  }
}