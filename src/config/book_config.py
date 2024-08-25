book_config = {
  "activated_filters": {
    "filter_up_min_ratio": True,
    "filter_high_before": True,
    "filter_correction_max_ratio": True,
    "filter_breakthrough_point": True,
    "filter_early_breakthrough": False,
    "filter_eps": False,
    "filter_recurring_eps": False,
    "filter_consolidation_time_window": False,
    "filter_relative_strength": False,
    "filter_market_index": False,
    "filter_chip": False,
    "filter_volume": False,
    "filter_signal_threshold": False,
  },
  "strategy_one": {
    "close_above_sma_period": 60,
    "up_min_ratio": 0.40,
    "up_time_window": 180,
    "no_high_before_ratio": 0.3,
    "no_high_before_window": 120,
    "correction_max_ratio": 0.40,
    "correction_max_time_window": 5,
    "cheat_ratio": 0.20,
    "valid_signal_window": 60,
    "consolidation_time_window": 5,
    "rs_threshold": 40,
    "rs_sma_period": 20,
    "init_stop_loss_ratio": 0.1,
    "dynamic_stop_loss_range_ratio": 0.5,

    "breakthrough_fuzzy": 0.03,
    "volume_avg_time_window": 120,
    "volume_avg_threshold": 200,
    "holding_days": 10,
    "market_index_sma_period": 3,
    "signal_threshold": 2,
    "volume_short_sma_period": 5,
    "volume_long_sma_period": 10
  },
  "chip_strategy": {
    "foreign_total_holdings_ratio_sma_period": 10,
    "local_self_holdings_ratio_sma_period": 10,
    "local_investor_holdings_ratio_sma_period": 5
  }
}