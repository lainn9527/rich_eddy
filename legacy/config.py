config = {
    "meta": {
        "from_year": 2020,
    },
    "sma_breakthrough_alignment_filter": {
        "week_half_month_diff_ratio": 0.03,
        "half_month_month_diff_ratio": 0.03,
    },
    "recurring_eps_filter": {"amplitudes": [0.05]},
    "strategy_one": {
        "start_days_before_current_date": 4745,
        "signal_before_days": 4745,
        "up_min_ratio": 0.45,
        "up_time_window": 60,
        "down_max_ratio": 0.25,
        "down_max_time_window": 30,
        "consolidation_time_window": 10,
        "breakthrough_fuzzy": 0.2,
    },
}
