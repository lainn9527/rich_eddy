import sys
from pathlib import Path


path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from src.analyze_data import *
from src.utils import read_tradable_stock


if __name__ == "__main__":
    config = {
        "sma_breakthrough_alignment_filter": {
            "week_half_month_diff_ratio": 0.03,
            "half_month_month_diff_ratio": 0.03,
        },
        "recurring_eps_filter": {"amplitudes": [0.05]},
    }

    tradable_codes = read_tradable_stock()
    codes = filter_by_layers(
        [sma_breakthrough_alignment_filter, recurring_eps_filter],
        tradable_codes,
        config,
    )

    print(codes)
    price_code = within_amplitude_trading(
        5,
        10,
        datetime.datetime.fromisoformat("2024-02-21"),
        datetime.datetime.fromisoformat("2024-02-21"),
        True,
        False,
    )
