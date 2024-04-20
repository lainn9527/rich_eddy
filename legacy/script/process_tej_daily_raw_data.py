import sys
from pathlib import Path


path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from src.process_data import (
    process_null_date_in_date_data,
    process_tej_raw_daily_data,
    transform_date_data_to_code_data,
)


if __name__ == "__main__":
    picked_daily_price_columns = [
        "股票代碼",
        "日期",
        "開盤價(元)",
        "最高價(元)",
        "最低價(元)",
        "收盤價(元)",
        "成交量(千股)",
        "成交值(千元)",
        "流通在外股數(千股)",
    ]

    # process_null_date_in_date_data('daily_price', 2010)
    process_tej_raw_daily_data("daily_price", True)
    transform_date_data_to_code_data("daily_price", 2010, picked_daily_price_columns)
