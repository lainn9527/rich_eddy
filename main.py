from pathlib import Path
from datetime import datetime

import talib
import numpy as np

from src.data_store.data_store import DataStore
from src.strategy.trend_strategy import TrendStrategy
from src.platform.platform import Platform
from src.broker.broker import Broker
from src.config.default import config

def main():
    start_date = datetime(2023, 11, 18)
    end_date = datetime(2024, 3, 1)

    platform = Platform({ "broker": Broker()})
    cash = 100000000
    holding_days = [5, 10, 13, 15, 18, 20]
    data_store = DataStore()
    for holding_day in holding_days:
        print(f"Start to run holding day {holding_day}")
        config["parameter"]["strategy_one"]["holding_days"] = holding_day
        strategy = TrendStrategy(platform, data_store, cash=cash, config=config["parameter"])
        platform.run(strategy, start_date, end_date)
        break

if __name__ == "__main__":
    print(datetime.now())
    main()
    print(datetime.now())