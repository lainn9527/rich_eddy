from pathlib import Path
from datetime import datetime

import sys
import talib
import numpy as np

from src.data_store.data_store import DataStore
from src.strategy.trend_strategy import TrendStrategy
from src.platform.platform import Platform
from src.broker.broker import Broker
from src.config.default import config, tuned_config

def random_combinations(config, n = 5):
    import random
    combinations = []
    for _ in range(n):
        config_comb = {}
        for key, values in config.items():
            low, high = values[0], values[-1]
            if type(low) == int:
                config_comb[key] = random.randint(low, high)
            elif type(low) == float:
                config_comb[key] = round(random.uniform(low, high), 2)
        combinations.append(["-".join([f"{key}_{value}" for key, value in config_comb.items()]), config_comb])
    return combinations

def tune():
    start_date = datetime(2022, 11, 18)
    start_date = datetime(2011, 11, 18)
    end_date = datetime(2024, 3, 1)

    platform = Platform({ "broker": Broker()})
    cash = 100000000
    explanatory_variables = {
        "up_min_ratio": [0.10, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45],
        "up_time_window": [45, 60, 75, 90],
        "down_max_ratio": [0.2, 0.25, 0.3, 0.35],
        "down_max_time_window": [20, 25, 30, 35],
        "consolidation_time_window": [5, 7, 10, 13, 15],
    }
    combinations = []
    # for key, values in explanatory_variables.items():
    #     if combinations == []:
    #         combinations = [[f"{key}_{value}", {key: value}] for value in values]
    #         continue
    #     new_combinations = []
    #     for combination in combinations:
    #         for value in values:
    #             new_combinations.append([f"{combination[0]}_{key}_{value}", {**combination[1], key: value}])
    #     combinations = new_combinations

    combinations = random_combinations(explanatory_variables, 100)
    turn_result_path = Path("result") / f"{sys.argv[1]}" if len(sys.argv) > 1 else Path("result") / f"tune_result_{len(combinations)}"
    turn_result_path.mkdir(parents=True, exist_ok=True)
    data_store = DataStore()
    for combination in combinations:
        result_name = combination[0]
        new_config = {**config["parameter"], "strategy_one": {**config["parameter"]["strategy_one"], **combination[1] }}
        result_path = turn_result_path/ result_name

        print(f"Start to run {result_name}")
        strategy = TrendStrategy(platform, data_store, cash=cash, config=new_config)
        platform.run(strategy, start_date, end_date, result_path)

def main():
    start_date = datetime(2022, 11, 18)
    start_date = datetime(2011, 11, 18)
    end_date = datetime(2024, 3, 1)

    platform = Platform({ "broker": Broker()})
    cash = 100000000
    data_store = DataStore()
    result_path = Path("result") / f"{sys.argv[1]}" if len(sys.argv) > 1 else None
    strategy = TrendStrategy(platform, data_store, cash=cash, config=tuned_config)
    platform.run(strategy, start_date, end_date, result_path)

if __name__ == "__main__":
    print(datetime.now())
    main()
    print(datetime.now())