from pathlib import Path
from datetime import datetime
from pyinstrument import Profiler
from functools import wraps
from multiprocessing import Pool
import argparse
import numpy as np

import sys
import talib
from src.data_store.data_store import DataStore
from src.strategy.trend_strategy import TrendStrategy
from src.strategy.chip_strategy import ChipStrategy
from src.platform.platform import Platform
from src.broker.broker import Broker
from src.config.default import config
from src.utils.utils import combine_config

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        default="tune",
        help="Could be run, tune, or random_tune",
    )
    parser.add_argument(
        "--path",
        "-p",
        type=str,
        default=None,
        help="Result path",
    )
    parser.add_argument(
        "--record",
        "-r",
        type=bool,
        default=False,
        help="Record all the result or not",
    )
    parser.add_argument(
        "--random_count",
        "-n",
        type=int,
        default=100,
        help="Random tune count",
    )
    return parser.parse_args()


def time_profiler(func):
    # only for local development
    @wraps(func)
    def wrapper(*args, **kwargs):
        profiler = Profiler()
        profiler.start()
        result = func(*args, **kwargs)
        profiler.stop()
        print(profiler.output_text(unicode=True, color=True))
        return result

    return wrapper

def random_combinations(config, n = 5):
    import random
    combinations = []
    for _ in range(n):
        config_comb = {}
        for key, values in config.items():
            if type(values[0]) == int:
                config_comb[key] = random.randint(values[0], values[-1])
            elif type(values[0]) == float:
                config_comb[key] = round(random.uniform(values[0], values[-1]), 2)
            elif type(values[0]) == bool:
                config_comb[key] = random.choice(values)
        combinations.append(config_comb)
    
    return combinations


def exhaust_combinations(config):
    combinations = [{}]
    for key, value in config.items():
        new_combinations = []
        for combination in combinations:
            new_combinations.extend([{**combination, key: value} for value in value])
        combinations = new_combinations
    
    return combinations


def build_combinations(source_config, explanatory_variables, mode = 'tune', n=5):
    # config and explanatory_variables is at { strategy_x: { key: value } } format
    all_config = {}
    # flatten the explanatory_variables
    for key, sub_config in explanatory_variables.items():
        for sub_key, values in sub_config.items():
            all_config[f"{key}|{sub_key}"] = values

    if mode == 'tune':
        combinations = exhaust_combinations(all_config)
    elif mode == 'random_tune':
        combinations = random_combinations(all_config, n)

    # group the config back to the original format
    grouped_configs = []
    for combination in combinations:
        grouped_config = {}
        for key, value in combination.items():
            key, sub_key = key.split("|")
            if key not in grouped_config:
                grouped_config[key] = {}
            grouped_config[key][sub_key] = value
        grouped_configs.append(combine_config(source_config, grouped_config))

    return grouped_configs

def tune(args: argparse.Namespace):
    explanatory_variables = {
        "activated_filters": {
            "filter_up_min_ratio": [True, False],
            "filter_down_max_ratio": [True, False],
            "filter_consolidation_time_window": [True, False],
            "filter_relative_strength": [True, False],
            "filter_market_index": [True, False],
            "filter_chip": [True, False],
            "filter_volume": [True, False],
            "filter_signal_threshold": [True, False],
        },
        "strategy_one": {
            "up_min_ratio": [0.3, 0.35, 0.4, 0.45],
            "up_time_window": [60, 75, 90],
            "down_max_ratio": [0.2, 0.25, 0.3, 0.35],
            "down_max_time_window": [20, 25, 30, 35],
            "consolidation_time_window": [5, 7, 10, 13, 15],
            "stop_loss_ratio": [0.03, 0.04, 0.05, 0.06],
            "holding_days": [3, 5, 7, 10],
            "rs_threshold": [70, 95],
            "rs_sma_period": [1, 2, 3],
        },
    }
    config_combinations = build_combinations(config["parameter"], explanatory_variables, mode=args.mode, n=args.random_count)
    turn_result_path = Path("result") / args.path if args.path != None else Path("result") / f"tune_result_{len(config_combinations)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    turn_result_path.mkdir(parents=True, exist_ok=True)

    multi_processing_payload = []
    for i, combination in enumerate(config_combinations):
        result_path = turn_result_path/ f"tune_{i}"
        multi_processing_payload.append([argparse.Namespace(mode="run", path=result_path, record=args.record), combination])
    with Pool(6) as p:
        p.starmap(main, multi_processing_payload)
    
    # start_date = datetime(2022, 11, 18)
    # start_date = datetime(2011, 11, 18)
    # end_date = datetime(2024, 3, 1)

    # platform = Platform({ "broker": Broker()})
    # cash = 10000000000
    # data_store = DataStore()
    # for i, combination in enumerate(config_combinations):
    #     result_path = turn_result_path/ f"tune_{i}"
    #     print(f"Start to run {i} combination")
    #     strategy = TrendStrategy(platform, data_store, cash=cash, config=combination)
    #     platform.run(strategy, start_date, end_date, result_path)


def main(arguments: argparse.Namespace, config):
    start_date = datetime(2022, 11, 18)
    start_date = datetime(2011, 11, 18)
    end_date = datetime(2023, 12, 31)
    platform = Platform({ "broker": Broker()})
    cash = 10000000000
    data_store = DataStore()
    result_path = Path(arguments.path) if arguments.path != None else None
    strategy = TrendStrategy(platform, data_store, cash=cash, config=config)
    # strategy = ChipStrategy(platform, data_store, cash=cash, config=config)
    platform.run(strategy, start_date, end_date, result_path, full_record=arguments.record)


if __name__ == "__main__":
    args = parse_args()
    print(datetime.now())
    if args.mode == "run":
        main(args, config["parameter"])
    elif args.mode == "tune" or args.mode == "random_tune":
        tune(args)
    print(datetime.now())