import sys
from pathlib import Path

path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))
import csv
import json
from multiprocessing import Pool

import plotly.figure_factory as ff

from legacy.analyze_data import analyze_signal, data_summary, find_strategy_one_signal
from legacy.config import config
from legacy.plot_data import plot_strategy_one_signal
from legacy.utils import read_all_stock_codes, read_daily_data, read_tradable_stock


def task(signals):
    signal_result = analyze_signal(signals, config)
    return signal_result


if __name__ == "__main__":
    codes = read_all_stock_codes()
    print(f"Codes: {len(codes)}")
    num_of_cores = 8
    split_size = len(codes) // num_of_cores + 1
    split_signals = []
    n = 0
    with open("strategy_one.json", "r") as fp:
        signals = json.load(fp)

    while n < len(codes):
        part_signals = {}
        for code in codes[n : n + split_size]:
            if code not in signals:
                continue
            part_signals[code] = signals[code]

        split_signals.append(part_signals)
        n += split_size

    with Pool(num_of_cores) as p:
        results = p.map(task, split_signals)

    all_signal_result = sum(results, [])
    all_hold_ratios, all_min_ratios, all_max_ratios = [], [], []
    for signal_result in all_signal_result:
        all_hold_ratios.append(signal_result["hold_in_2w"])
        all_max_ratios.append(signal_result["max_in_2w"])
        all_min_ratios.append(signal_result["min_in_2w"])

    all_signal_result = sorted(all_signal_result, key=lambda x: x["start_date"], reverse=False)
    with open('legacy_signal_one_result.csv', 'w') as file:
        header_written = False
        for signal_result in all_signal_result:
            writer = csv.DictWriter(file, signal_result.keys())
            if not header_written:
                writer.writeheader()
                header_written = True
            writer.writerow(signal_result)

    # data_summary("Hold ratio", all_hold_ratios)
    # data_summary("Max ratio", all_max_ratios)
    # data_summary("Min ratio", all_min_ratios)

    # Group data together
    # fig = ff.create_distplot(
    #     [all_hold_ratios, all_max_ratios, all_min_ratios],
    #     ["signal hold", "signal max", "signal min"],
    #     bin_size=0.2,
    # )
    # fig.show()
    # fig = ff.create_distplot([ready_breakthrough_max_ratios, ready_breakthrough_min_ratios], ['ready max', 'ready min'], bin_size=.2)
    # fig.show()
