import sys
from pathlib import Path


path_root = Path(__file__).parents[2]
print(path_root)
sys.path.append(str(path_root))

import datetime

from legacy.analyze_data import find_strategy_one_signal
from legacy.config import config
from legacy.plot_data import plot_strategy_one_signal


if __name__ == "__main__":
    codes = ["2230"]
    codes = ["2204", "2230", "3017", "3376", "3515"]
    # codes = ["3376"]
    signals, ready_breakthrough_signals = find_strategy_one_signal(codes, config)
    print(f"Signals: {len(signals)}")
    print(f"Ready breakthrough signals: {len(ready_breakthrough_signals)}")
    codes = sorted(list(signals.keys()))
    start_idx = 0
    print(list(signals.keys()))
    print(list(ready_breakthrough_signals.keys()))
    for code in codes[start_idx : start_idx + 10]:
        plot_strategy_one_signal(
            code,
            signals[code],
            datetime.datetime(year=config["meta"]["from_year"], month=1, day=1),
        )
    # for code, breakthrough_signals in ready_breakthrough_signals.items():
    #     plot_strategy_one_signal(code, breakthrough_signals)
