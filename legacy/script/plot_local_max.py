import sys
from pathlib import Path


path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

from legacy.config import config
from legacy.plot_data import plot_stock_with_signal


if __name__ == "__main__":
    codes = ["2204", "2230", "3017", "3376", "3515"]
    for code in codes:
        plot_stock_with_signal(code, config)
