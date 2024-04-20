import sys
from pathlib import Path


path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from src.analyze_data import filter_stock_by_volume


if __name__ == "__main__":
    filter_stock_by_volume(100, 120)
