import sys
from pathlib import Path


path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from src.process_data import process_quarter_finance


if __name__ == "__main__":
    source_path = Path("data/finance/raw/quarter_report.txt")
    dest_path = Path("data/finance/eps/eps.json")

    process_quarter_finance(source_path, dest_path)
