import sys
from pathlib import Path


path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from src.process_data import process_quarter_finance


if __name__ == "__main__":
    source_path = Path("data/finance/raw/quarter_report.txt")
    dest_path = Path("data/finance/eps/eps.json")
#     remove_tej_stock_name_column('data/finance/month.csv', 'data/finance/processed-month.csv')
#     process_month_finance()
#     get_eps_from_accumulative_eps()
