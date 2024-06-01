from pathlib import Path
from datetime import datetime
import os
import sys

from src.data_store.data_store import DataStore
from src.strategy.trend_strategy import TrendStrategy
from src.platform.platform import Platform
from src.broker.broker import Broker
from src.data_visualizer.data_visualizer import DataVisualizer

def random_get_codes(n):
    import random
    codes = [file_name[:-4] for file_name in os.listdir("data/daily_price/code")]
    random.shuffle(codes)
    return codes[:n]

def main():
    result_path = Path("result") / (f"{sys.argv[1]}" if len(sys.argv) > 1 else "local_no_bk")
    codes = ["2230"]
    codes = ["2388"]
    codes = ["2204", "2230", "3017", "3376", "3515", "2388"]
    # DataVisualizer.visualize_signal_one(codes)
    # DataVisualizer.visualize_order_record(Path("result/20240421_232355/order_record.csv"))
    # DataVisualizer.visualize_local_min_max(codes)
    DataVisualizer.visualize_trend_strategy(codes, result_path)
    
if __name__ == "__main__":
    main()