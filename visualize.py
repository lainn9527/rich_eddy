from pathlib import Path
from datetime import datetime

from src.data_store.data_store import DataStore
from src.strategy.trend_strategy import TrendStrategy
from src.platform.platform import Platform
from src.broker.broker import Broker
from src.data_visualizer.data_visualizer import DataVisualizer

def main():
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2024, 3, 1)

    codes = ["2230"]
    codes = ["2204", "2230", "3017", "3376", "3515"]
    # codes = ["3376"]
    DataVisualizer.visualize_signal_one(codes)
    
main()