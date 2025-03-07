from pathlib import Path
from datetime import datetime

from src.data_visualizer.future_order_visualizer import FutureOrderVisualizer
from src.data_processor.future_record_processor import FutureRecordProcessor


if __name__ == "__main__":
    record_path = Path("future_result") / "gap_100_5min_all" / "order_record.csv"
    data_path = Path("intra_day_data") / "future_kbar"
    start_date = datetime(2020, 1, 20)
    end_date = datetime(2026, 1, 30)
    FutureOrderVisualizer.visualize_pair_intra_strategy(record_path, data_path)
