from pathlib import Path
from datetime import datetime
from src.data_visualizer.future_order_visualizer import FutureOrderVisualizer


if __name__ == "__main__":
    record_path = Path("record/future_record2.csv")
    data_path = Path("intra_day_data") / "future_kbar"
    # FutureOrderVisualizer.visualize_future_record(record_path, data_path, datetime(2025, 2, 10), datetime(2025, 2, 15))

    data_dir = Path("future_result/gap_100_5min")
    FutureOrderVisualizer.visualize_gaps(data_dir, datetime(2025, 2, 10), datetime(2025, 2, 15))