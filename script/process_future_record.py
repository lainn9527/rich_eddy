from pathlib import Path

from src.data_processor.future_record_processor import FutureRecordProcessor

if __name__ == "__main__":
    data_path = Path("record/raw/combined.csv")
    order_df = FutureRecordProcessor.process_raw_data(data_path)
    order_df.to_csv("record/future_record2.csv", index=False)