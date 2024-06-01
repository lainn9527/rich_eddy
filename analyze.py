from pathlib import Path
import sys

from src.data_analyzer.trading_record_analyzer import TradingRecordAnalyzer


if __name__ == "__main__":
    result_path = Path("result") / (f"{sys.argv[1]}" if len(sys.argv) > 1 else "best2")
    original_result_dir = Path("result/adjust_volume_by_rs_95")
    target_result_dir = Path("result/without_rs_filter")
    # TradingRecordAnalyzer.compare_trading_record(original_result_dir, target_result_dir)
    TradingRecordAnalyzer.analyze(result_path)