import csv
import gc
import os

from datetime import datetime
from pathlib import Path
from typing import List

from .tej_data_processor import TejDataProcessor

class TejMarketIndexProcessor(TejDataProcessor):
    def extract_code_and_date_from_line(line: List[str]):
        code_with_name, trading_date = line[0], datetime.strptime(line[1], "%Y%m%d").date()
        code, name = code_with_name.split(" ")[0], code_with_name.split(" ")[1]
        return code, name, trading_date, line[2:]

    def is_valid_stock(code: str):
        return code[0].isalpha()
