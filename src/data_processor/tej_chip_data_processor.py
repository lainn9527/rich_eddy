from datetime import datetime
from typing import List

from .tej_data_processor import TejDataProcessor

class TejChipDataProcessor(TejDataProcessor):
    def extract_code_and_date_from_line(line: List[str]):
        code = line[0]
        trading_date = datetime.strptime(line[1], "%Y%m%d").date()
        return code, None, trading_date, line[2:]