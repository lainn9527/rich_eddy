import csv
from datetime import date
from pathlib import Path
from typing import List

from .tej_data_processor import TejDataProcessor
from ..utils.common import DataCategory, DataColumn

declare_report_data = {1: "05-15", 2: "08-14", 3: "11-14", 4: "03-31"}

class TejQuarterFinanceProcessor(TejDataProcessor):
    def month_to_quarter(month):
        return str((int(month) - 1) // 3 + 1)

    def extract_code_and_date_from_line(line: List[str]):
        code_with_name, year_month = line[0], line[1]
        code = code_with_name.split(" ")[0]
        year, quarter = year_month[:4], TejQuarterFinanceProcessor.month_to_quarter(year_month[4:])
        release_datetime = TejQuarterFinanceProcessor.year_quart_to_release_date(year, quarter)

        return code, None, release_datetime, line[2:]
    
    def is_valid_stock(code: str):
        return (
            code.isdecimal()
            and int(code) >= 1000
            and int(code) <= 9999
            and len(code) == 4
        )

    def process_quarter_finance(origin_file_path: Path, dest_data_dir: Path):
        processed_lines = []
        with open(origin_file_path) as fp:
            lines = TejQuarterFinanceProcessor.remove_null_token(fp.readlines(), ["\x00", "-"])
            lines = list(csv.reader(lines))

            raw_column_names = TejQuarterFinanceProcessor.get_column_names(DataCategory.Finance_Report)
            for line in lines[1:]:
                if len(line) != len(raw_column_names):
                    raise ValueError(
                        f"finance-report data length {len(line)} not equal to column length {len(raw_column_names)}"
                    )

                code_with_name, year_month = line[0], line[1]
                code = code_with_name.split(" ")[0]
                year, quarter = year_month[:4], TejQuarterFinanceProcessor.month_to_quarter(year_month[4:])
                release_datetime = TejQuarterFinanceProcessor.year_quart_to_release_date(year, quarter)

                processed_lines.append([code, release_datetime.strftime("%Y%m%d")] + line[2:])

            processed_lines.sort(key=lambda d: (int(d[1]), d[0]))
            processed_lines.insert(0, raw_column_names)

        # write full data
        if not dest_data_dir.exists():
            dest_data_dir.mkdir(parents=True, exist_ok=True)
        
        with open(dest_data_dir / "date_full.csv", "w") as fp:
            writer = csv.writer(fp)
            writer.writerows(processed_lines)

        # write picked column
        picked_lines = TejQuarterFinanceProcessor.pick_columns(DataCategory.Finance_Report, processed_lines)
        pick_data_path = dest_data_dir / "finance_report.csv"
        with open(pick_data_path, "w") as fp:
            writer = csv.writer(fp)
            writer.writerows(picked_lines)


    def year_quart_to_release_date(year, quarter):
        return date.fromisoformat(f"{year}-{declare_report_data[int(quarter)]}")