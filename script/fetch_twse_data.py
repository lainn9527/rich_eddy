import requests
import csv
import os

from datetime import datetime, timedelta
from pathlib import Path
from zipfile import ZipFile
from io import BytesIO

from src.data_processor.intra_day_data_processor import IntraDayDataProcessor


def fetch_twse_index_data(query_date: datetime):
    query_date_str = query_date.strftime("%Y%m%d")
    url = f"https://www.twse.com.tw/exchangeReport/MI_5MINS_INDEX?response=json&date={query_date_str}"
    request_data = requests.get(url).json()
    if request_data["stat"] != "OK":
        return []
    return [request_data["fields"]] + request_data["data"]


def fetch_twse_future_data(query_date: datetime):
    query_date_str = query_date.strftime("%Y_%m_%d")
    url = f"https://www.taifex.com.tw/file/taifex/Dailydownload/DailydownloadCSV/Daily_{query_date_str}.zip"
    request_data = requests.get(url)
    if len(request_data.content) < 1000 or request_data.status_code != 200:
        return []

    with BytesIO(request_data.content) as request_zip_file:
        with ZipFile(request_zip_file) as zip_file:
            csv_data = (
                zip_file.read(f"Daily_{query_date_str}.csv")
                .decode("big5")
                .split("\r\n")
            )
            csv_data = [line.split(",") for line in csv_data if line]

    return csv_data


def fetch_all_data(start_date: datetime, end_date: datetime, fetch_data_func: callable):
    all_data = {}
    current_date = start_date
    while current_date <= end_date:
        twse_index_data = fetch_data_func(current_date)
        if len(twse_index_data) > 0:
            all_data[current_date] = twse_index_data
        current_date += timedelta(days=1)
    return all_data


def write_data_to_csv_by_year(
    dest_dir: Path, year, file_name: str, csv_data: list[any]
):
    dest_file = dest_dir / str(year) / file_name
    dest_file.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(csv_data)


def rename_file_in_folder(folder: Path):
    all_files = os.listdir(folder)
    for file_name in all_files:
        date_part, ext = file_name.split(".")
        new_file_name = (
            datetime.strptime(date_part, "%Y%m%d").strftime("%Y-%m-%d") + "." + ext
        )
        (folder / file_name).rename(folder / new_file_name)


if __name__ == "__main__":
    dest_dir = Path("intra_day_data")
    twse_index_dest_dir = dest_dir / "twse_index"
    twse_index_kbar_dest_dir = dest_dir / "twse_index_kbar2"
    twse_future_dest_dir = dest_dir / "future"
    twse_future_kbar_dest_dir = dest_dir / "future_kbar2"

    IntraDayDataProcessor.transform_and_write_year_dir_data(
        dest_dir / "twse_index",
        IntraDayDataProcessor.transform_twse_index_to_kbar,
        dest_dir / "twse_index_kbar2",
        ["twse"],
    )
    # # fetch data from a range
    # end_date = datetime(2025, 3, 4) + timedelta(days=1)
    # start_date = end_date + timedelta(days=-6)

    # print(f"fetch data from {start_date} to {end_date}")
    # twse_index_data = fetch_all_data(start_date, end_date, fetch_twse_index_data)
    # twse_future_data = fetch_all_data(start_date, end_date, fetch_twse_future_data)
    # print(f"fetch data done: {twse_index_data.keys()}")
    # [write_data_to_csv_by_year(twse_index_dest_dir, date.year, f"{date.strftime('%Y-%m-%d')}.csv", twse_index_data[date]) for date in twse_index_data.keys()]
    # [write_data_to_csv_by_year(twse_future_dest_dir, date.year, f"{date.strftime('%Y-%m-%d')}.rpt", twse_future_data[date]) for date in twse_future_data.keys()]

    # IntraDayDataProcessor.transform_csv_and_write_dir_data(twse_index_data, IntraDayDataProcessor.transform_twse_index_to_kbar, dest_dir / "tmp2", ["twse"])
    # IntraDayDataProcessor.transform_csv_and_write_dir_data(twse_future_data, IntraDayDataProcessor.transform_twse_future_to_kbar, dest_dir / "tmp1", ["TX"])
    # print(f"write twse index data to {twse_index_dest_dir}")
    # print(f"write future data data to {twse_future_dest_dir}")
    # print(f"write future kbar data to {twse_future_kbar_dest_dir}")

  
    # fetch data from a today

    # year_dirs = os.listdir(dest_dir)
    # for year_dir in year_dirs:
    #     rename_file_in_folder(dest_dir / year_dir)
    # rename_file_in_folder(Path("intra_day_data") / "future" / "2024")

    # fetch_twse_future_data(datetime(2024, 8, 30))
