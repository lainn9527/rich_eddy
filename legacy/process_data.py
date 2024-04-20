import csv
import gc
import json
import os
import tarfile
from datetime import date
from pathlib import Path

from utils import get_column_names, replace_null_with_dash

# process raw data from tej and store in {year}/{date}.csv format
# def process_tej_raw_daily_data(data_category: str, write_compressed: bool = False):
#     year_to_daily_info = dict()
#     raw_data_dir = Path(f'data/{data_category}/raw')
#     file_names = os.listdir(raw_data_dir)
#     latest_date = date.fromisoformat('1900-01-01')

#     for file_name in file_names:
#         file_path = raw_data_dir / file_name
#         with open(file_path, 'r') as fp:
#             lines = list(csv.reader(fp))
#             for idx, line in enumerate(lines):
#                 # first field need to be code
#                 code_with_name, trading_date = line[0], date.fromisoformat(line[1])
#                 code = code_with_name.split(' ')[0]

#                 year = trading_date.year
#                 if year_to_daily_info.get(year) == None:
#                     year_to_daily_info[year] = dict()
#                 if year_to_daily_info.get(year).get(trading_date.isoformat()) == None:
#                     year_to_daily_info[year][trading_date.isoformat()] = []

#                 year_to_daily_info[year][trading_date.isoformat()].append([code] + line[1:])

#                 if trading_date > latest_date:
#                     latest_date = trading_date
#                 if idx % 100000 == 0:
#                     print(f'finish {idx} lines, total {len(lines)} lines')
#     print('finish reading all lines')

#     gc.collect()
#     print('finish gc')

#     column_names = get_column_names(data_category)
#     dest_dir_path = Path(f'data/{data_category}/date')
#     for year, daily_info_dict in year_to_daily_info.items():
#         year_dir_path = dest_dir_path / str(year)
#         if not year_dir_path.exists():
#             year_dir_path.mkdir(parents = True, exist_ok = True)

#         for trading_date, daily_info in daily_info_dict.items():
#             with open(f'{year_dir_path}/{trading_date}.csv', 'w') as fp:
#                 csv.writer(fp).writerows([column_names] + daily_info)
#         print(f'finish year {year}')

#     print('finish writing all files')

#     if write_compressed:
#         tar = tarfile.open(f'data/daily_price/date-data-{"2024-03-01"}.gz', 'w:gz')
#         tar.add(f'data/{data_category}/date', arcname = '',)
#         tar.close()

# def transform_date_data_to_code_data(data_category: str, start_year: int, picked_columns: list = None, with_column: bool = True):
#     code_info_dict = dict()
#     date_data_dir = Path(f'data/{data_category}/date')
#     year_dirs = os.listdir(date_data_dir)
#     if start_year != None :
#         year_dirs = list(filter(lambda x: int(x) >= start_year, year_dirs))

#     # get column names
#     column_names = get_column_names(data_category)
#     if picked_columns != None:
#         picked_columns_idx = list(map(lambda x: column_names.index(x), picked_columns))

#     for year_dir in year_dirs:
#         year = year_dir
#         year_dir_path = date_data_dir / year_dir
#         file_names = os.listdir(year_dir_path)
#         for file_name in file_names:
#             file_path = f'{year_dir_path}/{file_name}'
#             with open(file_path) as fp:
#                 try:
#                     lines = list(csv.reader(fp))
#                 except:
#                     print(f'error reading {file_path}')
#                     continue
#                 for line in lines[1:]:
#                     code = line[0]
#                     if picked_columns != None:
#                         line = [line[idx] for idx in picked_columns_idx]
#                     if code_info_dict.get(code) == None:
#                         code_info_dict[code] = []
#                     code_info_dict[code].append(line)
#         print(f'finish year {year}')
#     if picked_columns != None:
#         column_names = picked_columns
#     code_dir_path = Path(f'data/{data_category}/code')
#     if start_year != None:
#         code_dir_path = Path(f'data/{data_category}/code_from_{start_year}')
#     if not code_dir_path.exists():
#         code_dir_path.mkdir(parents = True, exist_ok = True)
#     for code, daily_info in code_info_dict.items():
#         sorted_daily_info = sorted(daily_info, key=lambda d: d[1])
#         if with_column:
#             sorted_daily_info = [column_names] + sorted_daily_info
#         with open(code_dir_path / f'{code}.csv', 'w') as fp:
#             csv.writer(fp).writerows(sorted_daily_info)

# def process_null_date_in_date_data(data_category: str, start_year: int):
#     date_data_dir = Path(f'data/{data_category}/date')
#     year_dirs = os.listdir(date_data_dir)
#     if start_year != None :
#         year_dirs = list(filter(lambda x: int(x) >= start_year, year_dirs))

#     for year_dir in year_dirs:
#         year_dir_path = date_data_dir / year_dir
#         file_names = os.listdir(year_dir_path)
#         for file_name in file_names:
#             file_path = f'{year_dir_path}/{file_name}'
#             with open(file_path, 'r') as fp:
#                 lines = fp.readlines()
#             with open(file_path, 'w') as fp:
#                 fp.writelines(replace_null_with_dash(lines))

#     # file_path = 'tmp/2024/2024-01-12.csv'
#     # with open(file_path, 'r') as fp:
#     #     lines = fp.readlines()
#     # with open(file_path, 'w') as fp:
#     #     processed_lines = replace_null_with_dash(lines)
#     #     fp.writelines(processed_lines)


def remove_tej_stock_name_column(fila_path: str, new_file_path: str):
    with open(fila_path, "r") as fp:
        csv_fp = csv.reader(fp)
        counter = 0
        lines = []
        for line in csv_fp:
            line.pop(1)
            for i in range(0, len(line)):
                line[i] = line[i].strip()
            lines.append(line)
            if len(lines) > 1000000:
                csv.writer(open(new_file_path, "a")).writerows(lines)
                lines = []
                counter += 1
                print(f"finish {1000000*counter} lines")
                gc.collect()
        if len(lines) > 0:
            csv.writer(open(new_file_path, "a")).writerows(lines)


# def process_quarter_finance(origin_file_path: Path, processed_file_path: Path):
#     def month_to_quarter(month):
#         return str((int(month) - 1) // 3 + 1)

#     code_to_eps = dict()
#     with open(origin_file_path) as fp:
#         lines = list(csv.reader(fp))
#         column_names = get_column_names("finance_report")
#         eps_column_idx = column_names.index("每股盈餘")
#         recurring_eps_column_idx = column_names.index("常續性EPS")

#         for line in lines[1:]:
#             if len(line) != len(column_names):
#                 raise ValueError(
#                     f"finance-report data length {len(line)} not equal to column length {len(column_names)}"
#                 )
#             line = replace_null_with_dash(line)
#             code_with_name, year_month, eps, recurring_eps = (
#                 line[0],
#                 line[1],
#                 line[eps_column_idx],
#                 line[recurring_eps_column_idx],
#             )
#             code = code_with_name.split(" ")[0]

#             year, quarter = year_month[:4], month_to_quarter(year_month[4:])
#             if code_to_eps.get(code) == None:
#                 code_to_eps[code] = [
#                     ["code", "year", "quarter", "eps", "recurring_eps"]
#                 ]
#             code_to_eps[code].append([code, year, quarter, eps, recurring_eps])

#     for k, v in code_to_eps.items():
#         rows = [v.pop(0)]
#         v.sort(key=lambda d: int(d[1] + d[2]))
#         rows.extend(v)
#         code_to_eps[k] = rows

#     # save code_to_eps as json file
#     with open(processed_file_path, "w") as fp:
#         json.dump(code_to_eps, fp)


def process_month_finance():
    def month_to_quarter(month):
        return str((int(month) - 1) // 3 + 1)

    code_to_eps = dict()
    with open("data/finance/processed-month.csv") as fp:
        lines = list(csv.reader(fp))
        column_name = lines[0]
        accumulative_eps_column_idx = column_name.index("累計每股稅後盈餘(WA)")
        for line in lines[1:]:
            code, year_month, accumulative_eps = (
                line[0],
                line[1],
                line[accumulative_eps_column_idx],
            )
            if accumulative_eps == "-":
                continue
            year, quarter = year_month[:4], month_to_quarter(year_month[4:6])
            if code_to_eps.get(code) == None:
                code_to_eps[code] = [["code", "year", "quarter", "accumulative_eps"]]
            code_to_eps[code].append([code, year, quarter, accumulative_eps])

    for k, v in code_to_eps.items():
        rows = [v.pop(0)]
        v.sort(key=lambda d: int(d[1] + d[2]))
        rows.extend(v)
        code_to_eps[k] = rows
    # save code_to_eps as json file
    with open("data/finance/month.json", "w") as fp:
        json.dump(code_to_eps, fp)


def get_eps_from_accumulative_eps():
    with open("data/finance/month.json", "r") as fp:
        month_profit = json.load(fp)
    code_eps_dict = dict()
    for code, rows in month_profit.items():
        if len(rows) < 3:
            continue
        q4_eps = float(rows[2][3]) - float(rows[1][3])
        code_eps_dict[code] = q4_eps
    with open("data/finance/eps/q4_eps.json", "w") as fp:
        json.dump(code_eps_dict, fp)


def append_price_data_from_fin_mind():
    data_dir = os.path.join("data", "append_data")
    original_dir = os.path.join("data", "code", "non_adjusted")

    file_names = os.listdir(data_dir)
    for file_name in file_names:
        with open(os.path.join(data_dir, file_name)) as fp:
            append_data = list(csv.reader(fp))
        for row in append_data:
            row[5] = int(float(row[5]) / 1000)
            row[6] = int(float(row[6]) / 1000)
        with open(os.path.join(original_dir, file_name)) as fp:
            original_data = list(csv.reader(fp))
        original_data.extend(append_data)
        with open(f"data/code/non_adjusted_appended/{file_name}", "w") as file:
            writer = csv.writer(file)
            writer.writerows(original_data)


def generate_tej_col_files():
    cols = {
        "daily_price": daily_price_cols.split(","),
        "adjusted_daily_price": adjusted_daily_price_cols.split(","),
        "chip": chip_cols.split(","),
        "month_profit": month_profit_cols.split(","),
        "finance_report": finance_report_cols.split(","),
        "adjusted_price": adjusted_eps_cols.split(","),
    }
    with open("data/tej_col_names.json", "w", encoding="utf-8") as fp:
        json.dump(cols, fp, ensure_ascii=False, indent=4)


def remove_invalid_row_1_from_daily_price():
    original_dir = Path("data/daily_price/code_from_2020")

    file_names = os.listdir(original_dir)
    for file_name in file_names:
        with open(original_dir / file_name, "r") as fp:
            lines = list(csv.reader(fp))
            if "-" in lines[1]:
                lines.pop(1)
        with open(original_dir / file_name, "w") as file:
            writer = csv.writer(file)
            writer.writerows(lines)
