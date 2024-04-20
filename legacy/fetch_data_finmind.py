import pandas as pd
from FinMind.data import DataLoader


data_dir = "data/append_data"
api = DataLoader()
token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJkYXRlIjoiMjAyNC0wMS0wNyAyMzozOTowMCIsInVzZXJfaWQiOiJsYWlubjk1MjciLCJpcCI6IjQyLjYwLjI1NC4yMTQifQ.SDUdrQ-2lfYHTtCML-8FvJPV_6bk08ThnK6Psq5s0Gg"
api.login_by_token(api_token=token)

with open("tradable_stock/360_500_999.txt", "r") as file:
    tradable_stock_codes = file.read().splitlines()[0].split(",")
for code in tradable_stock_codes:
    df = api.taiwan_stock_daily(
        stock_id=code, start_date="2023-11-18", end_date="2024-01-07"
    )
    if len(df) == 0:
        continue
    df["date"] = df["date"].apply(lambda x: x.replace("-", ""))
    df.to_csv(
        f"{data_dir}/{code}.csv",
        index=False,
        header=False,
        columns=[
            "date",
            "open",
            "max",
            "min",
            "close",
            "Trading_Volume",
            "Trading_money",
        ],
    )
