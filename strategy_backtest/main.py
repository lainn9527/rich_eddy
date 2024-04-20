import sys
from pathlib import Path


path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

import datetime
import io
import os

import backtrader as bt

# Import the backtrader platform
from backtrader.feeds import GenericCSVData

from src.utils import PriceType, read_daily_data, read_daily_data_by_codes
from strategy_backtest.sma_strategy import SmaStrategy


if __name__ == "__main__":
    # mod_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    # with open(os.path.join(mod_path, 'tradable_stock', 'above_4000_250.txt')) as fp:
    #     tradable_stock_codes = fp.readline().split(',')
    #     tradable_stock_codes = list(map(str.strip, tradable_stock_codes))
    FILE_DIR_PATH = Path("data/daily_price/code_from_2020")
    FILENAME = "2376.csv"
    adjusted_data_path = FILE_DIR_PATH / FILENAME
    non_adjusted_data_path = FILE_DIR_PATH / FILENAME

    # code_to_pv = read_daily_data_by_codes(['2330', '2376'], 'daily_price')
    # Create a cerebro entity

    cerebro = bt.Cerebro()
    cerebro.addstrategy(SmaStrategy)

    adjusted_data = GenericCSVData(
        dataname=adjusted_data_path,
        nullvalue=1,
        dtformat=("%Y%m%d"),
        datetime=1,
        open=2,
        high=3,
        low=4,
        close=5,
        volume=6,
        openinterest=-1,
    )
    non_adjusted_data = GenericCSVData(
        dataname=non_adjusted_data_path,
        nullvalue=1,
        dtformat=("%Y%m%d"),
        datetime=1,
        open=2,
        high=3,
        low=4,
        close=5,
        volume=6,
        openinterest=-1,
    )
    ##
    file_names = os.listdir(FILE_DIR_PATH)
    # filter out bonds(可轉債), special stocks(特別股), ETF(ETF)
    all_codes = [file_name.split(".")[0] for file_name in file_names]
    codes = list(
        filter(
            lambda code: code.isdecimal() and int(code) >= 1000 and int(code) <= 9999,
            all_codes,
        )
    )
    for code in codes:
        FILE_PATH = FILE_DIR_PATH / f"{code}.csv"
        print(code)
        cerebro.adddata(
            GenericCSVData(
                dataname=FILE_PATH,
                nullvalue=0.0,
                dtformat=("%Y%m%d"),
                datetime=1,
                open=2,
                high=3,
                low=4,
                close=5,
                volume=6,
                openinterest=-1,
            ),
            code,
        )
    ##
    # Add the Data Feed to Cerebro

    cerebro.adddata(non_adjusted_data, PriceType.NON_ADJUSTED.value)
    cerebro.adddata(adjusted_data, PriceType.ADJUSTED.value)

    # Set our desired cash start
    cerebro.broker.setcash(1000.0)
    cerebro.broker.setcommission(commission=0.005)
    cerebro.broker.set_coc(True)

    # Print out the starting conditions
    print("Starting Portfolio Value: %.2f" % cerebro.broker.getvalue())

    # Run over everything
    cerebro.run()

    # Print out the final result
    print("Final Portfolio Value: %.2f" % cerebro.broker.getvalue())

    cerebro.plot()
