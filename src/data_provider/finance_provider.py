from datetime import datetime, date
from pathlib import Path

from src.data_provider.base_provider import BaseProvider
from src.utils.common import (
    DataCategory,
    Instrument,
    Market,
    DataCategoryColumn,
    TimeFrame
)


class FinanceProvider(BaseProvider):
    def __init__(
        self,
        market: Market,
        instrument: Instrument,
        data_category: DataCategory,
        data_dir: Path,
        lazy_loading: bool = True,
    ):
        super().__init__(
            market=market,
            instrument=instrument,
            data_category=data_category,
            data_dir=data_dir,
            unit=TimeFrame.Quarterly,
            column_names=DataCategoryColumn.get_columns(data_category),
            lazy_loading=lazy_loading,
        )

    # def to_dataframe(self, data):
    #     df = pd.DataFrame(data, columns=DailyPriceProvider.column_names,).astype(
    #         {
    #             "code": "str",
    #             "date": "datetime64[ns]",
    #             "open": "float64",
    #             "high": "float64",
    #             "low": "float64",
    #             "close": "float64",
    #             "volume": "float64",
    #             "trading_value": "float64",
    #             "total_stocks": "int",
    #         }
    #     )
    #     df["date"] = pd.to_datetime(df["date"])

    #     return df
