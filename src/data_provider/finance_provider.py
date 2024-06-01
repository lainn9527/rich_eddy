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
