from datetime import datetime, timedelta
from typing import Dict
from pathlib import Path
import numpy as np
import json

from src.data_provider.base_provider import BaseProvider
from src.data_store.data_store import DataStore
from src.platform.platform import Platform
from src.utils.common import DataCategory, Instrument, Market, OrderSide, DataColumn, TechnicalIndicator
from src.utils.order import CoverOrderRecord, OrderRecord
from src.utils.utils import NumpyEncoder
from src.strategy.strategy import Strategy
from src.data_transformer.data_transformer import DataTransformer


class TrendStrategy(Strategy):
    def __init__(
        self,
        platform: Platform,
        data_store: DataStore,
        cash: float,
        config: Dict[str, any],
        log_level: str = "INFO",
    ):
        super().__init__(platform, data_store, cash, config, log_level)

    def prepare_data(self, start_date: datetime, end_date: datetime, result_dir: Path, full_record: bool):
        data_start_date = start_date - timedelta(days=250)  # extract more data for technical indicator
        [self.open_, self.high_, self.low_, self.close_, self.volume_], dates, codes = self.data_store.get_data(
            market=Market.TW,
            instrument=Instrument.Stock,
            data_category=DataCategory.Daily_Price,
            data_columns=[DataColumn.Open, DataColumn.High, DataColumn.Low, DataColumn.Close, DataColumn.Volume],
            start_date=data_start_date,
            end_date=end_date,
        )

        market_index_, _, market_index_codes = self.data_store.get_data(
            market=Market.TW,
            instrument=Instrument.StockIndex,
            data_category=DataCategory.Market_Index,
            data_columns=[DataColumn.Close],
            selected_codes=["Y9999"],
            start_date=data_start_date,
            end_date=end_date,
        )

        self.foreign_total_holdings_ratio_, self.local_self_holdings_ratio_, self.local_investor_holdings_ratio_ = self.data_store.get_aligned_data(
            target_dates=dates,
            target_codes=codes,
            market=Market.TW,
            instrument=Instrument.Stock,
            data_category=DataCategory.Chip,
            data_columns=[
                DataColumn.Foreign_Total_Holdings_Ratio,
                DataColumn.Local_Self_Holdings_Ratio,
                DataColumn.Local_Investor_Holdings_Ratio,
            ],
            fill_missing_date=False
        )
        self.close_sma_ = self.data_store.get_technical_indicator(TechnicalIndicator.SMA, self.close_, self.config["strategy_one"]["close_above_sma_period"])
        self.close_sam_5_ = self.data_store.get_technical_indicator(TechnicalIndicator.SMA, self.close_, 5)
        self.close_sam_10_ = self.data_store.get_technical_indicator(TechnicalIndicator.SMA, self.close_, 10)
        self.close_sam_20_ = self.data_store.get_technical_indicator(TechnicalIndicator.SMA, self.close_, 20)
        
        self.foreign_total_holdings_ratio_sma_ = self.data_store.get_technical_indicator(TechnicalIndicator.SMA, self.foreign_total_holdings_ratio_, self.config["chip_strategy"]["foreign_total_holdings_ratio_sma_period"])
        self.local_self_holdings_ratio_sma_ = self.data_store.get_technical_indicator(TechnicalIndicator.SMA, self.local_self_holdings_ratio_, self.config["chip_strategy"]["local_self_holdings_ratio_sma_period"])
        self.local_investor_holdings_ratio_sma_ = self.data_store.get_technical_indicator(TechnicalIndicator.SMA, self.local_investor_holdings_ratio_, self.config["chip_strategy"]["local_investor_holdings_ratio_sma_period"])

        twse_code = "Y9999"
        twse_code_idx = market_index_codes.index(twse_code)
        self.market_index_ = market_index_[:, [twse_code_idx]]

        self.relative_strength_ = DataTransformer.get_relative_strength(self.close_, self.market_index_)
        self.relative_strength_sma_ = self.data_store.get_technical_indicator(
            TechnicalIndicator.SMA, self.relative_strength_, self.config["strategy_one"]["rs_sma_period"])
        self.market_index_sma_ = self.data_store.get_technical_indicator(
            TechnicalIndicator.SMA, self.market_index_, self.config["strategy_one"]["market_index_sma_period"])

        # local min & max array
        self.signal_one_, analyze_material = self.get_signal_with_filter()
        analyze_material.update({
            "start_date": data_start_date.isoformat(),
            "end_date": end_date.isoformat()
        })

        if full_record:
            with open(result_dir / "analyze_material.json", "w") as fp:
                json.dump(analyze_material, fp, cls=NumpyEncoder)


        self.trading_dates = self.slice_data(dates, start_date, end_date)
        self.trading_codes = codes

    def get_signal_with_filter(self):
        strategy_one_config = self.config["strategy_one"]
        volume_avg_time_window = strategy_one_config["volume_avg_time_window"]
        volume_avg_threshold = strategy_one_config["volume_avg_threshold"]
        up_min_ratio = strategy_one_config["up_min_ratio"]
        up_time_window = strategy_one_config["up_time_window"]
        down_max_ratio = strategy_one_config["down_max_ratio"]
        down_max_time_window = strategy_one_config["down_max_time_window"]
        consolidation_time_window = strategy_one_config["consolidation_time_window"]
        breakthrough_fuzzy = strategy_one_config["breakthrough_fuzzy"]
        rs_threshold = strategy_one_config["rs_threshold"]
        signal_threshold = strategy_one_config["signal_threshold"]
        volume_short_sma_period = strategy_one_config["volume_short_sma_period"]
        volume_long_sma_period = strategy_one_config["volume_long_sma_period"]

        # prepare data
        # get true high & low
        true_high, true_low = DataTransformer.get_true_high_and_low(
            high=self.high_,
            low=self.low_,
            close=self.close_,
        )
        local_min_array, local_max_array = DataTransformer.get_local_ex(low=true_low, high=true_high)
        volume_short_sma_array = self.data_store.get_technical_indicator(TechnicalIndicator.SMA, self.volume_, volume_short_sma_period)
        volume_long_sma_array = self.data_store.get_technical_indicator(TechnicalIndicator.SMA, self.volume_, volume_long_sma_period)

        failed_breakthrough = np.full_like(self.close_, 0, dtype=int)
        signal_array = np.full_like(self.close_, 0, dtype=int)
        signal_objects = []
        failure_breakthrough_objects = []
        total_signal = 0
        self.filtered_reason = local_max_array.astype(int).copy()

        debug_code = ''
        debug_time = ''
        for code_idx in range(local_max_array.shape[1]):
            if code_idx == debug_code:
                code_idx

            local_min = local_min_array[:, code_idx]
            local_max = local_max_array[:, code_idx]
            close = self.close_[:, code_idx]
            high = self.high_[:, code_idx]
            low = self.low_[:, code_idx]
            volume = self.volume_[:, code_idx]
            close_sma = self.close_sma_[:, code_idx]
            relative_strength_sma = self.relative_strength_sma_[:, code_idx]
            market_index_close = self.market_index_[:, 0]
            market_index_sma = self.market_index_sma_[:, 0]
            foreign_total_holdings_ratio = self.foreign_total_holdings_ratio_[:, code_idx]
            foreign_total_holdings_ratio_sma = self.foreign_total_holdings_ratio_sma_[:, code_idx]
            local_self_holdings_ratio = self.local_self_holdings_ratio_[:, code_idx]
            local_self_holdings_ratio_sma = self.local_self_holdings_ratio_sma_[:, code_idx]
            local_investor_holdings_ratio = self.local_investor_holdings_ratio_[:, code_idx]
            local_investor_holdings_ratio_sma = self.local_investor_holdings_ratio_sma_[:, code_idx]
            volume_short_sma = volume_short_sma_array[:, code_idx]
            volume_long_sma = volume_long_sma_array[:, code_idx]

            all_local_min_idx_list = np.argwhere(local_min == True).reshape(-1)
            all_local_max_idx_list = np.argwhere(local_max == True).reshape(-1)

            # > up_time_window (上升一段時間)

            local_min_idx_list = all_local_min_idx_list[all_local_min_idx_list > up_time_window]
            local_max_idx_list = all_local_max_idx_list[all_local_max_idx_list > up_time_window]

            for i, local_max_idx in enumerate(local_max_idx_list):
                if debug_time == local_max_idx:
                    local_max_idx

                total_signal += 1
                local_max_value = high[local_max_idx]
                prev_low = low[local_max_idx - up_time_window: local_max_idx + 1].min()

                # 上升超過一定幅度
                if self.filter_up_min_ratio(
                    x=local_max_idx,
                    y=code_idx,
                    local_max_value=local_max_value,
                    prev_low=prev_low,
                    up_min_ratio=up_min_ratio
                ):
                    continue

                # 下降 (找到下一個最低點)
                next_low_array = local_min_idx_list[np.argwhere(local_min_idx_list > local_max_idx).reshape(-1)]
                if len(next_low_array) == 0 or next_low_array[0] > local_max_idx + down_max_time_window:
                    next_low_idx = local_max_idx + low[local_max_idx:local_max_idx+down_max_time_window+1].argmin()
                    next_low_value = low[next_low_idx]
                else:
                    next_low_idx = next_low_array[0]
                    next_low_value = low[next_low_idx]

                # 下降超過最大幅度則跳過
                if self.filter_down_max_ratio(
                    x=local_max_idx,
                    y=code_idx,
                    local_max_value=local_max_value,
                    next_low_value=next_low_value,
                    down_max_ratio=down_max_ratio
                ):
                    continue

                # 盤整 & 突破
                breakthrough_idx = next_low_idx
                breakthrough_points = np.argwhere(close >= local_max_value*(1 + breakthrough_fuzzy))
                breakthrough_points = breakthrough_points[breakthrough_points >= breakthrough_idx]

                if self.filter_breakthrough_point(
                    x=local_max_idx,
                    y=code_idx,
                    number_of_points=len(breakthrough_points)
                ):
                    continue
                
                breakthrough_point_idx = breakthrough_points[0]
                signal_object = {
                    "code_idx": code_idx,
                    "signal_idx": breakthrough_point_idx,
                    "start_max_idx": local_max_idx,
                    "middle_min_idx": next_low_idx,
                }
                failure_breakthrough_objects.append(signal_object)

                if self.filter_close_above_sma(
                    x=local_max_idx,
                    y=code_idx,
                    close=close[breakthrough_point_idx],
                    close_sma=close_sma[breakthrough_point_idx],
                ):
                    continue

                if self.filter_consolidation_time_window(
                    x=local_max_idx,
                    y=code_idx,
                    local_max_idx=local_max_idx,
                    breakthrough_point_idx=breakthrough_point_idx,
                    consolidation_time_window=consolidation_time_window
                ):
                    continue
                
                if self.filter_relative_strength(
                    x=local_max_idx,
                    y=code_idx,
                    relative_strength_sma=relative_strength_sma[breakthrough_point_idx],
                    rs_threshold=rs_threshold
                ):
                    continue
                

                if self.filter_market_index(
                    x=local_max_idx,
                    y=code_idx,
                    market_index_close=market_index_close[breakthrough_point_idx],
                    market_index_sma=market_index_sma[breakthrough_point_idx]
                ):
                    continue

                if self.filter_chip(
                    x=local_max_idx,
                    y=code_idx,
                    local_investor_holdings_ratio=local_investor_holdings_ratio[breakthrough_point_idx-1],
                    local_investor_holdings_ratio_sma=local_investor_holdings_ratio_sma[breakthrough_point_idx-1]
                ):
                    continue

                if self.filter_volume(
                    x=local_max_idx,
                    y=code_idx,
                    volume = volume[breakthrough_point_idx],
                    volume_short_sma=volume_short_sma[breakthrough_point_idx],
                    volume_long_sma=volume_long_sma[breakthrough_point_idx]
                ):
                    continue


                signal_array[breakthrough_point_idx, code_idx] += 1
                signal_objects.append(signal_object)
                failure_breakthrough_objects.pop()


        self.filter_signal_threshold(signal_array, signal_threshold)
        # reverse filtered_reason_mapper
        self.filtered_reason_mapper = {filter_name: filter_idx for filter_idx, filter_name in self.filtered_reason_mapper.items()}
        analyze_material = {
            "local_min": local_min_array,
            "local_max": local_max_array,
            "signal_objects": signal_objects,
            "failure_breakthrough_objects": failure_breakthrough_objects,
            "filtered_reason": self.filtered_reason,
            "filtered_reason_mapper": self.filtered_reason_mapper,
            "filtered_signal_count": self.filtered_signal_count
        }
        return signal_array, analyze_material

    def filtered_up_a_while(self, x, y, up_time_window):
        return x > up_time_window

    def filter_up_min_ratio(self, x, y, local_max_value, prev_low, up_min_ratio):
        # 上升超過一定幅度
        return (local_max_value - prev_low) / prev_low >= up_min_ratio

    def filter_down_max_ratio(self, x, y, local_max_value, next_low_value, down_max_ratio):
        return (local_max_value - next_low_value) / local_max_value <= down_max_ratio

    def filter_breakthrough_point(self, x, y, number_of_points):
        return number_of_points != 0

    def filter_consolidation_time_window(self, x, y, local_max_idx, breakthrough_point_idx, consolidation_time_window):
        return breakthrough_point_idx - local_max_idx >= consolidation_time_window
    
    def filter_relative_strength(self, x, y, relative_strength_sma, rs_threshold):
        return relative_strength_sma >= rs_threshold
    
    def filter_market_index(self, x, y, market_index_close, market_index_sma):
        return market_index_close >= market_index_sma

    def filter_chip(self, x, y, local_investor_holdings_ratio, local_investor_holdings_ratio_sma):
        return not np.isnan(local_investor_holdings_ratio_sma) and local_investor_holdings_ratio >= local_investor_holdings_ratio_sma

    def filter_volume(self, x, y, volume, volume_short_sma, volume_long_sma):
        return volume_short_sma >= volume_long_sma

    def filter_close_above_sma(self, x, y, close, close_sma):
        return close >= close_sma

    def filter_signal_threshold(self, signal_array, signal_threshold):
        signal_threshold_mask = (signal_array <= signal_threshold) & (signal_array > 0)
        self.filtered_signal_count["filter_singal_threshold"] = int(signal_threshold_mask.sum())
        self.filtered_reason[signal_threshold_mask] = 7
        signal_array[signal_threshold_mask] = 0
    
    def step(self, trading_date: datetime):
        super().step(trading_date)

        strategy_one_config = self.config["strategy_one"]
        holding_days = strategy_one_config["holding_days"]
        stop_loss_ratio = strategy_one_config["stop_loss_ratio"]
        signal_threshold = strategy_one_config["signal_threshold"]

        codes = self.get_trading_codes()
        open_price = self.open_[-1]
        high_price = self.high_[-1]
        low_price = self.low_[-1]
        close_price = self.close_[-1]
        signal_one = self.signal_one_[-1]
        sma_5 = self.close_sam_5_[-1]
        sma_10 = self.close_sam_10_[-1]
        sma_20 = self.close_sam_20_[-1]

        # adjust position
        for order_record in list(self.holdings.values()):
            code = order_record.order.code
            code_idx = codes.index(code)
            stop_loss_price = order_record.info["stop_loss_price"]

            if low_price[code_idx] < stop_loss_price:
                # 處理跌停板的狀況
                real_stop_loss_price = min(stop_loss_price, high_price[code_idx])
                self.cover_order(
                    order_record.order.order_id,
                    real_stop_loss_price,
                    order_record.order.volume,
                    "stop_loss"
                )
                continue

            order_record.info["holding_days"] += 1

            # if close_price[code_idx] < sma_20[code_idx] or sma_10[code_idx] < sma_20[code_idx]:
            #     self.cover_order(
            #         order_record.order.order_id,
            #         close_price[code_idx],
            #         order_record.order.volume,
            #         "sma_cross"
            #     )
            #     continue

            if order_record.info["holding_days"] >= holding_days:
                self.cover_order(
                    order_record.order.order_id,
                    close_price[code_idx],
                    order_record.order.volume,
                    "holding_days"
                )


        fixed_cash = 100000
        # make decision
        for idx, signal in enumerate(signal_one):
            if signal == 0:
                continue

            # # filter eps
            # if not (self.recurring_eps_[-1, idx] >= self.recurring_eps_[-120:, idx]).all():
            #     continue

            # min_eps = self.recurring_eps_[-120:, idx].min()
            # if (self.recurring_eps_[-1, idx] - self.recurring_eps_[-120:, idx].min()) / np.abs(min_eps) < 0.1:
            #     continue

            code = codes[idx]
            price = close_price[idx]
            volume = signal * fixed_cash // price
            stop_loss_price = price * (1 - stop_loss_ratio)
            self.place_order(
                market=Market.TW,
                instrument=Instrument.Stock,
                code=code,
                price=price,
                volume=volume,
                side=OrderSide.Buy,
                info={"stop_loss_price": stop_loss_price, "holding_days": 0}
            )
