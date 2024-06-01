from datetime import datetime, timedelta
from typing import Dict, List
import numpy as np

from src.data_provider.base_provider import BaseProvider
from src.data_store.data_store import DataStore
from src.platform.platform import Platform
from src.utils.common import DataCategory, Instrument, Market, OrderSide, DataColumn, TechnicalIndicator
from src.utils.order import CoverOrderRecord, OrderRecord
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
        

    def prepare_data(self, start_date: datetime, end_date: datetime):
        data_start_date = start_date - timedelta(days=250) # extract more data for technical indicator
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

        self.foreign_total_holdings_ratio_sma_ = self.data_store.get_technical_indicator(TechnicalIndicator.SMA, self.foreign_total_holdings_ratio_, self.config["chip_strategy"]["foreign_total_holdings_ratio_sma_period"])
        self.local_self_holdings_ratio_sma_ = self.data_store.get_technical_indicator(TechnicalIndicator.SMA, self.local_self_holdings_ratio_, self.config["chip_strategy"]["local_self_holdings_ratio_sma_period"])
        self.local_investor_holdings_ratio_sma_ = self.data_store.get_technical_indicator(TechnicalIndicator.SMA, self.local_investor_holdings_ratio_, self.config["chip_strategy"]["local_investor_holdings_ratio_sma_period"])

        twse_code = "Y9999"
        twse_code_idx = market_index_codes.index(twse_code)
        self.market_index_ = market_index_[:, [twse_code_idx]]

        self.relative_strength_ = DataTransformer.get_relative_strength(self.close_, self.market_index_)
        self.relative_strength_sma_ = self.data_store.get_technical_indicator(TechnicalIndicator.SMA, self.relative_strength_, self.config["strategy_one"]["rs_sma_period"])
        self.market_index_sma_ = self.data_store.get_technical_indicator(TechnicalIndicator.SMA, self.market_index_, self.config["strategy_one"]["market_index_sma_period"])
        
        # local min & max array
        self.signal_one_, analyze_material = self.get_signal()
        analyze_material.update({ "start_date": data_start_date.isoformat(), "end_date": end_date.isoformat() })
        
        self.trading_dates = self.slice_data(dates, start_date, end_date)
        self.trading_codes = codes

        return analyze_material

    def get_signal(self):
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

        # prepare data
        local_min_array, local_max_array = DataTransformer.get_local_ex(self.low_, self.high_)
        signal_array = np.full_like(self.close_, 0, dtype=int)
        signal_objects = []

        total_signal = 0
        market_index_filtered = 0
        up_min_ratio_filtered = 0
        down_max_ratio_filtered = 0
        breakthrough_points_filtered = 0
        consolidation_time_window_filtered = 0
        rs_threshold_filtered = 0
        chip_filtered = 0
        filtered_reason = local_max_array.astype(int).copy()

        for code_idx in range(local_max_array.shape[1]):
            local_min, local_max, close, high, low, volume, relative_strength_sma = local_min_array[:, code_idx], local_max_array[:, code_idx], self.close_[:, code_idx], self.high_[:, code_idx], self.low_[:, code_idx], self.volume_[:, code_idx], self.relative_strength_sma_[:, code_idx]
            market_index_close = self.market_index_[:, 0]
            market_index_sma = self.market_index_sma_[:, 0]
            foreign_total_holdings_ratio = self.foreign_total_holdings_ratio_[:, code_idx]
            foreign_total_holdings_ratio_sma = self.foreign_total_holdings_ratio_sma_[:, code_idx]
            local_self_holdings_ratio = self.local_self_holdings_ratio_[:, code_idx]
            local_self_holdings_ratio_sma = self.local_self_holdings_ratio_sma_[:, code_idx]
            local_investor_holdings_ratio = self.local_investor_holdings_ratio_[:, code_idx]
            local_investor_holdings_ratio_sma = self.local_investor_holdings_ratio_sma_[:, code_idx]


            all_local_min_idx_list, all_local_max_idx_list = np.argwhere(local_min == True).reshape(-1), np.argwhere(local_max == True).reshape(-1)
            
            # > up_time_window (上升一段時間)
            local_min_idx_list = all_local_min_idx_list[all_local_min_idx_list > up_time_window]
            local_max_idx_list = all_local_max_idx_list[all_local_max_idx_list > up_time_window]
            for i, local_max_idx in enumerate(local_max_idx_list):
                total_signal += 1
                
                local_max_value = high[local_max_idx]
                prev_low = low[local_max_idx - up_time_window : local_max_idx + 1].min()

                # 上升超過一定幅度
                if (local_max_value - prev_low) / prev_low < up_min_ratio:
                    up_min_ratio_filtered += 1
                    filtered_reason[local_max_idx, code_idx] = 6
                    continue
                
                # # 平均交易量大於 200
                # if volume[local_max_idx - volume_avg_time_window : local_max_idx + 1].mean() < volume_avg_threshold:
                #     continue

                # 下降 (找到下一個最低點)
                next_low_array = local_min_idx_list[np.argwhere(local_min_idx_list > local_max_idx).reshape(-1)]
                if len(next_low_array) == 0:
                    next_low_idx = local_max_idx + low[local_max_idx:local_max_idx+down_max_time_window+1].argmin()
                    next_low_value = low[next_low_idx]
                else:
                    next_low_idx = next_low_array[0]
                    next_low_value = low[next_low_idx]

                # 下降超過最大幅度則跳過
                if (local_max_value - next_low_value) / local_max_value > down_max_ratio:
                    down_max_ratio_filtered += 1
                    filtered_reason[local_max_idx, code_idx] = 2
                    continue

                # 盤整 & 突破
                breakthrough_idx = next_low_idx
                breakthrough_points = np.argwhere(close >= local_max_value*(1.03))
                breakthrough_points = breakthrough_points[breakthrough_points >= breakthrough_idx]
                if len(breakthrough_points) == 0:
                    breakthrough_points_filtered += 1
                    filtered_reason[local_max_idx, code_idx] = 3
                    continue
                
                breakthrough_point_idx = breakthrough_points[0]
                if breakthrough_point_idx - local_max_idx < consolidation_time_window:
                    consolidation_time_window_filtered += 1
                    filtered_reason[local_max_idx, code_idx] = 4
                    continue
                
                if relative_strength_sma[breakthrough_point_idx] < rs_threshold:
                    rs_threshold_filtered += 1
                    filtered_reason[local_max_idx, code_idx] = 5
                    continue

                if market_index_close[breakthrough_point_idx] < market_index_sma[breakthrough_point_idx]:
                    market_index_filtered += 1
                    filtered_reason[local_max_idx, code_idx] = 8
                    continue

                if not np.isnan(local_investor_holdings_ratio_sma[breakthrough_point_idx-1]) and local_investor_holdings_ratio[breakthrough_point_idx-1] < local_investor_holdings_ratio_sma[breakthrough_point_idx-1]:
                    chip_filtered += 1
                    filtered_reason[local_max_idx, code_idx] = 9
                    continue

                signal_array[breakthrough_point_idx, code_idx] += 1
                
                signal_objects.append({
                    "code_idx": code_idx,
                    "signal_idx": breakthrough_point_idx,
                    "start_max_idx": local_max_idx,
                    "middle_min_idx": next_low_idx,
                })


        signal_threshold_mask = (signal_array <= signal_threshold) & (signal_array > 0)
        signal_threshold_filtered = signal_threshold_mask.sum()
        filtered_reason[signal_threshold_mask] = 7
        signal_array[signal_threshold_mask] = 0

        print(f"up_min_ratio_filtered: {(100 * up_min_ratio_filtered/total_signal):.2f}")
        print(f"down_max_ratio_filtered: {(100 * down_max_ratio_filtered/total_signal):.2f}")
        print(f"breakthrough_points_filtered: {(100 * breakthrough_points_filtered/total_signal):.2f}")
        print(f"consolidation_time_window_filtered: {(100 * consolidation_time_window_filtered/total_signal):.2f}")
        print(f"rs_threshold_filtered: {(100 * rs_threshold_filtered/total_signal):.2f}")
        print(f"market_index_filtered: {(100 * market_index_filtered/total_signal):.2f}")
        print(f"signal_threshold_filtered: {(100 * signal_threshold_filtered/total_signal):.2f}")
        print(f"chip_filtered: {(100 * chip_filtered/total_signal):.2f}")

        analyze_material = {
            "local_min": local_min_array,
            "local_max": local_max_array,
            "signal_objects": signal_objects,
            "filtered_reason": filtered_reason,
            "filtered_reason_mapper": {
                1: "Signal",
                2: "Down max ratio",
                3: "No breakthrough points",
                4: "Consolidation time window",
                5: "RS threshold",
                6: "Up min ratio",
                7: "Signal threshold",
                8: "Market index",
                9: "Chip"
            }
        }
        return signal_array, analyze_material


    def step(self, trading_date: datetime):
        super().step(trading_date)

        strategy_one_config = self.config["strategy_one"]
        holding_days = strategy_one_config["holding_days"]
        stop_loss_ratio = strategy_one_config["stop_loss_ratio"]
        signal_threshold = strategy_one_config["signal_threshold"]

        codes = self.get_trading_codes()
        open_price, high_price, low_price, close_price, signal_one = self.open_[-1], self.high_[-1], self.low_[-1], self.close_[-1], self.signal_one_[-1]
        
        # adjust position
        for order_record in list(self.holdings.values()):
            code = order_record.order.code
            code_idx = codes.index(code)
            stop_loss_price = order_record.info["stop_loss_price"]

            if close_price[code_idx] < stop_loss_price:
                self.cover_order(
                    order_record.order.order_id,
                    stop_loss_price,
                    order_record.order.volume,
                    "stop_loss"
                )
                continue

            order_record.info["holding_days"] += 1
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
                info= { "stop_loss_price": stop_loss_price, "holding_days": 0 }
            )
