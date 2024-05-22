from datetime import datetime, timedelta
from numpy.lib.stride_tricks import sliding_window_view
from typing import List
import numpy as np
import json

class DataTransformer:
    def get_local_ex(low: np.ndarray, high: np.ndarray):
        argmin_array = sliding_window_view(low, 3, axis=0).argmin(axis=2)
        # item == 1 means it is a local minimum (high-low-high)
        min_array = argmin_array == 1
        # the first row can not be a local minimum, the last row can
        min_array = np.concatenate([
            np.full((1, low.shape[1]), False),
            min_array,
            np.reshape(argmin_array[-1] == 2, (1, low.shape[1])),
        ], axis=0)

        argmax_array = sliding_window_view(high, 3, axis=0).argmax(axis=2)
        max_array = argmax_array == 1
        max_array = np.concatenate([
            np.full((1, high.shape[1]), False),
            max_array,
            np.reshape(argmax_array[-1] == 2, (1, high.shape[1])),
        ], axis=0)

        # # remove noise
        # prev_signal = np.full((low.shape[1]), 0.0) # 0: no signal, 1: max, 2: min
        # prev_signal_idx = np.full((low.shape[1]), -1)
        # prev_signal_value = np.full((low.shape[1]), 0.0)
        # for i in range(1, len(min_array)):
        #     cur_max_signal = max_array[i]
        #     cur_max_signal_value = high[i]
        #     redundant_max_signal = (cur_max_signal == True) & (prev_signal == 1) & (cur_max_signal_value > prev_signal_value)
        #     max_array[prev_signal_idx[redundant_max_signal], redundant_max_signal] = False

        #     cur_min_signal = min_array[i]
        #     cur_min_signal_value = low[i]
        #     redundant_min_signal = (cur_min_signal == True) & (prev_signal == 2) & (cur_min_signal_value < prev_signal_value)
        #     min_array[prev_signal_idx[redundant_min_signal], redundant_min_signal] = False
            
        #     prev_signal[cur_max_signal == True] = 1
        #     prev_signal[cur_min_signal == True] = 2
        #     prev_signal_idx[(cur_max_signal == True) | (cur_min_signal == True)] = i
        #     prev_signal_value[cur_max_signal == True] = cur_max_signal_value[cur_max_signal == True]
        #     prev_signal_value[cur_min_signal == True] = cur_min_signal_value[cur_min_signal == True]
  
        return min_array, max_array


    def get_middle_ex(low: np.ndarray, high: np.ndarray, level: int = 1):
        def transform_middle_ex(local_array: np.ndarray, value: np.ndarray, ex_type: str):
            for i in range(value.shape[1]):
                local_filter = local_array[:, i]
                local_idx = np.argwhere(local_filter == True).reshape(-1)
                local_value = value[local_idx, i]
                if len(local_value) < 3:
                    continue
                if ex_type == "min":
                    arg_array = sliding_window_view(local_value, 3, axis=0).argmin(axis=1)
                elif ex_type == "max":
                    arg_array = sliding_window_view(local_value, 3, axis=0).argmax(axis=1)
                middle_array = arg_array == 1
                middle_array = np.concatenate([[False], middle_array, [arg_array[-1]==2]], axis=0)
                middle_idx = local_idx[middle_array]
                local_filter[:] = False
                local_filter[middle_idx] = True

        min_array, max_array = DataTransformer.get_local_ex(low, high)
        while level != 0:
            transform_middle_ex(min_array, low, "min")
            transform_middle_ex(max_array, high, "max")

            level -= 1

        return min_array, max_array


    def get_legacy_signal_one(trading_dates: List[datetime], trading_codes: List[str]):
        with open("strategy_one.json", "r") as fp:
            signal_dict = json.load(fp)

        signal_array = np.full((len(trading_dates), len(trading_codes)), 0, dtype=int)
        for code, signals in signal_dict.items():
            code_idx = trading_codes.index(code)
            for signal in signals:
                try:
                    date_idx = trading_dates.index(datetime.fromisoformat(signal["signal_date"]))
                except ValueError:
                    continue
                signal_array[date_idx, code_idx] += 1
        
        return signal_array


    def get_signal_one(
        config: dict,
        close_array: np.ndarray,
        high_array: np.ndarray,
        low_array: np.ndarray,
        volume_array: np.ndarray,
        relative_strength_sma_array: np.ndarray,
        dates: List[datetime] = None,
    ):
        strategy_one_config = config["strategy_one"]
        volume_avg_time_window = strategy_one_config["volume_avg_time_window"]
        volume_avg_threshold = strategy_one_config["volume_avg_threshold"]
        up_min_ratio = strategy_one_config["up_min_ratio"]
        up_time_window = strategy_one_config["up_time_window"]
        down_max_ratio = strategy_one_config["down_max_ratio"]
        down_max_time_window = strategy_one_config["down_max_time_window"]
        consolidation_time_window = strategy_one_config["consolidation_time_window"]
        breakthrough_fuzzy = strategy_one_config["breakthrough_fuzzy"]
        rs_threshold = strategy_one_config["rs_threshold"]

        local_min_array, local_max_array = DataTransformer.get_middle_ex(low_array, high_array)
        mark_array = np.full_like(close_array, 0, dtype=int)
        signal_array = np.full_like(close_array, 0, dtype=int)
        for code_idx in range(local_max_array.shape[1]):
            local_min, local_max, close, high, low, volume, relative_strength_sma = local_min_array[:, code_idx], local_max_array[:, code_idx], close_array[:, code_idx], high_array[:, code_idx], low_array[:, code_idx], volume_array[:, code_idx], relative_strength_sma_array[:, code_idx]
            all_local_min_idx_list, all_local_max_idx_list = np.argwhere(local_min == True).reshape(-1), np.argwhere(local_max == True).reshape(-1)
            
            # > up_time_window (上升一段時間)
            local_min_idx_list = all_local_min_idx_list[all_local_min_idx_list > up_time_window]
            local_max_idx_list = all_local_max_idx_list[all_local_max_idx_list > up_time_window]
            for i, local_max_idx in enumerate(local_max_idx_list):
                # if i == 218 - (all_local_max_idx_list <= up_time_window).sum():
                    # print("debug")

                local_max_value = high[local_max_idx]
                prev_low = low[local_max_idx - up_time_window : local_max_idx + 1].min()
                

                # 上升超過一定幅度
                if (local_max_value - prev_low) / prev_low < up_min_ratio:
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
                    continue

                # 盤整 & 突破
                breakthrough_idx = next_low_idx
                breakthrough_points = np.argwhere(close >= local_max_value)
                breakthrough_points = breakthrough_points[breakthrough_points >= breakthrough_idx]
                if len(breakthrough_points) == 0:
                    break

                breakthrough_point_idx = breakthrough_points[0]
                if breakthrough_point_idx - local_max_idx < consolidation_time_window:
                    continue

                if relative_strength_sma[breakthrough_point_idx] < rs_threshold:
                    continue

                mark_array[local_max_idx, code_idx] = breakthrough_point_idx
                signal_array[breakthrough_point_idx, code_idx] += 1
        
        return signal_array, mark_array


    def get_relative_strength(codes: List[str], close: np.ndarray, market_index_close: np.ndarray, time_period: int = 1):
        close_change = np.diff(close, n=time_period, axis=0) / close[:-time_period]
        market_index_close_change = np.diff(market_index_close, n=time_period, axis=0) / market_index_close[:-time_period]
        relative_strength_rate = close_change / market_index_close_change
        relative_strength = np.full_like(relative_strength_rate, 0, dtype=float)
        for pct in range(1, 100):
            px = np.nanpercentile(relative_strength_rate, pct, axis=1)
            relative_strength[relative_strength_rate > px.reshape(-1, 1)] = pct
        for i in range(1, 10):
            pct = 99 + (i*0.1)
            px = np.nanpercentile(relative_strength_rate, pct, axis=1)
            relative_strength[relative_strength_rate > px.reshape(-1, 1)] = pct
            
        return np.concatenate([np.full((time_period, close.shape[1]), np.nan), relative_strength], axis=0)