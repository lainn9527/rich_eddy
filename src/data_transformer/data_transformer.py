from datetime import datetime, timedelta
from numpy.lib.stride_tricks import sliding_window_view
from typing import List
import numpy as np
import json
from src.utils.redis_client import RedisClient

class DataTransformer:
    def get_true_high_and_low(high: np.ndarray, low: np.ndarray, close: np.ndarray):
        true_high, true_low = np.copy(high), np.copy(low)
        true_high[1:] = np.maximum(high[1:], close[:-1])
        true_low[1:] = np.minimum(low[1:], close[:-1])
        
        return true_high, true_low
    

    def get_local_ex(low: np.ndarray, high: np.ndarray):
        argmin_array = sliding_window_view(low, 3, axis=0).argmin(axis=2)
        
        # item == 1 means it is a local minimum (high-low-high)
        min_array = argmin_array == 1
        
        # the first row can not be a local minimum, the last row can
        min_array = np.concatenate([
            np.full((1, low.shape[1]), False),
            min_array,
            np.full((1, low.shape[1]), False),
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
            debug = -1
            for i in range(value.shape[1]):
                if i == debug:
                    debug
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


    def get_correct_ex(low: np.ndarray, high: np.ndarray, level: int = 1):
        def transform_middle_ex(local_array: np.ndarray, value: np.ndarray, ex_type: str):
            abs_ex_ratio = 0.1
            debug = -1
            local_array = local_array.copy().astype(int)
            for i in range(value.shape[1]):
                if i == debug:
                    debug
                local_filter = local_array[:, i]
                local_idx = np.argwhere(local_filter == True).reshape(-1)
                local_value = value[local_idx, i]
                if len(local_value) < 3:
                    continue
                if ex_type == "min":
                    arg_array = sliding_window_view(local_value, 3, axis=0).argmin(axis=1)
                    abs_ex_filter = np.concatenate([[0.0], np.diff(local_value)]) / local_value < -abs_ex_ratio
                elif ex_type == "max":
                    arg_array = sliding_window_view(local_value, 3, axis=0).argmax(axis=1)
                    abs_ex_filter = np.concatenate([[0.0], np.diff(local_value)]) / local_value > abs_ex_ratio
                middle_array = arg_array == 1
                signal_detected_idx = local_idx[2:][middle_array] + 1
                augmented_middle_array = np.concatenate([[False], middle_array, [False]], axis=0)
                middle_idx = local_idx[augmented_middle_array]
                local_filter[:] = -1
                local_filter[middle_idx] = signal_detected_idx
                local_filter[local_idx[abs_ex_filter]] = local_idx[abs_ex_filter]
            return local_array

        min_array, max_array = DataTransformer.get_local_ex(low, high)
        while level != 0:
            min_array = transform_middle_ex(min_array, low, "min")
            max_array = transform_middle_ex(max_array, high, "max")
            level -= 1
        
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

    def get_middle_ex2(low: np.ndarray, high: np.ndarray, level: int = 1):
        def transform_middle_ex(local_array: np.ndarray, value: np.ndarray, ex_type: str):
            abs_ex_ratio = 0.1
            debug = -1
            for i in range(value.shape[1]):
                if i == debug:
                    debug
                local_filter = local_array[:, i]
                local_idx = np.argwhere(local_filter == True).reshape(-1)
                local_value = value[local_idx, i]
                if len(local_value) < 3:
                    continue
                if ex_type == "min":
                    arg_array = sliding_window_view(local_value, 3, axis=0).argmin(axis=1)
                    abs_ex_filter = np.concatenate([[0.0], np.diff(local_value)]) / local_value < -abs_ex_ratio
                elif ex_type == "max":
                    arg_array = sliding_window_view(local_value, 3, axis=0).argmax(axis=1)
                    abs_ex_filter = np.concatenate([[0.0], np.diff(local_value)]) / local_value > abs_ex_ratio
                middle_array = arg_array == 1
                middle_array = np.concatenate([[False], middle_array, [arg_array[-1]==2]], axis=0)
                middle_idx = local_idx[middle_array|abs_ex_filter]
                local_filter[:] = False
                local_filter[middle_idx] = True

        min_array, max_array = DataTransformer.get_local_ex(low, high)
        while level != 0:
            transform_middle_ex(min_array, low, "min")
            transform_middle_ex(max_array, high, "max")

            level -= 1
        
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


    def get_relative_strength(close: np.ndarray, market_index_close: np.ndarray, time_period: int = 1):
        cache_key = f"relative_strength_shape_{close.shape[0]}_{close.shape[1]}_time_period_{time_period}"
        
        if RedisClient.has(cache_key):
            return RedisClient.get_np_array(cache_key)

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

        value = np.concatenate([np.full((time_period, close.shape[1]), np.nan), relative_strength], axis=0)
        RedisClient.set_np_array(cache_key, value)

        return value
    
