import json
import numpy as np
import os

from typing import List
from pathlib import Path


class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def split_payload(payload: List[any], split_size=2):
    n = 0
    split_payload = []

    for i in range(split_size):
        remain_cores = split_size - i
        remain_payload = payload[n:]
        split_size = len(remain_payload) // remain_cores
        split_size = split_size + int(len(remain_payload) % remain_cores != 0)
        split_payload.append(remain_payload[:split_size])
        n += split_size

    return split_payload


def replace_null_with_empty(string):
    if type(string) == str:
        string = [string]

    return list(map(lambda x: x.replace("\x00", "").replace(",-,", ",,"), string))


def combine_config(source, target):
    source = source.copy()
    target = target.copy()
    for key, value in target.items():
        if type(value) == dict:
            source[key] = combine_config(source[key], value)
        else:
            source[key] = value
    return source
