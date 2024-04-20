from typing import List


def split_payload(payload: List[any], split_size = 2):
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