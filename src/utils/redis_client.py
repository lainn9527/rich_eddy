from datetime import datetime
from typing import List
import redis
import json
import struct
import numpy as np

class RedisClient:
    client = redis.Redis(host='localhost', port=6379)

    def set_np_array(key, value):
        print(f"set np array cache from key: {key}")
        h, w = value.shape
        shape = struct.pack('>II',h,w)
        encoded = shape + value.tobytes()

        return RedisClient.client.set(key, encoded)

    def get_np_array(key):
        print(f"get np array cache from key: {key}")
        encoded = RedisClient.client.get(key, )
        h, w = struct.unpack('>II',encoded[:8])
        return np.frombuffer(encoded[8:]).reshape(h,w)


    def get_datetime_array(key):
        value = json.loads(RedisClient.client.get(key))
        return [datetime.fromisoformat(date) for date in value]
    

    def set_datetime_array(key, value: List[datetime]):
        value = json.dumps([date.isoformat() for date in value])
        return RedisClient.client.set(key, value)
    

    def set_json(key, value):
        return RedisClient.client.set(key, json.dumps(value, ensure_ascii=False))


    def get_json(key):
        value = RedisClient.client.get(key)
        return json.loads(value)


    def get(key):
        return RedisClient.client.get(key)


    def set(key, value):
        return RedisClient.client.set(key, value)


    def has(key):
        return RedisClient.client.exists(key)