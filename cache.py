import pickle
import json
import time
import hashlib
import uuid
from collections import deque, OrderedDict


__VERSION__ = '0.10'


class MemCache():

    MAX_LENGTH = 16

    @classmethod
    def __get_container(cls):
        cls._EMPTY = uuid.uuid1()
        try:
            cls.__container
        except AttributeError:
            cls.__container = dict()
            cls.__queue = deque()
        return cls.__container, cls.__queue

    @classmethod
    def _set(cls, key, data):
        c, dq = cls.__get_container()
        dq.append(key)
        while len(dq) > cls.MAX_LENGTH:
            k_ = dq.popleft()
            del c[k_]

        cache_obj = {
            '__timestamp': time.time(),
            'data': pickle.dumps(data),
        }
        c[key] = cache_obj
    
    @classmethod
    def _get(cls, key):
        c, dq = cls.__get_container()
        try:
            cache_obj = c[key]
            return True, pickle.loads(cache_obj['data'])
        except KeyError:
            pass
        return False, None

    @classmethod
    def _md5(cls, obj):
        j = json.dumps(obj).encode()
        h = hashlib.new('md5', j)
        return h.hexdigest()
    

def mem_cached(*args, **kwargs):

    def decorator(func):
        prefix = 'generic'
        try:
            prefix = kwargs['prefix']
        except KeyError:
            pass

        def wrapper(*args, **kwargs):
            k = '{}://{}'.format(prefix, MemCache._md5([args, OrderedDict(kwargs)]))
            cached, d = MemCache._get(k)
            if cached:
                return d
            d = func(*args, **kwargs)
            MemCache._set(k, d)
            return d

        return wrapper

    return decorator


@mem_cached()
def test(n):
    return n * n

@mem_cached(prefix='prefixed')
def test2(n):
    return n - 3

if __name__ == '__main__':
    print(test(9))
    print(test(8))
    print(test(9))
    print(test2(9))

        
