import pickle
import json
import time
import hashlib
import warnings
from collections import deque, OrderedDict


__VERSION__ = '0.10'


class MemCache():

    __MAX_LENGTH = 1024
    __MAX_BYTES = 8 * 1024 * 1024


    @classmethod
    def __get_container(cls):
        try:
            cls.__container
        except AttributeError:
            cls.__container = dict()
            cls.__queue = deque()
            cls.__bytes = 0
        return cls.__container, cls.__queue

    @classmethod
    def _set(cls, key, data):
        c, dq = cls.__get_container()

        pkl_str = pickle.dumps(data)
        if cls.__MAX_BYTES and len(pkl_str) > cls.__MAX_BYTES:
            warnings.warn('Object {} is too large to be cached in memory'.format(key), ResourceWarning)

        cache_obj = {
            '__timestamp': time.time(),
            'data': pkl_str,
        }
        c[key] = cache_obj
        dq.append(key)
        cls.__bytes += len(pkl_str)

        if cls.__MAX_LENGTH:
            while len(dq) > cls.__MAX_LENGTH:
                k_ = dq.popleft()
                cls.__bytes -= len(c[k_]['data'])
                del c[k_]
        if cls.__MAX_BYTES:
            while cls.__bytes > cls.__MAX_BYTES:
                k_ = dq.popleft()
                cls.__bytes -= len(c[k_]['data'])
                del c[k_]
    
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
    return ([0] * n) * n

@mem_cached(prefix='prefixed')
def test2(n):
    return n - 3

if __name__ == '__main__':
    print(test(9))
    print(test(8))
    print(len(test(9)))
    print('test2', test2(9))
    print(len(test(9)))
    for i in range(900, 920):
        print(i, len(test(i)))
    for i in range(5):
        print('908', len(test(918)))
    print('900', len(test(9000)))

        
