from functools import wraps
from time import time


# Decorate a function to measure its runtime, taken from https://stackoverflow.com/a/27737385
def timing(f):  # type: ignore[no-untyped-def]
    @wraps(f)
    def wrap(*args, **kw):  # type: ignore[no-untyped-def]
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r took: %2.4f sec\n' % (f.__name__, te-ts))
        return result
    return wrap
