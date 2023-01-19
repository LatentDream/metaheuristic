import time

def timer(func):
    """
    Python wrapper to print the time of execute of a func
    :param func: func to run
    :return: wrapper
    """
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        end = time.time()
        print(f"The time of execution: {end - start}s")
        return res

    return wrapper