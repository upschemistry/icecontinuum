from time import time
import numpy as np

"""
Performance testing functions for the ice model rolled into an importable module.
@author: Max B
"""


#Meta testing parameters
def multiple_test_avg_time(func, args, n_tests = 50, **kwargs):
    """
    Test a function n_tests times and return the average time taken for the function.
    """
    times = []
    for i in range(n_tests):
        start = np.float64(time())
        func(*args)
        times.append(time()-start)

    retval = float(np.mean(times))
    print("Time to run "+str(func.__name__)+" on average for "+ str(n_tests) +" tests: ", retval, "seconds")
    return retval
