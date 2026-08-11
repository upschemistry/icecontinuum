import numpy as np
from numba import njit, float64, int32, types

@njit
def test1():
    l = 5
    dy = np.zeros((5,))
    for i in range(1,l):
        print(i)
        dy[i] = i
    return dy

@njit
def test2(x):
    print(x[0])