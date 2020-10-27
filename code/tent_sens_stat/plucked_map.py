from numpy import *
import numba
@numba.jit(nopython=True)
def tent_basic(x,s):
    if x < 1:
        return min(2*x/(1-s), \
                2 - 2*(1-x)/(1+s))
    return min(2*(2-x)/(1-s), \
            2 - 2*(x-1)/(1+s))

@numba.jit(nopython=True)
def oscillation(x,s):
    if x < 0.5:
        return tent_basic(2*x,s)/2
    return 2-tent_basic(2-2*x,s)/2


@numba.jit(nopython=True)
def frequency(x,s,n):
    return oscillation(2**n*x - floor(2**n*x),s)/2**n + \
            2*floor(2**n*x)/2**n

@numba.jit(nopython=True)
def osc_tent(x, s, n):
    return min(frequency(x,s,n), frequency(2-x,s,n))

