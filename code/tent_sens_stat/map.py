@numba.jit(nopython=True)
def tent(x, n, s):
    if x < 1+s:
        return 2*x/(1+s)
    return 2/(1-s)*(2-x)


