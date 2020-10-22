import numba
from numpy import *
@numba.jit(nopython=True)
def tent(x, n, s):
    if x < 1+s:
        return 2*x/(1+s)
    return 2/(1-s)*(2-x)

@numba.jit(nopython=True)
def accumulate(density, n, m, s):
    nbins = len(density)
    dx = 2.0/nbins
    x = 2*rand()
    for i in range(n):
        x = tent(x, m, s)
        bno = int(x//dx)
        density[bno] += 1/n
    return density

nbins = 2**8
density = zeros(nbins)
'''
x = linspace(0, 2, nbins)[1:-1]
y = zeros_like(x)
for i, xi in enumerate(x):
    y[i] = tent(xi, 1, 0)

fig = figure(figsize=(8,5))
ax = fig.add_subplot(111)
ax.plot(x,y)
ax.xaxis.set_tick_params(labelsize=40)
ax.yaxis.set_tick_params(labelsize=40)
'''


