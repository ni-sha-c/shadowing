import numba
from numpy import *
@numba.jit(nopython=True)
def tent_basic(x,s):
    if x < 1:
        return min(2*x/(1-s), \
                2 - 2*(1-x)/(1+s))
    return min(2*(2-x)/(1-s), \
            2 - 2*(x-1)/(1+s))

@numba.jit(nopython=True)
def oscillation(x, s, n):
    return tent_basic(2**n*x - floor(2**n*x),s)/2**n + \
            2*floor(2**n*x)/2**n

@numba.jit(nopython=True)
def osc_tent(x, s, n):
    return min(oscillation(x,s,n), oscillation(2-x,s,n))

@numba.jit(nopython=True)
def accumulate(nbins, n, m, s):
    density = zeros(nbins)
    dx = 2.0/nbins
    x = 2*rand()
    for i in range(n):
        x = osc_tent(x, s, m)
        bno = int(x//dx)
        density[bno] += 1/n/dx
    return density
'''
nbins = 2**9
x = linspace(0, 2, nbins)[1:-1]
y = zeros_like(x)
for i, xi in enumerate(x):
    y[i] = osc_tent(xi, 0.5, 5)
fig = figure(figsize=(8,5))
ax = fig.add_subplot(111)
ax.plot(x, y)
ax.xaxis.set_tick_params(labelsize=40)
ax.yaxis.set_tick_params(labelsize=40)
'''


nbins = 2**12
nrep = 1
density = zeros(nbins)
for m in range(nrep):
    print("repeat {}".format(m))
    density += accumulate(nbins, 1000000000, 6, 0.5)/nrep
x = linspace(0, 2, nbins+2)[1:-1]
fig = figure(figsize=(8,5))
ax = fig.add_subplot(111)
ax.fill_between(x, 0, density)
ax.xaxis.set_tick_params(labelsize=40)
ax.yaxis.set_tick_params(labelsize=40)

