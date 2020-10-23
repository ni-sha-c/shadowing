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

@numba.jit(nopython=True)
def analytical_rho(x,s):
    if x < 1:
        return (1-s)/2
    return (1+s)/2

nbins = 2**12
nrep = 1
density = zeros(nbins)
s = 0.5
n = 6
for m in range(nrep):
    print("repeat {}".format(m))
    density += accumulate(nbins, 1000000, n, s)/nrep
x = array(linspace(0, 2, nbins+2)[1:-1])
#for i,xi in enumerate(x):
#    density[i] = analytical_rho(2*2**n*xi - 2*floor(2**n*xi),s)
fig = figure(figsize=(8,5))
ax = fig.add_subplot(111)
ax.fill_between(x, 0, density)
ax.xaxis.set_tick_params(labelsize=40)
ax.yaxis.set_tick_params(labelsize=40)

