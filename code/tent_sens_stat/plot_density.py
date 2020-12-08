import numba
from numpy import *
from plucked_map import *
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
    if x < 0.5:
        return (1-s)**2/2
    elif x < 1:
        return (1-s*s)/2
    elif x < 1.5:
        return (1+s)**2/2
    return (1-s*s)/2

nbins = 2**12
nrep = 5
density = zeros(nbins)
s = 0.1
n = 3
for m in range(nrep):
    print("repeat {}".format(m))
    density += accumulate(nbins, 1000000000, n, s)/nrep
x = array(linspace(0, 2, nbins+2)[1:-1])
#for i,xi in enumerate(x):
#    density[i] = analytical_rho(xi,s)
fig = figure(figsize=(8,5))
ax = fig.add_subplot(111)
ax.fill_between(x, 0, density)
ax.xaxis.set_tick_params(labelsize=40)
ax.yaxis.set_tick_params(labelsize=40)

