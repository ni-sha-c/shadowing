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

nbins = 2**13
ns = 5
nn = 5
n_arr = [0, 1, 2, 4, 6]
s_arr = [0, 0.1, 0.2, 0.3, 0.5]

x = linspace(0, 2, nbins+2)[1:-1]
ys = zeros((ns,nbins))
yn = zeros((nn,nbins))
for k,nk in enumerate(n_arr):
    for i, xi in enumerate(x):
        yn[k,i] = osc_tent(xi, 0.5, nk)
fig = figure(figsize=(8,6))
ax = fig.add_subplot(111)
for k,nk in enumerate(n_arr):
    ax.plot(x, yn[k],lw=2.5,label="n = {}".format(nk))
ax.xaxis.set_tick_params(labelsize=40)
ax.yaxis.set_tick_params(labelsize=40)
leg = ax.legend(fontsize=30)
ax.axis("scaled")

for k,sk in enumerate(s_arr):
    for i, xi in enumerate(x):
        ys[k,i] = osc_tent(xi, sk, 0)
fig = figure(figsize=(8,6))
ax = fig.add_subplot(111)
for k,sk in enumerate(s_arr):
    ax.plot(x, ys[k],lw=2.5,label="s = {}".format(sk))
ax.xaxis.set_tick_params(labelsize=40)
ax.yaxis.set_tick_params(labelsize=40)
ax.axis("scaled")
leg = ax.legend(fontsize=30)
