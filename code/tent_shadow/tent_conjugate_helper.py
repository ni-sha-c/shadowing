import numba
from pylab import *
from numpy import *

@numba.jit(nopython=True)
def next(x, s):
    if x < 1+s:
        return 2*x/(1+s)
    else:
        return 4/(1-s) - 2*x/(1-s)

@numba.jit(nopython=True)
def xi(x, s, j):
    if j==0 and x<1+s:
        return 0
    elif j==0 and x>=(1+s):
        return 1
    xj = x
    for k in range(j):
        xj = next(xj,s)
    if xj < 1+s:
        return xi(x, s, j-1)
    return 1 - xi(x,s,j-1)


n = 500
x = linspace(0.,2, n)
for s in [0, 0.05, 0.4, 0.7]:
    fig = figure(figsize=(16,18))
    ax = fig.add_subplot(111)
    xij = zeros_like(x)
    for j in range(5,6):
        for k,xk in enumerate(x):
            xij[k] = xi(xk, s, j)
        ax.plot(x, xij, label=r'$\xi_{{{},{}}}$'.format(s,j), lw=2.5)
        ax.xaxis.set_tick_params(labelsize=40)
        ax.yaxis.set_tick_params(labelsize=40)
        leg = ax.legend(fontsize=40)

    savefig('tent_tilted_conjugacy_helper_{}.png'.format(s))
