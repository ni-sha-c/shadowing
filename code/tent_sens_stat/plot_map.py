import numba
from numpy import *
from plucked_map import *
nbins = 2**13
ns = 5
nn = 5
n_arr = [0, 2, 4, 6]
s_arr = [0.2, 0.5]

x = linspace(0, 2, nbins+2)[1:-1]
ys = zeros((ns,nbins))
yn = zeros((nn,nbins))
for kn,nk in enumerate(n_arr):
    for k,sk in enumerate(s_arr):
        for i, xi in enumerate(x):
            yn[k,i] = osc_tent(xi, sk, nk)
    fig = figure(figsize=(8,6))
    ax = fig.add_subplot(111)
    for k,sk in enumerate(s_arr):
        ax.plot(x, yn[k],lw=2.5,label="s = {}".format(sk))
    ax.xaxis.set_tick_params(labelsize=40)
    ax.yaxis.set_tick_params(labelsize=40)
    leg = ax.legend(fontsize=30)
    ax.axis("scaled")
