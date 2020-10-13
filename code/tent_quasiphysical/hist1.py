import numba
from pylab import *
from numpy import *

width = 2**13

@numba.jit(nopython=True)
def transform(x, p, flip):
    if flip:
        addition = (random.rand() < p)
    else:
        addition = (random.rand() > p)
    x = (x * 2) + addition
    if x >= width:
        x = (2 * width - x - 1) % width
        return x, not flip
    else:
        return x, flip

@numba.jit(nopython=True)
def accumulate(n, p):
    xhist = zeros(width)
    x = random.randint(0,width)
    flip = False
    for i in range(n):
        xhist[x] += 1
        x, flip = transform(x, p, flip)
    return xhist

n = 100000000

figure(figsize=(16,18))
density = array(accumulate(n, 0.6), float) / n * width / 2
x = linspace(0,2,width*2+1)[1:-1:2]
fill_between(x, density)
ylim([0,1])
for tick in gca().xaxis.get_major_ticks():
    tick.label.set_fontsize(40)
for tick in gca().yaxis.get_major_ticks():
    tick.label.set_fontsize(40)
savefig('tent_quasiphysical_alternaive1_hist_p_0.6.png')
