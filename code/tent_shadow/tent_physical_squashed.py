import numba
from pylab import *
from numpy import *

@numba.jit(nopython=True)
def maps(x, s):
    if x < pi:
        return x * (2-s)
    else:
        return (2*pi - x) * (2-s)

@numba.jit(nopython=True)
def physical(n, s, width):
    count = zeros(width, dtype=uint64)
    x = random.rand() * 2 * pi
    for i in range(100):
        x = maps(x, s)
    for i in range(n):
        x = maps(x, s)
        count[int(floor(x * width / (2*pi)))] += 1
    return count

n = 10000000000
for s in [0.01, 0.05, 0.2, 0.5]:
    width = 2**13
    density = physical(n, s, width) * width / 2 / n
    x = linspace(0,2,width*2+1)[1:-1:2]
    figure(figsize=(16,18))
    fill_between(x, density)
    ylim([0,1])
    for tick in gca().xaxis.get_major_ticks():
        tick.label.set_fontsize(40)
    for tick in gca().yaxis.get_major_ticks():
        tick.label.set_fontsize(40)
    savefig('tent_squashed_physical_density_{}.png'.format(s))
