import numba
from pylab import *
from numpy import *

@numba.jit(nopython=True)
def find(x, y, s):
    if x < 1:
        return (y + y * (2-y) * s / 2) / 2
    else:
        return 2 - (y + y * (2-y) * s / 2) / 2

@numba.jit(nopython=True)
def shadow(n, s, width):
    count = zeros(width, dtype=uint64)
    x = random.rand() * 2
    y = x
    for i in range(100):
        y = find(x, y, s)
        x = x / 2 + random.randint(0,2)
    for i in range(n):
        y = find(x, y, s)
        count[int(floor(y * width / 2))] += 1
        x = x / 2 + random.randint(0,2)
    return count
'''
n = 10000000000
for s in [0.01, 0.05, 0.2, 0.9]:
    width = 2**13
    density = shadow(n, s, width) * width / 2 / n
    x = linspace(0,2,width*2+1)[1:-1:2]

    figure(1, figsize=(12,12))
    y = linspace(0,2,1025)
    plot(hstack([(y + y * (2-y) * s / 2) / 2,
                 2 - (y + y * (2-y) * s / 2)[-1::-1] / 2]),
         hstack([y, y[-1::-1]]))

    figure(figsize=(16,18))
    fill_between(x, density)
    ylim([0,1])
    for tick in gca().xaxis.get_major_ticks():
        tick.label.set_fontsize(40)
    for tick in gca().yaxis.get_major_ticks():
        tick.label.set_fontsize(40)
    savefig('tent_pinched_shadow_density_{}.png'.format(s))

figure(1)
axis('scaled')
for tick in gca().xaxis.get_major_ticks():
    tick.label.set_fontsize(30)
for tick in gca().yaxis.get_major_ticks():
    tick.label.set_fontsize(30)
savefig('pinched_tent_map.png')
'''
