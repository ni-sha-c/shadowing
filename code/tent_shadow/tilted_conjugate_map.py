from pylab import *
from numpy import *


for s in [0, 0.1, 0.2, 0.3, 0.4, 0.5]:
    xAll, yAll = array([[0, 2], [0, 2]])
    for i in range(22):
        # xNew = xAll[1::2] / 2
        yNew = (yAll[1:] + yAll[:-1]) / 2
        # yNew = yAll[1::2] / 2 + minimum(yAll[1::2], 1 - yAll[1::2]) * s
        weightLeft, weightRight = zeros([2, yAll.size - 1])
        weightLeft[::2] = (1-s)
        weightRight[::2] = (1+s)
        weightLeft[1::2] = (1+s)
        weightRight[1::2] = (1-s)
        xNew = (xAll[:-1] * weightLeft + xAll[1:] * weightRight) / 2
        xAll = hstack([xAll, xNew])
        yAll = hstack([yAll, yNew])
        iSort = xAll.argsort()
        xAll = xAll[iSort]
        yAll = yAll[iSort]
    
    figure(1, figsize=(12,12))
    plot(xAll, yAll)

    figure(2, figsize=(12,12))
    plot([0, 1+s, 2], [0,2,0])

    nSample = 2**20
    x = linspace(0,2,nSample*2+1)[1:-1:2]
    i = searchsorted(xAll, x)
    assert i.min() > 0
    r = (xAll[i] - x) / (xAll[i] - xAll[i-1])
    y = yAll[i-1] * r + yAll[i] * (1 - r)

    figure(figsize=(16,18))
    width = 2**13
    density = histogram(y, linspace(0,2,width+1))[0] / nSample * width / 2
    fill_between(linspace(0,2,width*2+1)[1:-1:2], density)
    ylim([0,1])
    for tick in gca().xaxis.get_major_ticks():
        tick.label.set_fontsize(40)
    for tick in gca().yaxis.get_major_ticks():
        tick.label.set_fontsize(40)
    savefig('tilted_conjugate_density_{}.png'.format(s))

figure(1)
axis('scaled')
for tick in gca().xaxis.get_major_ticks():
    tick.label.set_fontsize(30)
for tick in gca().yaxis.get_major_ticks():
    tick.label.set_fontsize(30)
savefig('tilted_conjugate_map.png')

figure(2)
axis('scaled')
for tick in gca().xaxis.get_major_ticks():
    tick.label.set_fontsize(30)
for tick in gca().yaxis.get_major_ticks():
    tick.label.set_fontsize(30)
savefig('tilted_tent_map.png')
