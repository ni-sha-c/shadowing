from pylab import *
from numpy import *

x = pi/2 * ones(2)
s = array([0, 1E-5])
dx = [x[1] - x[0]]
for i in range(25):
    x = minimum(2 * x, 4 * (1+s) - 2 * x)
    dx.append(x[1] - x[0])

figure(figsize=(12,12))
semilogy(absolute(dx), 'o')
for tick in gca().xaxis.get_major_ticks():
    tick.label.set_fontsize(30)
for tick in gca().yaxis.get_major_ticks():
    tick.label.set_fontsize(30)
savefig('tent_ivp.png')

figure(figsize=(12,12))
for s in [0, 0.1, 0.2, 0.3, 0.4, 0.5]:
    plot([0, 1+s, 2+2*s], [0,2+2*s,0])
axis('scaled')
for tick in gca().xaxis.get_major_ticks():
    tick.label.set_fontsize(30)
for tick in gca().yaxis.get_major_ticks():
    tick.label.set_fontsize(30)
savefig('scaled_tent_map.png')
