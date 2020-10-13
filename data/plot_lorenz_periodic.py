from pylab import *
from matplotlib.colors import LogNorm
from numpy import *

a = load('lorenz_density_10.0_28.0_2.6666666666666665_0.01_5000.npy')
a[a == 0] = 0.1
figure(figsize=(24,24))
imshow(a.T, extent=[-20,20,0,50],
       norm=LogNorm(vmin=0.2, vmax=a.max()), cmap='inferno', origin='lower')

styles = ['-', '--', ':']
for i in range(3):
    t = load('lorenz_periodic_{}.npy'.format(i+1))
    plot(t[:,0], t[:,2], styles[i] + 'b', lw=4)
axis('scaled')
for tick in gca().xaxis.get_major_ticks():
    tick.label.set_fontsize(50)
for tick in gca().yaxis.get_major_ticks():
    tick.label.set_fontsize(50)
savefig('lorenz_periodic.png')
