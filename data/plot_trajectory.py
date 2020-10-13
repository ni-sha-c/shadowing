from pylab import *
from matplotlib.colors import LogNorm
from numpy import *

for i in [6,7,9]:
    a = array(load('lorenz_trajectory_{}.npy'.format(10**i)), float)
    a[a == 0] = 0.1
    figure(figsize=(24,24))
    imshow(a.T, extent=[-20,20,0,50],
           norm=LogNorm(vmin=0.2, vmax=a.max()), cmap='inferno', origin='lower')
    axis('scaled')
    cbar = colorbar()
    cbar.ax.tick_params(labelsize=50)
    for tick in gca().xaxis.get_major_ticks():
        tick.label.set_fontsize(50)
    for tick in gca().yaxis.get_major_ticks():
        tick.label.set_fontsize(50)
    savefig('lorenz_trajectory_{}.png'.format(10**i/1000))
