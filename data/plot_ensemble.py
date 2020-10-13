from pylab import *
from matplotlib.colors import LogNorm
from numpy import *

for i in [10,15,50]:
    a = load('lorenz_density_10.0_28.0_2.6666666666666665_0.01_{}.npy'.format(i*100))
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
    savefig('lorenz_ensemble_{}.png'.format(i))
