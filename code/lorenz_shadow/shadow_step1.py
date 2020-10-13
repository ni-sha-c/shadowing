from pylab import *
from numpy import *

zmax0 = load('../../data/lorenz_zmax_10_28_2.6666666666666665.npy')
#zmax1 = load('../../data/lorenz_zmax_10_30_2.6666666666666665.npy')
#zmax1 = load('../../data/lorenz_zmax_12_28_2.6666666666666665.npy')
#zmax1 = load('../../data/lorenz_zmax_10_28_3.3333333333333335.npy')

zgrid = linspace(28, 53, 2401)

#density0 = histogram(zmax0[:,2], zgrid[::2])[0] / zmax0.shape[0] * (zgrid.size //2) / (zgrid[-1] - zgrid[0])
#fill_between(zgrid[1::2], density0)
#
#figure()
#density1 = histogram(zmax1[:,2], zgrid[::2])[0] / zmax0.shape[0] * (zgrid.size //2) / (zgrid[-1] - zgrid[0])
#fill_between(zgrid[1::2], density1)

def shadow_density(zmax0, zmax1, fname):
    isCross0 = (zmax0[:-1,0] > 0) ^ (zmax0[1:,0] > 0)
    isCross1 = (zmax1[:-1,0] > 0) ^ (zmax1[1:,0] > 0)
    
    iL1 = (~isCross1).nonzero()[0][zmax1[1:,2][~isCross1].argsort()]
    iR1 = (isCross1).nonzero()[0][zmax1[1:,2][isCross1].argsort()]
    zL1 = zmax1[iL1+1,2]
    zR1 = zmax1[iR1+1,2]
    zL1[-1] = inf
    zR1[-1] = inf
    
    z = zmax0[-1,2]
    i1seq = zeros(zmax0.shape[0]-1, int)
    for i0 in range(1, zmax0.shape[0]):
        if isCross0[-i0]:
            i1 = iR1[searchsorted(zR1, z)]
        else:
            i1 = iL1[searchsorted(zL1, z)]
        z = zmax1[i1, 2]
        i1seq[-i0] = i1

    save(fname, zmax1[i1seq])
    
    figure()
    zgrid = linspace(32, 50, 2401)
    density1 = histogram(zmax1[i1seq,2], zgrid[::2])[0] / zmax0.shape[0] * (zgrid.size //2) / (zgrid[-1] - zgrid[0])
    fill_between(zgrid[1::2], density1)
    axis([zgrid[0], zgrid[-1], 0, 0.18])
    
zmax1 = load('../../data/lorenz_zmax_15_28_2.6666666666666665.npy')
shadow_density(zmax0, zmax1, 'shadow_start_15_28_2.6666666666666665.npy')
title('sigma')
zmax1 = load('../../data/lorenz_zmax_10_30_2.6666666666666665.npy')
shadow_density(zmax0, zmax1, 'shadow_start_10_30_2.6666666666666665.npy')
title('rho')
zmax1 = load('../../data/lorenz_zmax_10_28_3.3333333333333335.npy')
shadow_density(zmax0, zmax1, 'shadow_start_10_28_3.3333333333333335.npy')
title('beta')
