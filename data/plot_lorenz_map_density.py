from pylab import *
from numpy import *
import glob
import sys

for fname in sys.argv[1:]:
    density = load(fname)
    print(density.shape)
    width = 2100
    assert density.size == width
    x = linspace(29,50,width*2+1)[1:-1:2]
    density /= density.sum() * (x[1] - x[0])
    
    figure(figsize=(16,18))
    fill_between(x, density)
    xlim([29, 50])
    ylim([0, 0.18])
    xticks(fontsize=40)
    yticks(fontsize=40)
    savefig('{}.png'.format(fname.rsplit('.', maxsplit=1)[0]))
