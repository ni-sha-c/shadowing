from pylab import *
from numpy import *
import sys

density = load(sys.argv[1])
print(density.shape)
width = 2100
assert density.size == width
x = linspace(29,50,width*2+1)[1:-1:2]

figure(figsize=(16,18))
fill_between(x, density)
xticks(fontsize=40)
yticks(fontsize=40)
savefig('{}.png'.format(sys.argv[1].rsplit('.', maxsplit=1)[0]))
