import os
import sys
import json
from pylab import *
from numpy import *
from scipy.optimize import curve_fit

params0 = json.load(open(sys.argv[1]))
params1 = json.load(open(sys.argv[2]))

def fit0(dz, pwr):
    return pow(dz*1000, pwr)

def fit1(dz, pwr, p):
    return fit0(dz, pwr) + polyval(p, dz)

def fit2(dz, pwr, p, a, b):
    return fit1(dz, pwr, p) + polyval(a, dz) / polyval(b, dz)

dz = hstack([linspace(0, 0.1, 501), linspace(0.1, 1, 901)[1:], linspace(1, 10, 901)[1:]])
eta = linspace(0, 1, 1001)
z0, z1 = [], []
for i in range(eta.size):
    pwr = params0['pwr'] * (1 - eta[i]) + params1['pwr'] * eta[i]
    zMax = params0['zMax'] * (1 - eta[i]) + params1['zMax'] * eta[i]
    zSep = params0['zSep'] * (1 - eta[i]) + params1['zSep'] * eta[i]
    pL = array(params0['pL']) * (1 - eta[i]) + array(params1['pL']) * eta[i]
    pR = array(params0['pR']) * (1 - eta[i]) + array(params1['pR']) * eta[i]
    aL = array(params0['aL']) * (1 - eta[i]) + array(params1['aL']) * eta[i]
    aR = array(params0['aR']) * (1 - eta[i]) + array(params1['aR']) * eta[i]
    bL = array(params0['bL']) * (1 - eta[i]) + array(params1['bL']) * eta[i]
    bR = array(params0['bR']) * (1 - eta[i]) + array(params1['bR']) * eta[i]
    z0.append(hstack([zSep - dz[::-1], zSep + dz[1:]]))
    z1.append(hstack([zMax - fit2(dz[::-1], pwr, pL, aL, bL), zMax - fit2(dz[1:], pwr, pR, aR, bR)]))

z0, z1 = array([z0, z1])
figure(figsize=(24,24))
contourf(z0, eta[:,newaxis] * ones_like(z0), z1, 1000)
cbar = colorbar()
cbar.ax.tick_params(labelsize=50)
for tick in gca().xaxis.get_major_ticks():
    tick.label.set_fontsize(50)
for tick in gca().yaxis.get_major_ticks():
    tick.label.set_fontsize(50)
savefig('lorenz_zmap_{}_{}.png'.format(
    os.path.basename(sys.argv[1]).split('.')[0],
    os.path.basename(sys.argv[2]).split('.')[0]))
