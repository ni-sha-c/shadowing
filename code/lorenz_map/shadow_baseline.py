import os
import sys
import json
import numba
from pylab import *
from numpy import *
from scipy.optimize import curve_fit

params0 = json.load(open(sys.argv[1]))

@numba.jit(nopython=True)
def polyval(p, x):
    y = 0
    for pi in p:
        y *= x
        y += pi
    return y

@numba.jit(nopython=True)
def fit0(dz, pwr):
    return pow(dz * 1000, pwr)

@numba.jit(nopython=True)
def fit1(dz, pwr, p):
    f0 = fit0(dz, pwr)
    pz = polyval(p, dz)
    return f0 + pz

@numba.jit(nopython=True)
def fit2(dz, pwr, p, a, b):
    f1 = fit1(dz, pwr, p)
    num = polyval(a, dz)
    den = polyval(b, dz)
    return f1 + num / den

@numba.jit(nopython=True)
def trajectory(z, n, zMax, zSep, pwr, pL, pR, aL, aR, bL, bR):
    for i in range(100):
        if z > zSep:
            z = zMax - fit2(z - zSep, pwr, pR, aR, bR)
        else:
            z = zMax - fit2(zSep - z, pwr, pL, aL, bL)
    
    zT = zeros(n)
    for i in range(n):
        if z > zSep:
            z = zMax - fit2(z - zSep, pwr, pR, aR, bR)
        else:
            z = zMax - fit2(zSep - z, pwr, pL, aL, bL)
        zT[i] = z
    return zT

pwr = params0['pwr']
zMin = params0['zMin']
zMax = params0['zMax']
zSep = params0['zSep']
pL = array(params0['pL'])
pR = array(params0['pR'])
aL = array(params0['aL'])
aR = array(params0['aR'])
bL = array(params0['bL'])
bR = array(params0['bR'])

width = 2100
x = linspace(29,50,width*2+1)[1:-1:2]
density = zeros_like(x)

n = 10000000
z = (zSep + zMax) / 2
for i in range(200):
    z = trajectory(z, n, zMax, zSep, pwr, pL, pR, aL, aR, bL, bR)
    # save('shadow_baseline_{}_{}.npy'.format(sys.argv[1].rsplit('.', 1)[0], i), z)
    density += histogram(z, linspace(29,50,width+1))[0]
    z = z[-1]

save('baseline_density_{}.npy'.format(sys.argv[1]), density)
