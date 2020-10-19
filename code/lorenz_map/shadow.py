import os
import sys
import json
import numba
from pylab import *
from numpy import *
from scipy.optimize import curve_fit

@numba.jit(nopython=True)
def polyval(p, x):
    y, dy = 0, 0
    for pi in p:
        dy *= x
        dy += y
        y *= x
        y += pi
    return y, dy

@numba.jit(nopython=True)
def fit0(dz, pwr):
    return pow(dz * 1000, pwr), 1000 * pwr * pow(dz * 1000, pwr - 1)

@numba.jit(nopython=True)
def fit1(dz, pwr, p):
    f0, f0grad = fit0(dz, pwr)
    pz, pzgrad = polyval(p, dz)
    return f0 + pz, f0grad + pzgrad

@numba.jit(nopython=True)
def fit2(dz, pwr, p, a, b):
    print(dz)
    f1, f1grad = fit1(dz, pwr, p)
    num, numgrad = polyval(a, dz)
    den, dengrad = polyval(b, dz)
    return f1 + num / den, f1grad + numgrad / den - num * dengrad / den / den

@numba.jit(nopython=True)
def find2(dzNext, dzMin, dzMax, pwr, p, a, b):
    if dzNext <= dzMin or dzNext <= 0:
        return 0
    dz0 = 10.0
    for idz0 in range(4):
        dz0 /= 10
        dz = dz0
        #print()
        for i in range(10):
            if dz <= dzMax:
                f2, f2grad = fit2(dz, pwr, p, a, b)
            else:
                f2Max, f2grad = fit2(dzMax, pwr, p, a, b)
                f2 = f2Max + f2grad * (dz - dzMax)
            #print(dzNext, f2, f2grad, dz, exp((dzNext - f2) / f2grad / dz))
            if abs(dzNext - f2) < 1E-8 or abs(dzNext - f2) < 1E-2 and dz < 1E-8:
                return dz
            dz *= exp((dzNext - f2) / f2grad / dz)
    print(dzNext, dzMin, dzMax, pwr, p, a, b)
    return inf

@numba.jit(nopython=True)
def shadow(zTraj, zSep0, z0, zMin, zMax, zSep, pwr, pL, pR, aL, aR, bL, bR):
    zShadow = zeros_like(zTraj)
    z = z0
    dzMinL = fit2(0, pwr, pL, aL, bL)[0]
    dzMinR = fit2(0, pwr, pR, aR, bR)[0]
    dzMaxL = zMax - zSep
    dzMaxR = zSep - zMin
    for i in range(zTraj.size-1,-1,-1):
        if zTraj[i] > zSep0:
            z = zSep + find2(zMax - z, dzMinR, dzMaxR, pwr, pR, aR, bR)
        else:
            z = zSep - find2(zMax - z, dzMinL, dzMaxL, pwr, pL, aL, bL)
        zShadow[i] = z
        if not isfinite(z):
            return zShadow
    return zShadow

if __name__ == '__main__':
    params0 = json.load(open(sys.argv[1]))
    params1 = json.load(open(sys.argv[2]))
    
    zSep0 = params0['zSep']
    
    pwr = params1['pwr']
    zMin = params1['zMin']
    zMax = params1['zMax']
    zSep = params1['zSep']
    pL = array(params1['pL'])
    pR = array(params1['pR'])
    aL = array(params1['aL'])
    aR = array(params1['aR'])
    bL = array(params1['bL'])
    bR = array(params1['bR'])
    
    width = 2100
    x = linspace(29,50,width*2+1)[1:-1:2]
    density = zeros_like(x)
    
    zs0 = (zMax + zSep) / 2
    for i in range(99999, -1, -1):
        fname = 'data/shadow_baseline_{}_{}.npy'.format(sys.argv[1].rsplit('.', 1)[0], i)
        if os.path.exists(fname):
            print(i)
            z = load(fname)
            zs = shadow(z, zSep0, zs0, zMin, zMax, zSep, pwr, pL, pR, aL, aR, bL, bR)
            if not isfinite(zs).all():
                break
            zs0 = zs[0]
            density += histogram(zs, linspace(29,50,width+1))[0]
    
    save('lorenz_shadow_density_{}.npy'.format(sys.argv[2]).rsplit('.', 1)[0], density)
