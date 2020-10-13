import os
import sys
import json
from pylab import *
from numpy import *
from scipy.optimize import curve_fit

z = load(sys.argv[1])[:,2]
imax = z[1:].argmax()
def fit0(dz, p):
    return pow(dz*1000, p)

zSep, zMax = z[imax:imax+2]
zMin = z.min()
print(zSep, zMax, file=sys.stderr)

sel = (z[1:] > percentile(z, 90)).nonzero()[0]
zSep, pwr = curve_fit(
        lambda z, zSep, p : zMax - fit0(abs(z - zSep), p),
        z[sel], z[sel+1], (zSep, 1./3))[0]
print(zSep, zMax, pwr, file=sys.stderr)

iL = (z[:-1] <= zSep).nonzero()[0]
iR = (z[:-1] >= zSep).nonzero()[0]

figure(figsize=(12,12))
plot(z[iL], z[iL+1], '.', ms=1)
plot(z[iR], z[iR+1], '.', ms=1)

figure(figsize=(12,12))
loglog(zSep - z[iL], zMax - z[iL+1], '.', ms=1)
loglog(z[iR] - zSep, zMax - z[iR+1], '.', ms=1)

figure(figsize=(12,12))
plot(z[iL], zMax - z[iL+1] - fit0(zSep - z[iL], pwr), '.k', ms=1)
plot(z[iR], zMax - z[iR+1] - fit0(z[iR] - zSep, pwr), '.k', ms=1)
gca().set_xticks(linspace(30,45,4))
for tick in gca().xaxis.get_major_ticks():
    tick.label.set_fontsize(30)
for tick in gca().yaxis.get_major_ticks():
    tick.label.set_fontsize(30)

pL, pR = (0, 0.01091624, 0.44126003, -0.30532497), \
         (0, 0.01260884, 0.4280573, -0.28548578)
print(pL, pR, file=sys.stderr)

selL = iL[(z[iL+1] < percentile(z, 20)).nonzero()[0]]
selR = iR[(z[iR+1] < percentile(z, 20)).nonzero()[0]]
pL = curve_fit(
        lambda dz, *p : fit0(dz, pwr) + polyval(p, dz),
        zSep - z[selL], zMax - z[selL + 1], pL)[0]
pR = curve_fit(
        lambda dz, *p : fit0(dz, pwr) + polyval(p, dz),
        z[selR] - zSep, zMax - z[selR + 1], pR)[0]
print(pL, pR, file=sys.stderr)

def fit1L(dz):
    return fit0(dz, pwr) + polyval(pL, dz)

def fit1R(dz):
    return fit0(dz, pwr) + polyval(pR, dz)

figure(figsize=(12,12))
plot(z[iL], zMax - z[iL+1] - fit1L(zSep - z[iL]), '.k', ms=1)
plot(z[iR], zMax - z[iR+1] - fit1R(z[iR] - zSep), '.k', ms=1)
gca().set_xticks(linspace(30,45,4))
for tick in gca().xaxis.get_major_ticks():
    tick.label.set_fontsize(30)
for tick in gca().yaxis.get_major_ticks():
    tick.label.set_fontsize(30)

aL, aR = [-0.05288846,  0.22830716], [-0.05096505,  0.21626066]
bL, bR = [ 0.623915  , -0.33983043,  2.23848407,  1.        ], \
         [ 0.60037571, -0.31208347,  2.07960497,  1.        ]
print(aL, aR, file=sys.stderr)
print(bL, bR, file=sys.stderr)

abL = curve_fit(
        lambda dz, *ab : fit1L(dz) + polyval(ab[:2], dz) / polyval(ab[2:] + (1,), dz),
        zSep - z[iL], zMax - z[iL + 1], aL + bL[:-1])[0]
abR = curve_fit(
        lambda dz, *ab : fit1R(dz) + polyval(ab[:2], dz) / polyval(ab[2:] + (1,), dz),
        z[iR] - zSep, zMax - z[iR + 1], aR + bR[:-1])[0]
aL = abL[:2]
aR = abR[:2]
bL[:-1] = abL[2:]
bR[:-1] = abR[2:]
print(aL, aR, file=sys.stderr)
print(bL, bR, file=sys.stderr)

def fit2L(dz):
    return fit1L(dz) + polyval(aL, dz) / polyval(bL, dz)

def fit2R(dz):
    return fit1R(dz) + polyval(aR, dz) / polyval(bR, dz)

figure(figsize=(12,12))
plot(z[iL], zMax - z[iL+1] - fit2L(zSep - z[iL]), '.k', ms=1)
plot(z[iR], zMax - z[iR+1] - fit2R(z[iR] - zSep), '.k', ms=1)
gca().set_xticks(linspace(30,45,4))
ylim([-0.01, 0.01])
for tick in gca().xaxis.get_major_ticks():
    tick.label.set_fontsize(30)
for tick in gca().yaxis.get_major_ticks():
    tick.label.set_fontsize(30)
savefig('lorenz_map_fit_error_{}.png'.format(os.path.basename(sys.argv[1])))

figure(1)
dz = linspace(0, 10, 10001)
plot(zSep - dz, zMax - fit2L(dz), '--k')
plot(zSep + dz, zMax - fit2R(dz), '--k')
gca().set_xticks(linspace(30,45,4))
gca().set_yticks(linspace(30,45,4))
for tick in gca().xaxis.get_major_ticks():
    tick.label.set_fontsize(30)
for tick in gca().yaxis.get_major_ticks():
    tick.label.set_fontsize(30)
savefig('lorenz_map_fit_{}.png'.format(os.path.basename(sys.argv[1])))

json.dump({
        'zSep' : zSep,
        'zMin' : zMin,
        'zMax' : zMax,
        'pwr'  : pwr,
        'pL'   : list(pL),
        'pR'   : list(pR),
        'aL'   : list(aL),
        'aR'   : list(aR),
        'bL'   : list(bL),
        'bR'   : list(bR)
        }, sys.stdout, indent=4, separators=(',', ': '))
