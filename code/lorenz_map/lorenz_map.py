import os
import sys
import numba
from numpy import *

@numba.jit(nopython=True)
def ddt(xyz):
    x, y, z = xyz
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return array([dxdt, dydt, dzdt])

@numba.jit(nopython=True)
def step(xyz0_zint_tint):
    xyz0 = xyz0_zint_tint[:3]
    ddt0 = ddt(xyz0)
    dt = 0.001
    xyz1 = xyz0 + 0.5 * dt * ddt0
    ddt1 = ddt(xyz1)
    xyz1 = xyz0 + dt * ddt1
    zint = xyz0_zint_tint[3] + (xyz0[2] + xyz1[2]) / 2 * dt
    tint = xyz0_zint_tint[4] + dt
    ddt1 = ddt(xyz1)
    zmax = zeros(5)
    if ddt0[2] >= 0 and ddt1[2] < 0:
        if xyz0[2] > xyz1[2]:
            zmax[:3] = xyz0
            zmax[3] = xyz0_zint_tint[3]
            zmax[4] = xyz0_zint_tint[4]
            zint = (xyz0[2] + xyz1[2]) / 2 * dt
            tint = dt
        else:
            zmax[:3] = xyz1
            zmax[3] = zint
            zmax[4] = tint
            zint = 0
            tint = 0
    xyz0_zint_tint[3] = zint
    xyz0_zint_tint[4] = tint
    xyz0_zint_tint[:3] = xyz1
    return zmax

@numba.jit(nopython=True)
def evolve(xyz, nsteps):
    zmax = []
    for i in range(10000):
        step(xyz)
    for i in range(nsteps):
        z = step(xyz)
        if z[2] > 0:
            zmax.append(z)
    return xyz, zmax

sigma = 10
rho = 28
beta = 8./3
nbatches = int(sys.argv[1])
xyz = array([1, 1, 28.])
for i in range(nbatches):
    xyz, zmax = evolve(xyz, 1000000000)
    save('lorenz_zmax_{}_{}_{}_{:05d}.npy'.format(sigma, rho, beta, i), zmax)
