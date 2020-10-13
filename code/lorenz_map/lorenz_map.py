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
def step(xyz0):
    ddt0 = ddt(xyz0)
    dt = 0.001
    xyz1 = xyz0 + 0.5 * dt * ddt0
    ddt1 = ddt(xyz1)
    xyz1 = xyz0 + dt * ddt1
    ddt1 = ddt(xyz1)
    if ddt0[2] >= 0 and ddt1[2] < 0:
        if xyz0[2] > xyz1[2]:
            zmax = xyz0.copy()
        else:
            zmax = xyz1.copy()
    else:
        zmax = zeros(3)
    xyz0[:] = xyz1
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
