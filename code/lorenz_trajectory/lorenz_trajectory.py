import os
import sys
import numba
from numpy import *

spec = [
    ('x0', numba.float64),
    ('y0', numba.float64),
    ('dx', numba.float64),
    ('dy', numba.float64),
    ('nx', numba.int32),
    ('ny', numba.int32),
    ('count', numba.uint64[:,:]),
]

@numba.experimental.jitclass(spec)
class Hist2d:
    def __init__(self, x0, y0, dx, dy, nx, ny):
        self.x0 = x0
        self.y0 = y0
        self.dx = dx
        self.dy = dy
        self.nx = nx
        self.ny = ny
        self.count = zeros((nx, ny), dtype=uint64)

    def accumulate(self, x, y):
        i, j = int((x - self.x0) / self.dx), int((y - self.y0) / self.dy)
        if i >= 0 and j >= 0 and i < self.nx and j < self.ny:
            self.count[i, j] += 1

@numba.jit(nopython=True)
def ddt(xyz):
    sigma = 10
    rho = 28
    beta = 8./3
    x, y, z = xyz
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return array([dxdt, dydt, dzdt])

@numba.jit(nopython=True)
def step(xyz):
    dt = 0.001
    xyz1 = xyz + 0.5 * dt * ddt(xyz)
    xyz += dt * ddt(xyz1)

@numba.jit(nopython=True)
def evolve(nsteps):
    nx = nz = 2048
    hist = Hist2d(-20, 0, 40./nx, 50./nz, nx, nz)
    xyz = array([0.01, 0.01, 28.])
    for i in range(nsteps):
        step(xyz)
        hist.accumulate(xyz[0], xyz[2])
    return hist.count

nsteps = int(sys.argv[1])
save('lorenz_trajectory_{}.npy'.format(nsteps), evolve(nsteps))
