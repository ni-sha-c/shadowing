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
    xyz0 += dt * ddt1

@numba.jit(nopython=True)
def evolve(xyz0, steps):
    history = []
    xyz = xyz0
    for i in range(steps):
        history.append(xyz.copy())
        step(xyz)
    history.append(xyz)
    return history

sigma = 10
rho = 28
beta = 8./3

xyz0, steps = array([-13.92609234,  -7.54780951,  39.82352798]), 1558
xyz = array(evolve(xyz0, steps))
save('lorenz_periodic_1.npy', xyz, '-')
plot(xyz[:,0], xyz[:,2])

xyz0, steps = array([12.99988069,   7.76732599,  37.74858325]), 3083
xyz = array(evolve(xyz0, steps))
save('lorenz_periodic_2.npy', xyz)
plot(xyz[:,0], xyz[:,2], '--')

xyz0, steps = array([-13.14298792,  -7.7604085 ,  38.04801711]), 4566
xyz = array(evolve(xyz0, steps))
save('lorenz_periodic_3.npy', xyz)
plot(xyz[:,0], xyz[:,2], ':')
