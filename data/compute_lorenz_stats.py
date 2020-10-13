from numpy import *

for i in range(3):
    t = load('lorenz_periodic_{}.npy'.format(i+1))
    print(t[:,2].mean(), exp(-t[:,2]**2/2).mean())

a = load('lorenz_density_10.0_28.0_2.6666666666666665_0.01_5000.npy')

za = linspace(0, 50, 2048*2+1)[1:-1:2]
xa = linspace(-20, 20, 2048*2+1)[1:-1:2]
z_mean, gaussian_mean = (a * za).sum() / a.sum(), (a * exp(-za**2/2)).sum() / a.sum()
z2_mean, gaussian2_mean = (a * za**2).sum() / a.sum(), (a * exp(-za**2/2)**2).sum() / a.sum()
z_std, gaussian_std = sqrt(z2_mean - z_mean**2), sqrt(gaussian2_mean - gaussian_mean**2) 
print(z_mean, gaussian_mean)
print(z_std / sqrt(a.sum()), gaussian_std / sqrt(a.sum()))
