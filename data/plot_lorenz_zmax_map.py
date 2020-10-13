from pylab import *
from numpy import *

z = load('lorenz_zmax_10_28_2.6666666666666665.npy')[:,2]
'''
figure(figsize=(16,16))
plot(z[:-1], z[1:], '.', ms=1)
axis('scaled')
axis([28,49,28,49])
xticks(linspace(29, 49, 5))
yticks(linspace(29, 49, 5))
for tick in gca().xaxis.get_major_ticks():
    tick.label.set_fontsize(40)
for tick in gca().yaxis.get_major_ticks():
    tick.label.set_fontsize(40)
plot([28,49], [28,49], '--k')
savefig('lorenz_zmax_10_28_2.6666666666666665.png')

figure(figsize=(16,16))
plot(z[:-2], z[2:], '.', ms=1)
axis('scaled')
axis([28,49,28,49])
xticks(linspace(29, 49, 5))
yticks(linspace(29, 49, 5))
for tick in gca().xaxis.get_major_ticks():
    tick.label.set_fontsize(40)
for tick in gca().yaxis.get_major_ticks():
    tick.label.set_fontsize(40)
plot([28,49], [28,49], '--k')
savefig('lorenz_zmax2_10_28_2.6666666666666665.png')

figure(figsize=(16,16))
plot(z[:-3], z[3:], '.', ms=1)
axis('scaled')
axis([28,49,28,49])
xticks(linspace(29, 49, 5))
yticks(linspace(29, 49, 5))
for tick in gca().xaxis.get_major_ticks():
    tick.label.set_fontsize(40)
for tick in gca().yaxis.get_major_ticks():
    tick.label.set_fontsize(40)
plot([28,49], [28,49], '--k')
savefig('lorenz_zmax3_10_28_2.6666666666666665.png')

figure(figsize=(16,16))
plot([0, 1, 2], [0, 2, 0])
axis('scaled')
axis([0,2,0,2])
xticks(linspace(0, 2, 3))
yticks(linspace(0, 2, 3))
for tick in gca().xaxis.get_major_ticks():
    tick.label.set_fontsize(40)
for tick in gca().yaxis.get_major_ticks():
    tick.label.set_fontsize(40)
savefig('tent_map.png')

figure(figsize=(16,16))
sToLine = {0 : '-', 0.2 : '--', 0.4 : ':'}
for s in sToLine:
    plot([0, 0.5+s, 1, 1.5-s, 2], [0, 1, 2, 1, 0], sToLine[s])
axis('scaled')
axis([0,2,0,2])
xticks(linspace(0, 2, 3))
yticks(linspace(0, 2, 3))
for tick in gca().xaxis.get_major_ticks():
    tick.label.set_fontsize(40)
for tick in gca().yaxis.get_major_ticks():
    tick.label.set_fontsize(40)
savefig('tent_maps.png')

close('all')
'''
z0 = z

figure(figsize=(10,10))
plot(z0[:-1], z0[1:], '.', ms=1)
#z = load('lorenz_zmax_12_28_2.6666666666666665.npy')[:,2]
#plot(z[:-1], z[1:], '.', ms=1)
z = load('lorenz_zmax_15_28_2.6666666666666665.npy')[:,2]
plot(z[:-1], z[1:], '.', ms=1)
axis('scaled')
axis([28,52,28,52])
xticks(linspace(30, 50, 5))
yticks(linspace(30, 50, 5))
for tick in gca().xaxis.get_major_ticks():
    tick.label.set_fontsize(40)
for tick in gca().yaxis.get_major_ticks():
    tick.label.set_fontsize(40)
savefig('lorenz_zmax_sigma.png')

figure(figsize=(10,10))
zmid = z0[z0.argmax()-1]
zmin = z0[ma.array(z0[2:], mask=z0[:-2] > z0[1:-1]).argmax()]
plot((z0[:-1] - zmin) / (zmid - zmin),
     (z0[1:] - zmin) / (zmid - zmin), '.', ms=1)
#z = load('lorenz_zmax_12_28_2.6666666666666665.npy')[:,2]
#zmid = z[z.argmax()-1]
#zmin = percentile(z[z<zmid], 10)
#plot((z[:-1] - zmin) / (zmid - zmin),
#     (z[1:] - zmin) / (zmid - zmin), '.', ms=1)
z = load('lorenz_zmax_15_28_2.6666666666666665.npy')[:,2]
zmid = z[z.argmax()-1]
zmin = z[ma.array(z[2:], mask=z[:-2] > z[1:-1]).argmax()]
plot((z[:-1] - zmin) / (zmid - zmin),
     (z[1:] - zmin) / (zmid - zmin), '.', ms=1)
axis('scaled')
axis([-5,8,-5,8])
xticks(linspace(-4, 8, 7))
yticks(linspace(-4, 8, 7))
for tick in gca().xaxis.get_major_ticks():
    tick.label.set_fontsize(40)
for tick in gca().yaxis.get_major_ticks():
    tick.label.set_fontsize(40)
savefig('lorenz_zmax_sigma_scaled.png')

figure(figsize=(10,10))
plot(z0[:-1], z0[1:], '.', ms=1)
#z = load('lorenz_zmax_10_29_2.6666666666666665.npy')[:,2]
#plot(z[:-1], z[1:], '.', ms=1)
z = load('lorenz_zmax_10_30_2.6666666666666665.npy')[:,2]
plot(z[:-1], z[1:], '.', ms=1)
axis('scaled')
axis([28,52,28,52])
xticks(linspace(30, 50, 5))
yticks(linspace(30, 50, 5))
for tick in gca().xaxis.get_major_ticks():
    tick.label.set_fontsize(40)
for tick in gca().yaxis.get_major_ticks():
    tick.label.set_fontsize(40)
savefig('lorenz_zmax_rho.png')

figure(figsize=(10,10))
zmid = z0[z0.argmax()-1]
zmin = z0[ma.array(z0[2:], mask=z0[:-2] > z0[1:-1]).argmax()]
plot((z0[:-1] - zmin) / (zmid - zmin),
     (z0[1:] - zmin) / (zmid - zmin), '.', ms=1)
#z = load('lorenz_zmax_10_29_2.6666666666666665.npy')[:,2]
#zmid = z[z.argmax()-1]
#zmin = percentile(z[z<zmid], 10)
#plot((z[:-1] - zmin) / (zmid - zmin),
#     (z[1:] - zmin) / (zmid - zmin), '.', ms=1)
z = load('lorenz_zmax_10_30_2.6666666666666665.npy')[:,2]
zmid = z[z.argmax()-1]
zmin = z[ma.array(z[2:], mask=z[:-2] > z[1:-1]).argmax()]
plot((z[:-1] - zmin) / (zmid - zmin),
     (z[1:] - zmin) / (zmid - zmin), '.', ms=1)
axis('scaled')
axis([-5,8,-5,8])
xticks(linspace(-4, 8, 7))
yticks(linspace(-4, 8, 7))
for tick in gca().xaxis.get_major_ticks():
    tick.label.set_fontsize(40)
for tick in gca().yaxis.get_major_ticks():
    tick.label.set_fontsize(40)
savefig('lorenz_zmax_rho_scaled.png')

figure(figsize=(10,10))
plot(z0[:-1], z0[1:], '.', ms=1)
#z = load('lorenz_zmax_10_28_3.0.npy')[:,2]
#plot(z[:-1], z[1:], '.', ms=1)
z = load('lorenz_zmax_10_28_3.3333333333333335.npy')[:,2]
plot(z[:-1], z[1:], '.', ms=1)
axis('scaled')
axis([28,52,28,52])
xticks(linspace(30, 50, 5))
yticks(linspace(30, 50, 5))
for tick in gca().xaxis.get_major_ticks():
    tick.label.set_fontsize(40)
for tick in gca().yaxis.get_major_ticks():
    tick.label.set_fontsize(40)
savefig('lorenz_zmax_beta.png')

figure(figsize=(10,10))
zmid = z0[z0.argmax()-1]
zmin = z0[ma.array(z0[2:], mask=z0[:-2] > z0[1:-1]).argmax()]
plot((z0[:-1] - zmin) / (zmid - zmin),
     (z0[1:] - zmin) / (zmid - zmin), '.', ms=1)
#z = load('lorenz_zmax_10_28_3.0.npy')[:,2]
#zmid = z[z.argmax()-1]
#zmin = percentile(z[z<zmid], 10)
#plot((z[:-1] - zmin) / (zmid - zmin),
#     (z[1:] - zmin) / (zmid - zmin), '.', ms=1)
z = load('lorenz_zmax_10_28_3.3333333333333335.npy')[:,2]
zmid = z[z.argmax()-1]
zmin = z[ma.array(z[2:], mask=z[:-2] > z[1:-1]).argmax()]
plot((z[:-1] - zmin) / (zmid - zmin),
     (z[1:] - zmin) / (zmid - zmin), '.', ms=1)
axis('scaled')
axis([-5,8,-5,8])
xticks(linspace(-4, 8, 7))
yticks(linspace(-4, 8, 7))
for tick in gca().xaxis.get_major_ticks():
    tick.label.set_fontsize(40)
for tick in gca().yaxis.get_major_ticks():
    tick.label.set_fontsize(40)
savefig('lorenz_zmax_beta_scaled.png')
