import matplotlib.pyplot as plt
import scipy.linalg as la
import scipy.fft as fft
import numpy as np
import math


def fft_diff(v, n):
    vhat = fft.fft(v)
    a = int(-n/2+1)
    b = int(n/2)
    indices = list(range(b))
    indices.append(0)
    indices.extend([i for i in range(a,0)])
    what = [1j*idx*vi for idx,vi in zip(indices,vhat)]
    w = np.real(fft.ifft(what))
    return w

N = 128
h = 2*math.pi/N
x = [i*h for i in range(N)]

t = 0
dt = h/4

c = [0.2 + (math.sin(xi-1))**2 for xi in x]
v = [math.exp(-100*(xi-1)**2) for xi in x]
vold = [math.exp(-100*(xi - .2*dt - 1)**2) for xi in x]
data = [v]
tdata = [t]

tmax = 8
tplot = 0.15

plt.figure()
plotgap = int(tplot/dt)
dt = tplot/plotgap
nplots = int(tmax/tplot)

for i in range(nplots):
    for j in range(plotgap):
        t += dt
        w = fft_diff(v,N)
        dv = [-2*dt*ci*wi for ci,wi in zip(c,w)]
        vnew = [voi + dvi for voi,dvi in zip(vold,dv)]
        vold = v
        v = vnew
    data.append(v)
    tdata.append(t)
    plt.plot(x,v)

plt.show()
plt.savefig("output6.png")       

