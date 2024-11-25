# Polynomials and corresponding equipotential curves
import matplotlib.pyplot as plt
import numpy as np
from math import *

N = 16

plt.figure()

for i in range(1,3):
    if i == 1:
        s = ' equispaced points'
        x = [-1 + 2*j/N for j in range(N+1)]
    else:
        s = ' Chebyshev points'
        x = [cos(pi*j/N) for j in range(N+1)]

    p = np.poly(x)
    xx = np.linspace(-1,1,200)
    pp = np.polyval(p,xx)

    y = [0 for xi in x]
    plt.subplot(2,2,((2*i)-1))
    plt.plot(x,y)
    plt.plot(xx,pp)

    plt.subplot(2,2,2*i)
    plt.plot(np.real(x), np.imag(x))
    xgrid = np.linspace(-1.4,1.4,141)
    ygrid = np.linspace(-1.12,1.12,113)

    print(xgrid)
    print(ygrid)
    xx,yy = np.meshgrid(xgrid, ygrid)
    zz = [xi+1j*yi for xi,yi in zip(xx,yy)]
    pp = np.polyval(p,zz)
    pp = np.absolute(pp)

    levels = [10**(j) for j in range(-4,1)]
    plt.contour(xx,yy,pp,levels)
    plt.xlim(-1.4,1.4)
    plt.ylim(-1.12,1.12)

plt.savefig('output10.png')
