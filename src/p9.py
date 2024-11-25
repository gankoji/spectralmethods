import matplotlib.pyplot as plt
import scipy.linalg as la
import scipy.fft as fft
import numpy as np
from math import *

N = 16
xx = np.arange(-1.01,1.01,.005)

plt.figure()

for i in range(2):
    if i==0:
        s = 'Equispaced points'
        x = np.arange(-1,1+1/N,2/N)
    else:
        s = 'Chebyshev points'
        x = [cos(pi*i/N) for i in range(N+1)]

    plt.subplot(1,2,i+1)
    u = [1/(1 + 16*(xi**2)) for xi in x]
    uu = [1/(1 + 16*(xxi**2)) for xxi in xx]
    p = np.polyfit(x,u,N)
    pp = np.polyval(p,xx)
    plt.plot(x,u,'.',markersize=12)
    plt.plot(xx,pp,linewidth=.8)
    plt.title(s)
    error = la.norm(pp-uu, np.inf)
    plt.text(-.5,.5, f'max error = {error:.6f}')

plt.show()
plt.savefig('output9.png')
