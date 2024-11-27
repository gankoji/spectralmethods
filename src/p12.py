import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from math import *
from cheb import cheb

Nmax = 50
E = np.zeros((4,Nmax))

for N in range(1,Nmax+1):
    D,x = cheb(N)
    v = np.array([fabs(xi)**3 for xi in x])
    vp = np.array([3*xi*fabs(xi) for xi in x])
    E[0,N-1] = la.norm(np.dot(D,v) - vp, np.inf)

    v = np.array([exp(-(xi**(-2))) for xi in x])
    vp = np.array([2*vi/(xi**3) for vi,xi in zip(v,x)])
    E[1,N-1] = la.norm(np.dot(D,v) - vp, np.inf)

    v = np.array([1/(1+(xi**2)) for xi in x])
    vp = np.array([-2*xi*(vi**2) for xi,vi in zip(x,v)])
    E[2,N-1] = la.norm(np.dot(D,v) - vp, np.inf)

    v = np.array([xi**10 for xi in x])
    vp = np.array([10*(xi**9) for xi in x])
    E[3,N-1] = la.norm(np.dot(D,v) - vp, np.inf)

titles = ['$|x^3|$','$exp(-x^{-2})$','$1/(1+x^2)$','$x^{10}$']
plt.figure()
for iplot in range(1,5):
    plt.subplot(2,2,iplot)
    plt.semilogy(list(range(1,Nmax+1)), E[iplot-1,:],'.')
    plt.plot(list(range(1,Nmax+1)), E[iplot-1,:])
    plt.axis([0,Nmax,1e-16, 1e3])
    plt.title(titles[iplot-1])
    plt.xlabel('N')
    plt.ylabel('error')

mng = plt.get_current_fig_manager()
mng.full_screen_toggle()
plt.show()
plt.savefig('output12.png')

