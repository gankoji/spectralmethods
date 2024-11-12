import matplotlib.pyplot as plt
import scipy.linalg as la
import scipy.fft as fft
import numpy as np
import math

def discretize(N):
    h = 2*math.pi/N
    x = [h*i for i in range(1,N+1)]
    c = [0]
    c.extend([(1/math.tan(i*h/2))*0.5*(-1)**i for i in range(1,N)])
    r = [1]
    r.extend([c[i] for i in range(N-1,0,-1)])
    D = la.toeplitz(c,r)

    return x,D

Nmax = 50
Nvec = list(range(6,Nmax,2))
E = np.zeros((4,len(Nvec)))

for j,N in enumerate(Nvec):
    x,D = discretize(N)

    v = [math.fabs(math.sin(xi))**3 for xi in x]
    vprime = [3*math.sin(xi)*math.cos(xi)*math.fabs(math.sin(xi)) for xi in x]
    E[0,j] = la.norm(np.dot(D,v)-vprime, np.inf)

    v = [math.exp(-math.sin(xi/2)**(-2)) for xi in x]
    vprime = [.5*vi*math.sin(xi)/(math.sin(xi/2)**4) for vi,xi in zip(v,x)]
    E[1,j] = la.norm(np.dot(D,v)-vprime, np.inf)

    v = [1./(1+math.sin(xi/2)**2) for xi in x]
    vprime = [-math.sin(xi/2)*math.cos(xi/2)*(vi**2) for vi,xi in zip(v,x)]
    E[2,j] = la.norm(np.dot(D,v)-vprime, np.inf)

    v = [math.sin(10*xi) for xi in x]
    vprime = [10*math.cos(10*xi) for xi in x]
    E[3,j] = la.norm(np.dot(D,v)-vprime, np.inf)


titles = ['$|sin(x)|^3$','$exp(-sin^{-2}(x/2))$','$1/(1+sin^2(x/2))$','$sin(10x)$']

plt.figure()
for iplot in range(4):
    plt.subplot(2,2,iplot+1)
    plt.semilogy(Nvec,E[iplot],'.',markersize=12) 
    plt.plot(Nvec,E[iplot],linewidth=.8)
    plt.grid(True,which='both')
    plt.xlabel('N')
    plt.ylabel('error')
    plt.title(titles[iplot])

plt.show()
plt.savefig('output7.png')
