import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
from math import *
from cheb import cheb

xx = np.linspace(-1, 1, 200)
uu = [exp(xi)*sin(5*xi) for xi in xx]

plt.figure()
plt.title('Chebyshev differentiation of $f(x) = e^{x}sin(5x)$')
for N in [10, 20]:
    D, x = cheb(N)
    u = np.array([exp(xi)*sin(5*xi) for xi in x])
    plt.subplot(2,2,int(2*N/10 - 1))
    if N == 10:
        plt.title('f(x)')
    plt.plot(xx,uu)
    plt.plot(x,u,'bo')
    uprime = np.array([exp(xi)*(sin(5*xi) + 5*cos(5*xi)) for xi in x])
    error = np.dot(D,u) - uprime
    plt.subplot(2,2, int(2*N/10))
    if N == 10:
        plt.title('error in $f\'(x)$')
    plt.plot(error)

plt.show()
plt.savefig('output11.png')


