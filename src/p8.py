from math import *
import scipy.linalg as la
import numpy as np

L = 8
Nvec = list(range(6,36,6))
for N in Nvec:
    h = 2*pi/N
    x = [h*i for i in range(N+1)]
    x = [L*(pi-xi)/pi for xi in x]
    column = [-pi**2/(3*h**2)-1/6]
    column.extend([-.5*(-1)**i/sin(h*i/2)**2 for i in range(1,N+1)])
    D2 = (pi/L)**2*la.toeplitz(column)
    eigenvalues, eigenvectors = la.eig(-D2 + np.diag(np.pow(x,2)))
    eigenvalues = np.sort(eigenvalues)
    print(f'N = {N}')
    print(eigenvalues[:5])
