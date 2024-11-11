import matplotlib.pyplot as plt
import scipy.linalg as la
import numpy as np
import math

Nvec = [2**x for x in range(3,13)] # Nvec = 2.^(3:12)

plt.figure()

for N in Nvec:
    # Create the grid
    h = 2*math.pi/N
    x = [-math.pi + h*i for i in range(N)]

    # Calculate the function on that grid, and it's analytical derivative
    u = [math.exp(math.sin(xj)) for xj in x]
    uprime = [math.cos(xj)*uj for xj,uj in zip(x,u)]

    # Create the 4th order differentiation matrix
    row = [0]*N
    row[1] = 2/3/h
    row[2] = -1/12/h
    row[-1] = -2/3/h
    row[-2] = 1/12/h
    r = np.array(row)
    c = -r.transpose()
    D = np.array(la.toeplitz(c,r))

    error = la.norm(np.dot(D,u) - uprime, np.inf)
    plt.loglog(N, error, '.', 'markersize', 15)

plt.grid(True, which='both')
plt.xlabel("N")
plt.ylabel("Error")
plt.title("Convergence of 4th order finite difference")
N4 = [n**(-4) for n in Nvec]

plt.semilogy(Nvec, N4, '--')
plt.text(105,5e-8,r'$N^{-4}$',fontsize=18, usetex=True)
plt.show()
plt.savefig('output1.png')
