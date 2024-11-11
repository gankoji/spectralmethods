import matplotlib.pyplot as plt
import scipy.linalg as la
import numpy as np
import math

Nvec = range(2,100,2)

plt.figure()

for N in Nvec:
    # Create the grid
    h = 2*math.pi/N
    x = [-math.pi + h*i for i in range(N)]

    # Calculate the function on that grid, and it's analytical derivative
    u = [math.exp(math.sin(xj)) for xj in x]
    uprime = [math.cos(xj)*uj for xj,uj in zip(x,u)]

    # Create the spectral differentiation matrix
    c = [0]
    c.extend([(1/math.tan(i*h/2))*0.5*(-1)**i for i in range(1,N)])
    r = [1]
    r.extend([c[i] for i in range(N-1,0,-1)])
    D = np.array(la.toeplitz(c,r))

    error = la.norm(np.dot(D,u) - uprime, np.inf)
    plt.loglog(N, error, '.', 'markersize', 15)

plt.grid(True, which='both')
plt.xlabel("N")
plt.ylabel("Error")
plt.title("Convergence of spectral differentiation")
N4 = [n**(-4) for n in Nvec]

plt.show()
plt.savefig('output2.png')
