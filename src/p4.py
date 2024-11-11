import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import math

N = 24
h = 2*math.pi/N
x = [h*i for i in range(N)]
col = [0]
col.extend([.5*(-1)**i*(1/math.tan(i*h/2)) for i in range(1,N)])
row = [0]
row.extend([col[i] for i in range(N-1,0,-1)])
D = la.toeplitz(col,row)

plt.figure()

# First, a hat function
v = [max(0, 1-math.fabs(xi-math.pi)/2) for xi in x]
plt.subplot(2,2,1)
plt.title("Function")
plt.plot(x,v)
plt.subplot(2,2,2)
plt.title("Spectral Derivative")
plt.plot(x,np.dot(D,v))

# Then, e^sin(x)
v = [math.exp(math.sin(xi)) for xi in x]
vprime = [math.cos(xi)*math.exp(math.sin(xi)) for xi in x]
plt.subplot(2,2,3)
plt.plot(x,v)
plt.subplot(2,2,4)
plt.plot(x,np.dot(D,v))

error = la.norm(np.dot(D,v) - vprime, np.inf)
plt.text(2.2,1.4,'max error = ' + str(error))
print(error)
plt.show()
plt.savefig('output4.png')
