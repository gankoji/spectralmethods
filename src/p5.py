import matplotlib.pyplot as plt
import scipy.linalg as la
import scipy.fft as fft
import numpy as np
import math

N = 24
a = int(-N/2+1)
b = int(N/2)
h = 2*math.pi/N
x = [h*i for i in range(N)]

plt.figure()

def fft_diff(v):
    vhat = fft.fft(v)
    indices = list(range(b))
    indices.append(0)
    indices.extend([i for i in range(a,0)])
    what = [1j*idx*vi for idx,vi in zip(indices,vhat)]
    w = np.real(fft.ifft(what))
    return w

# First, a hat function
v = [max(0, 1-math.fabs(xi-math.pi)/2) for xi in x]
w = fft_diff(v)

plt.subplot(2,2,1)
plt.title("Function")
plt.plot(x,v)
plt.subplot(2,2,2)
plt.title("Spectral Derivative")
plt.plot(x,w)

# Then, e^sin(x)
v = [math.exp(math.sin(xi)) for xi in x]
vprime = [math.cos(xi)*math.exp(math.sin(xi)) for xi in x]
w = fft_diff(v)
error = la.norm(w - vprime, np.inf)

plt.subplot(2,2,3)
plt.plot(x,v)
plt.subplot(2,2,4)
plt.plot(x,w)
plt.plot(x,vprime)

plt.text(2.2,1.4,'max error = ' + str(error))
plt.show()
plt.savefig('output4.png')
