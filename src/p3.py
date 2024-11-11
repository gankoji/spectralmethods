import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import math

h = 1
xmax = 5

def v1(x):
    return x==0

def v2(x):
    return math.fabs(x) <= 3

def v3(x):
    return max(0, 1 - math.fabs(x)/3)

x = np.arange(-xmax, xmax+h, h)
xx = np.arange(-xmax-h/20,xmax+h/20,h/10)
plt.figure()
for t in range(3):
    plt.subplot(3,1,t+1)
    
    if t == 0:
        v = [v1(xj) for xj in x]
    elif t == 1:
        v = [v2(xj) for xj in x]
    else:
        v = [v3(xj) for xj in x]

    plt.plot(x,v,'.',markersize=14)
    plt.grid(visible=True,which='both')

    p = [0 for xxj in xx]
    for i,xi in enumerate(x):
        for j,xxj in enumerate(xx):
            # We evaluate our interpolant at each xj, and sum them all
            a = (xxj-xi)/h
            b = math.pi*a
            p[j] += v[i]*math.sin(b)/b

    plt.plot(xx,p,linewidth=.7)

plt.subplot(3,1,1)
plt.title('Comparison of band-limited interpolation vs smoothness')
plt.show()
plt.savefig('output3.png')

    
