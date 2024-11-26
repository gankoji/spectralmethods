import numpy as np
from math import *

def cheb(N):
    if N <= 0:
        return 0, 1

    xs = [cos(pi*i/N) for i in range(N+1)]
    D = np.zeros((N+1,N+1))
    for i in range(N+1):
        for j in range(N+1):
            if i==0 and j==0:
                D[0,0] = D00(N)
            elif i==0 and j==N:
                D[i,j] = D0N(N)
            elif i==N and j==0:
                D[i,j] = -D0N(N)
            elif i==N and j==N:
                D[i,j] = DNN(N)
            elif i==j:
                D[i,j] = DNjj(j,xs,N)
            else:
                D[i,j] = DNij(i,j,xs,N)

    return D, xs
                

def c(i, N):
    if i==0 or i==N:
        return 2
    return 1

def D00(N):
    return (2*N**2+1)/6

def D0N(N):
    return 1/2*((-1)**N)

def DNN(N):
    return -D00(N)

def DNjj(j,xs,N):
    return -xs[j]/(2*(1-xs[j]**2))

def DNij(i,j,xs,N):
    ci = c(i,N)
    cj = c(j,N)
    xi = xs[i]
    xj = xs[j]

    return (ci/cj)*((-1)**(i+j))/(xi-xj)

if __name__ == '__main__':
    print(cheb(1))
    print(cheb(2))
    print(cheb(3))
