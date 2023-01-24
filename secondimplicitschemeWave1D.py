#Implicit scheme check convergence rate
#need 1: tridiagonal solver for initial guess than use in iterative scheme
#or try inverse matrix 
# First order scheme verses 2nd order scheme
import numpy as np
import matplotlib.pyplot as plt
import utils_implicit as utils

def exact_sol(x, t, L=2.5):
    return x*(L - x)*(1 + 0.5*t)


def I(x):
    #exact solution at t = 0
    return exact_sol(x, 0)

def V(x):
    return 0.5*exact_sol(x, 0)

def f(x, t, c=1.0):
    return 2*(1 + 0.5*t)*c**2


def implicit(x, t):
    dx = x[1] - x[0] 
    dt = t[1] - t[0]
    Nx = len(x) - 1
    Nt = len(t) - 1
    c = 1.0
    CFL = c*dt/dx
    print("dx, dt, cfl =", dx, dt, CFL)
    C2  =  CFL**2 # (cdt/dx)^2
    d = np.zeros_like(x) 
    n = 0
    unm1 = np.zeros_like(x)
    un = np.zeros_like(x)
    u = np.zeros_like(x)
    for i in range(len(x)):
        un[i] = I(x[i])
    # Tridiagonal Solver for second order scheme Mitchell book 199-200 page
    for i in range(1, Nx):
        d[i]  = un[i+1] - 2.0*(1.0 - 2.0/C2) *un[i] + un[i-1] - dt *(V(x[i+1]) - 2.0*(1.0  + 2.0/C2)*V(x[i]) + V(x[i-1])) + 0.5* dt**2*f(x[i], t[n])  
    d[0] = un[1] + un[0] - 2.0*(1.0 -2.0/C2) * un[0] - dt *(V(x[1]) + V(x[0]) - 2.0*(1.0 + 2.0/C2) *V(x[0]) ) + 0.5*dt**2*f(x[0], t[n])
    d[Nx] = un[Nx] + un[Nx-1] - 2.0*(1.0 - 2.0/C2) *un[Nx] - dt *(V(x[Nx]) - 2*(1.0 + 2.0/C2) *V(x[Nx]) + V(x[Nx-1])) + 0.5*dt**2*f(x[Nx], t[n])

    a = np.ones(len(x) - 1) * (-1.0)
    c = np.ones(len(x) - 1) * (-1.0)
    b = np.ones(len(x))*(2.0 + 4.0/C2)
    for i in range(len(x)):
        u = utils.TDMAsolver(a, b, c, d) 
    #Bcds
    u[0] = 0.0
    u[Nx] = 0.0
    #Now compare result at t = 0 
    error = np.abs(un - exact_sol(x, t[0])).max()
    plt.figure()
    plt.plot(x, exact_sol(x, t[0]), label='exact')
    plt.plot(x, un , label='numeric')
    plt.legend()
    plt.title("at t=0, error = {0:.5e}".format(error))
    plt.show()
    error1 = np.abs(u - exact_sol(x, t[1])).max()
    plt.figure()
    plt.plot(x, exact_sol(x, t[1]), label='exact')
    plt.plot(x, u , label='numeric')
    plt.legend()
    plt.title("at t=1dt, error = {0:.5e}".format(error1))
    plt.show()
    print("diff = ", error1) 
    #n=1  and so on
    #switch un, unm1
    unm1[:] = un
    un[:] = u
    for n in range(1, Nt):
        aa = np.ones(len(x) - 1) * (-1.0)
        cc = np.ones(len(x) - 1) * (-1.0)
        bb = np.ones(len(x))*(2.0 + 4.0/C2)
        dd = np.zeros_like(x)
        for i in range(1, len(x)-1):
            dd[i] =  2.0*(un[i+1] - (2.0 - 4.0/C2)*un[i] + un[i-1]) + 1.0*(unm1[i+1] - (2.0 + 4.0/C2)*unm1[i] + unm1[i-1]) + dt**2*f(x[i], t[n])
        dd[0] =  2.0*(un[1] - 2.0*(1.0 - 2.0/C2) *un[0] + un[0]) + 1.0*(unm1[1] - 2.0*(1.0 + 2.0/C2) *unm1[0] + unm1[0]) + dt**2*f(x[0], t[n])
        dd[-1] = 2.0*(un[-1] - 2.0*(1.0 - 2.0/C2) *un[-1] + un[-2]) + 1.0*(unm1[-1] - 2.0*(1.0 + 2.0/C2) *unm1[-1] + unm1[-2]) + dt**2*f(x[-1], t[n])
        
        u = utils.TDMAsolver(aa, bb, cc, dd)
        u[0] = 0.0
        u[Nx] = 0.0
        #swicth 
        unm1[:] = un
        un[:] = u
        if n%10==0:
            error = np.abs(un - exact_sol(x, t[n])).max()
            plt.plot(x, exact_sol(x, t[n]), label='exact')
            plt.plot(x, u, label='numeric_at t={0:.3f}'.format(t[n]))
            plt.legend()
            plt.title("error = {0:.5e}".format(error))
            plt.show()
    return x, t, u 

x = np.linspace(0, 2.5, 200)
t =  np.linspace(0, 1, 100)
implicit(x, t)


