#Implicit scheme check convergence rate
#need 1: tridiagonal solver for initial guess than use in iterative scheme
#or try inverse matrix 
# First order scheme verses 2nd order scheme
import numpy as np
import matplotlib.pyplot as plt
import utils_implicit as utils

#quadratic in space exact solution
def exact_sol(x, t, L=2.5):
    return x*(L - x)*(1 + 0.5*t)

def I(x):
    return exact_sol(x, 0)

def V(x):
    return 0.5*exact_sol(x, 0)

def f(x, t, c= 1.5):
    return 2*(1 + 0.5*t)*c**2

#First Order Implicit Scheme with Tridiagonal solver
def implicit(x, t):
    """
    choose CFL given 
    dx, dt and c =1.5 in this case
    """
    dx = x[1] - x[0] 
    dt = t[1] - t[0]
    Nx = len(x) - 1
    Nt = len(t) - 1
    c = 1.5
    CFL = c*dt/dx
    print("dx, dt, cfl =", dx, dt, CFL)
    C2  =  CFL**2 # (cdt/dx)^2
    d = np.zeros_like(x) 
    n = 0
    # we need to solve tridiagonal system 
    # we use Thomas algorithm that maybe an issue
    for i in range(Nx+1):
        d[i]  =  I(x[i]) + dt * V(x[i]) + 0.5*dt**2*f(x[i], t[n])
    a = np.ones(len(x) - 1) * (-0.5*C2)
    c = np.ones(len(x) - 1) * (-0.5*C2)
    b = np.ones(len(x))*(1 + C2)
    u_nm1 = np.zeros_like(x) 
    u_n = np.zeros_like(x) 
    u = np.zeros_like(x) 
    for i in range(len(x)):
        u_nm1[i] = I(x[i])
        u_n[:] = utils.TDMAsolver(a, b, c, d) 
    #Bcds
    u_n[0] = 0.0
    u_n[Nx] = 0.0
    diff = np.abs(u_n - exact_sol(x, t[1])).max()
    print("diff = ", round(diff, 3) )
    #Now compare result at this step
    plt.plot(x, exact_sol(x, t[1]), label='exact')
    plt.plot(x, u_n , label='numeric')
    plt.legend()
    plt.title("error = {0:.3e}".format(diff))
    plt.show()
    quit()
    #n=1  and so on
    for n in range(1, Nt):
        aa = np.ones(len(x)-1)*(-C2)
        cc = np.ones(len(x)-1)*(-C2)
        bb = np.ones(len(x))*(1.0 + 2.0*C2)
        dd = np.zeros_like(x)
        for i in range(len(x)):
            dd[i] = u_n[i] - u_nm1[i] + dt**2*f(x[i], t[n])
        u = utils.TDMAsolver(aa, bb, cc, dd)
        #swicth 
        u_nm1[:] = u_n
        u_n[:] = u
        if n%10==0:
            error = np.abs(u - exact_sol(x, t[n])).max()
            plt.plot(x, exact_sol(x, t[n]), label='exact')
            plt.plot(x, u, label='numeric_at t={0:.3f}'.format(t[n]))
            plt.legend()
            plt.title("error= {0:.3e}".format(error))
            plt.show()
    return x, t, u 

x = np.linspace(0, 2.5, 100)
t =  np.linspace(0, 5, 100)
implicit(x, t)


