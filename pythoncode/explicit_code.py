#!/usr/bin/env python
# python explicit_code.py  
# Note that in this code we keep dx  or dt =fixed and stay CFL under control and check convergence of resolutions [ulow-umed]  [umed-uhigh] and see if we get second order convergence but it do not work
# in order to get convergence we need to keep nCFL fix and vary dt, which will also vary dx to keep CFL fixed.

# we hope this will not be the case in implcit integrator


import numpy as np
import os
import matplotlib.pyplot as plt
import scipy 
from scipy.interpolate import interp1d, splev, splrep


def solver(I, V, f, c, L, dt, dx, T, user_action=None):
    """
    Solve  u_tt(x, t) = c^2 * u_xx(x, t) + f(x, t)
    on x: [0,L]  and t: (0,T]
    """
    Nt = int(round(T/dt))
    t = np.linspace(0, Nt*dt, Nt+1)   # Mesh points in time
    Nx = int(round(L/dx))
    x = np.linspace(0, L, Nx+1) # Mesh points in space
    C = c*dt/dx
    #assert or use this
    if C > 1.0:
        print("CFL condition violated, fix dx, dt")
        quit()
    else:
        print("CFL is < 1")
    C2 = C**2                         # Help variable in the scheme
    # Make sure dx and dt are compatible with x and t
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    if f is None or f == 0 :
        f = lambda x, t: 0
    if V is None or V == 0:
        V = lambda x: 0

    u     = np.zeros(Nx+1)   
    u_n   = np.zeros(Nx+1)   
    u_nm1 = np.zeros(Nx+1)   

    # Load initial condition into u_n
    for i in range(0, Nx+1):
        u_n[i] = I(x[i])

    if user_action is not None:
        user_action(u_n, x, t, 0)

    # Special formula for first time step
    n = 0
    for i in range(1, Nx):
        u[i] = u_n[i] + dt*V(x[i]) + \
               0.5*C2*(u_n[i-1] - 2*u_n[i] + u_n[i+1]) + \
               0.5*dt**2*f(x[i], t[n])
    u[0] = 0;  u[Nx] = 0

    if user_action is not None:
        user_action(u, x, t, 1)

    #uexact = u_exact_sol(x, t[n])
    u_nm1[:] = u_n;  u_n[:] = u

    for n in range(1, Nt):
        # Update all inner points at time t[n+1]
        for i in range(1, Nx):
            u[i] = - u_nm1[i] + 2*u_n[i] + \
                     C2*(u_n[i-1] - 2*u_n[i] + u_n[i+1]) + \
                     dt**2*f(x[i], t[n])

        # Insert boundary conditions
        u[0] = 0;  u[Nx] = 0
        if user_action is not None:
            if user_action(u, x, t, n+1):
                break

        # Switch variables before next step
        u_nm1[:] = u_n;  u_n[:] = u
       
    return  u, x, t




def test_quadratic():
    """
    Check that u(x, t) = x(L -x)(1 + 0.5*t) is
    exactly reproduced given f, V, I
    for solution of d2/dt2u = d2u/dx2 + f(x,t) 
    """
    def u_exact(x, t):
        return x*(L-x)*(1 + 0.5*t)

    def I(x):
        return u_exact(x, 0)

    def V(x):
        return 0.5*u_exact(x, 0)

    def f(x, t):
        return 2*(1 + 0.5*t)*c**2

    L = 2.5 
    c = 1.0
    dx = 0.01
    dt = 0.99*dx # CFL will be <1 C*(L/Nx)/c
    T = 10
    def assert_no_error(u, x, t, n):
        u_e = u_exact(x, t[n])
        diff = np.abs(u - u_e).max()
        tol = 1E-12
        if diff < tol:
            print("success")
        assert diff < tol

    solver(I, V, f, c, L, dt, dx, T, user_action=assert_no_error)
    u1, x1, t1  = solver(I, V, f, c, L, dt, dx, T)
    u2, x2, t2  = solver(I, V, f, c, L, dt/2.0, dx, T)
    u3, x3, t3  = solver(I, V, f, c, L, dt/4.0, dx, T)
    diff1 = np.abs(u1 - u2)
    diff2 = np.abs(u2 - u3)
    plt.figure()
    plt.title("convergence for O(dt)^2")
    plt.plot(x1, diff1, lw=2, label = 'low')
    plt.plot(x1, diff2, label = 'high')
    plt.plot(x1, 4*diff2, ls='--', alpha=0.8, label= '4*high')
    plt.xlabel("x")
    plt.legend()
    plt.ylabel("residual at final time ={0}". format(T))
    plt.show()
    #Convergence in dx
    # we need to be careful here for CFL prop 1/dx
    # we also need interpolation or reduce domain size
    # for difference
    dt = 0.01
    dx =  4.0*0.01
    u1, x1, t1  = solver(I, V, f, c, L, dt, dx, T)
    u2, x2, t2  = solver(I, V, f, c, L, dt, dx/2.0, T)
    u3, x3, t3  = solver(I, V, f, c, L, dt, dx/4.0, T)
    #u2atx1 = u2[0::2]
    f2_x1 = interp1d(x2, u2)
    f3_x1 = interp1d(x3, u3)
    u2_x1 = f2_x1(x1)
    u3_x1 = f3_x1(x1)
    diff1 = np.abs(u1 - u2_x1)
    #u3atx2 = u3[0::2]
    diff2 = np.abs(u2_x1 - u3_x1)
    f1 = interp1d(x1, u1, kind='cubic', fill_value="extrapolate")
    f2 = interp1d(x2, u2, kind='cubic', fill_value="extrapolate")
    diff1 = np.abs(u2 - f1(x2))
    diff2 = np.abs(u3 - f2(x3))
    plt.figure()
    plt.title("convergence for O(dx)^2")
    plt.plot(x2, diff1, lw=2, label = 'low')
    plt.plot(x3, diff2, label = 'high')
    plt.plot(x3, 4*diff2, ls='--', alpha=0.8, label= '4*high')
    plt.xlabel("x")
    plt.legend()
    plt.ylabel("residual at final time ={0}". format(T))
    plt.show()



def convergence_rates(
    u_exact,                 # Python function for exact solution
    I, V, f, c, L,           # physical parameters
    dt0, num_meshes, dx, T):  # numerical parameters
    """
    Half the time step and estimate convergence rates for
    for num_meshes simulations.
    """
    # First define an appropriate user action function
    global error
    error = 0   #error computed in the user action function

    def compute_error(u, x, t, n):
        global error  # must be global to be altered here
        # (otherwise error is a local variable, different
        # from error defined in the parent function)
        if n == 0:
            error = 0
        else:
            error = max(error, np.abs(u - u_exact(x, t[n])).max())

    # Run finer and finer resolutions and compute true errors
    E = []
    h = []  #dt, solver adjusts dx such that C=dt*c/dx
    dt = dt0
    for i in range(num_meshes):
        solver(I, V, f, c, L, dt, dx, T,
               user_action=compute_error)
        # error is computed in the final call to compute_error
        E.append(error)
        h.append(dt)
        dt /= 2   #halve the time step for next simulation
    print( 'E:', E)
    print('h:', h)
    plt.figure(figsize=(8,4))
    plt.plot(h, E, 'r-*', label='hvsE')
    plt.plot(np.array(h), np.array(h)**2, 'k:', label='hvsh^2')
    plt.plot(np.array(h), np.array(h), 'b:', label='hvsh')
    plt.xlabel("dt", fontsize=15)
    plt.ylabel("abs error", fontsize=15)
    plt.title("convergence with dx = {0} and fixed dt".format(dx))
    plt.loglog()
    plt.grid()
    plt.legend(fontsize=15)
    plt.show()
     #Convergence rates for two consecutive experiments
    r = [np.log(E[i]/E[i-1])/np.log(h[i]/h[i-1])
         for i in range(1,num_meshes)]
    return r

#
def test_convrate_sincos():
    n = m = 2
    L = 1.0
    u_exact = lambda x, t: np.cos(m*np.pi/L*t)*np.sin(m*np.pi/L*x)

    r = convergence_rates(
        u_exact=u_exact,
        I=lambda x: u_exact(x, 0),
        V=lambda x: 0,
        f=0,
        c=1,
        L=L,
        dt0=0.1,
        num_meshes=6,
        dx=0.1,
        T=5)
    print( 'rates sin(x)*cos(t) solution:', \
          [round(r_,2) for r_ in r])
    #assert abs(r[-1] - 2) < 0.002



def test_convrate_quadratic():
    L = 2.5
    u_exact = lambda x, t: x*(L-x)*(1 + 0.5*t) 
    r = convergence_rates(
        u_exact=u_exact,
        I=lambda x: u_exact(x, 0),
        V=lambda x: 0.5*u_exact(x, 0),
        f=lambda x, t : 2*(1 + 0.5*t),
        c=1,
        L=L,
        dt0=0.1,
        num_meshes=6,
        dx=0.1,
        T=2)
    print( 'rates x(L-x)(1+1/2t)  solution:', \
          [round(r_,2) for r_ in r])
    #assert abs(r[-1] - 2) < 0.002




if __name__ == '__main__':
    print('see')
