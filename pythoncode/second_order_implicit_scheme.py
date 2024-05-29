#Implicit scheme check convergence rate
import numpy as np
import matplotlib.pyplot as plt
import utils_implicit as utils


globL = 2.5 #1 #2.0*np.pi

def exact_sol(x,t, L=globL):
    #return np.sin(np.pi*x)*np.cos(np.pi*t)#x*(L-x)*(1+0.5*t)
    return x*(L-x)*(1+0.5*t)
def init(x):
    return exact_sol(x, 0)

def diff_init(x):
    return 0.5*exact_sol(x, 0) #0.0

def source(x, t):
    return  2*(1 + 0.5*t) #0.0




def solver(I, V, f, c, L, dt, dx, T, user_action=None):
    """
    Mitchell scheme careful with source term
    and ist step calculationsgrid 
    """
    Nt = int(round(T/dt))
    t = np.linspace(0, Nt*dt, Nt+1)
    Nx = int(round(L/dx))
    x = np.linspace(0, L, Nx+1)
    C = dt*c/dx
    C2 = C**2
    #check dx and dt are compatible:
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    print("after dx, dt , CFL= ", dx, dt, C)
    if f is None or f == 0:
        f = lambda x, t: 0
    if V is None or V == 0:
        V = lambda x: 0
    u = np.zeros(Nx+1)
    u_n = np.zeros(Nx+1)
    u_nm1 = np.zeros(Nx+1)
    n = 0 # actually this will be step 1 as we have u at t=0 and at t=-1
    d0 = np.zeros(Nx+1) 
    for i in range(1, Nx):
        d0[i]  =  I(x[i+1])  + I(x[i-1]) - 2.0*(1.0 - 2.0/C2)*I(x[i]) \
                - dt*(V(x[i+1]) + V(x[i-1]) - 2.0*(1.0 + 2.0/C2)*V(x[i]))\
                + 2*dt**2/C2*f(x[i], t[n])
    d0[0] = I(x[1]) + I(x[0]) - 2.0*(1.0 - 2.0/C2)*I(x[0]) \
            - dt*(V(x[1]) + V(x[0]) - 2.0*(1.0 + 2.0/C2)*V(x[0]))\
                + 2*dt**2/C2*f(x[0], t[n])
    d0[Nx] = I(x[Nx]) + I(x[Nx-1]) - 2.0*(1.0 - 2.0/C2)*I(x[Nx]) \
                - dt*(V(x[Nx]) + V(x[Nx-1]) - 2.0*(1.0 + 2.0/C2)*V(x[Nx]))\
                +2*dt**2/C2*f(x[Nx], t[n])

    a0 = np.ones(Nx) * (-1.0)
    c0 = np.ones(Nx) * (-1.0)
    b0 = np.ones(Nx+1)*(2.0 + 4.0/C2)

    for i in range(Nx+1):
        u_nm1[i] = I(x[i])
    #remeber in implicit scheme we need an extra time step
    if user_action is not None:
        user_action(u_nm1, x, t, 0)
    #we get u at n=1 now
    u_n[:] = utils.TDMAsolver(a0, b0, c0, d0)
    u_n[0] = 0
    u_n[Nx] = 0
    u_ex = exact_sol(x, t[n+1], L=globL)
    plt.figure()
    plt.plot(x, u_n, 'k', label='numeric')
    plt.plot(x, u_ex, 'r--', label='exact')
    plt.legend()
    plt.figure()
    plt.plot(x, u_n - u_ex)
    plt.show()
    #Bcds
    #now we check error at n=1
    if user_action is not None:
        user_action(u_n, x, t, 1)

    print("total Time, total steps", T, Nt)
    for n in range(1, int(Nt)): 
        a0 = np.ones(Nx)*(-1.0)
        c0 = np.ones(Nx)*(-1.0)
        b0 = np.ones(Nx+1)*(2.0 + 4.0/C2)
        d0 = np.zeros(Nx+1)
        #a0[0] = 0.0
        #a0[-1] = 0.0
        #c0[0] = 0.0
        #c0[-1]=0.0
        for i in range(1, Nx):
            d0[i] = 2.0*((u_n[i+1] + u_n[i-1]) - 2.0*(1.0 - 2.0/C2)*u_n[i]) \
                    +((u_nm1[i+1] + u_nm1[i-1]) - 2.0*(1.0 + 2.0/C2)*u_nm1[i])\
                    + 4.0*dt**2/C2*f(x[i], t[n])
        d0[0] =  2.0*((u_n[1] + u_n[0]) - 2.0*(1.0 - 2.0/C2)*u_n[0]) \
                    + ((u_nm1[1] + u_nm1[0]) - 2.0*(1.0 + 2.0/C2)*u_nm1[0])\
                    + 4.0* dt**2/C2*f(x[0], t[n])
        d0[Nx] = 2.0*((u_n[Nx] + u_n[Nx-1]) - 2.0*(1.0 - 2.0/C2)*u_n[Nx]) \
                    + ((u_nm1[Nx] + u_nm1[Nx-1]) - 2.0*(1.0 + 2.0/C2)*u_nm1[Nx])\
                    + 4.0*dt**2/C2*f(x[Nx], t[n])
        u[:] = utils.TDMAsolver(a0, b0, c0, d0)
        u[0] = 0.0
        u[Nx] = 0.0

        Errgrid = np.abs(u - exact_sol(x, t[n+1], L=globL))
        # Now we will get u at n+1 given values at n and n-1
        #in user function we need exact solution at n+1
        if user_action is not None:
            if user_action(u, x, t, n+1):
                break
        #switch 
        u_nm1[:] = u_n
        u_n[:] = u
        #saving the data for x, t and u[x,t] for 2D plots

    u_ex = exact_sol(x, t[n+1], L=globL)
    u_ex2 = exact_sol(x, t[n], L=globL)
    plt.figure()
    plt.plot(x, u, 'k', label='numeric')
    plt.plot(x, u_ex, 'r--', label='exact')
    plt.plot(x, u_ex2, 'g:', alpha= 0.8, label='exact previous t')
    plt.legend()

    plt.figure()
    plt.plot(x, np.abs(u_n - u_ex), label='exact - numeric')
    plt.legend()
    plt.semilogy()
    plt.show()
 
    #return u, x, t 
    return x, Errgrid, u, t






dx1 = 0.01/2.0
dt1 = 0.01/2.0
Totaltime = 5
xmax = globL#2.5 
x1, E1, u1, t1 = solver(init, diff_init, source, 1, xmax, dt1, dx1, Totaltime)
x2, E2, u2, t2 = solver(init, diff_init, source, 1, xmax, dt1/2.0, dx1, Totaltime)
x3, E3, u3, t3 = solver(init, diff_init, source, 1, xmax, dt1/4.0, dx1, Totaltime)
ratiohigh = np.abs(u3 - u2)
ratiolow = np.abs(u2 -u1 )


plt.figure()
plt.plot(x3, ratiolow, label='h-h2')
plt.plot(x3, ratiohigh, label='h2-h4')
plt.plot(x3, 4.0*ratiohigh, ls=':' , label='4.0*h2-h4')
plt.xlabel("r")
plt.ylabel("residual")
plt.legend()
plt.figure(figsize=(8,5))
plt.plot(x1, E1, label='low', lw=2)
plt.plot(x2, E2, label='medium')
plt.plot(x3, E3, label='high')
plt.plot(x2, 4*E2, alpha=0.8, ls='--', label='4*medium')
plt.xlabel("x")
plt.ylabel("Abs[Exact- Numeric]")
plt.title('O(dt^2)   total time = {0}'.format(Totaltime))
plt.legend()
plt.show()

