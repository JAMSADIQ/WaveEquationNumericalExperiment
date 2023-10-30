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
    plt.plot(x, u_n - u_ex)
    plt.show()
 
    #return u, x, t 
    return x, Errgrid, u, t

##solver(I, V, f, c, L, dt, C, T, user_action=None)





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
quit()


#
#x1, E1, u1, t1 = solver(init, diff_init, source, 1, xmax, dt1, dx1, Totaltime)
#x2, E2, u2, t2 = solver(init, diff_init, source, 1, xmax, dt1/2.0, dx1, Totaltime)
#
#plt.figure(figsize=(8,5))
#plt.plot(x1, E1, label='low')
#plt.plot(x2, E2, label='high')
#plt.plot(x2, 4*E2, label='4*high')
#plt.xlabel("x")
#plt.ylabel("Abs[Exact- Numeric]")
#plt.title('O(dt^2)   total time = {0}'.format(Totaltime))
#plt.legend()
#plt.show()
#
#

dt1 = 0.001
dx1 = 0.002
x1, E1, u1, t1 = solver(init, diff_init, source, 1, xmax, dt1, dx1, Totaltime)
x2, E2, u2, t2 = solver(init, diff_init, source, 1, xmax, dt1, dx1/2.0, Totaltime)
#### Interpolate onto x2

import scipy
from scipy.interpolate import interp1d, splev, splrep
f = interp1d(x1, E1, kind='cubic')
E1interp = f(x2)

plt.figure(figsize=(8,5))
plt.plot(x2, E1interp, label='low')
plt.plot(x2, E2, label='high')
plt.plot(x2, 4*E2, label='4*high')
plt.xlabel("x")
plt.ylabel("Abs[Exact- Numeric]")
plt.title('O(dx^2) total time = {0}'.format(Totaltime))
plt.legend()
plt.show()
quit()
##plt.ylim(ymax=4)
#plt.title("total_time = 1, dt = {0}".format(dt1))
##plt.savefig("complete_Log_ratio_error_with_dt_{0}.png".format(dt1))
##plt.show()
#plt.figure(figsize=(8,5))
#plt.plot(x1, E1, label="dt")
#plt.plot(x2n, E2n, label="dt/2")
#plt.xlabel("x")
#plt.ylabel("|Exact - Numeric|")
#plt.semilogy()
#plt.title("dt = 0.001")
##plt.ylim(ymax=1e-6)
#plt.legend()
#plt.title("total_time = 1, dt = {0}".format(dt1))
##plt.savefig("complete_error_at_steps_with_{0}.png".format(dt1))
#plt.show()
#quit()
def convergence_rates(
    u_exact,                 # Python function for exact solution
    I, V, f, c, L,           # physical parameters
    dt0, num_meshes, C, T):  # numerical parameters
    """
    Half the time step and estimate convergence rates for
    for num_meshes simulations.
    """
    # First define an appropriate user action function
    global error
    error = 0  # error computed in the user action function

    def compute_error(u, x, t, n, plot_u=False):
        global error  # must be global to be altered here
        # (otherwise error is a local variable, different
        # from error defined in the parent function)
        print("n =", n)
        if n == 0:
            error = 0
        else:
            #error = max(error, np.abs(u[int(len(x)/2.0)]- u_exact(x[int(len(x)/2.0)], t[n])).max())
            error = max(error, np.abs(u - u_exact(x, t[n])).max())
        if plot_u ==True:
            if n==10:
                plt.figure()
                plt.plot(x, u, label='numeric')
                plt.plot(x, u_exact(x, t[n]), label='exact')
                plt.plot(x, u - u_exact(x, t[n]), 'k+')
                plt.legend()
                plt.xlabel('x')
                plt.ylabel('u(x, t={0:.3e})'.format(t[n]))
                #plt.title("Max_abs_error={0:.5e}".format(error))
                plt.show()

    # Run finer and finer resolutions and compute true errors
    E = []
    h = []  # dt, solver adjusts dx such that C=dt*c/dx
    dt = dt0
    for i in range(num_meshes):
        solver(I, V, f, c, L, dt, C, T,
               user_action=compute_error)
        E.append(error)
        h.append(dt)
        dt /= 2  # halve the time step for next simulation
    #print( 'E:', E)
    #print('h:', h)
    plt.figure(figsize=(8,4))
    plt.plot(h, E, 'r-*', label='hvsE')
    plt.plot(np.array(h), np.array(h)**2, 'k:', label='hvsh^2')
    plt.plot(np.array(h), np.array(h), 'b:', label='hvsh')
    plt.xlabel("h")
    plt.ylabel("abs error")
    plt.loglog()
    plt.grid()
    plt.legend()
    plt.show()
    # Convergence rates for two consecutive experiments
    r = [np.log(E[i]/E[i-1])/np.log(h[i]/h[i-1])
         for i in range(1,num_meshes)]
    return r




#def test_convrate_sincos():
#    n = m = 2
#    L = 1.0
#    u_exact = lambda x, t: np.cos(m*np.pi/L*t)*np.sin(m*np.pi/L*x)
#
#    r = convergence_rates(
#        u_exact=u_exact,
#        I=lambda x: u_exact(x, 0),
#        V=lambda x: 0,
#        f=lambda x, t:0,
#        c=1,
#        L=L,
#        dt0=0.1,
#        num_meshes=6,
#        C=0.9,
#        T=2)
#    print( 'rates sin(x)*cos(t) solution:', \
#          [round(r_,2) for r_ in r])
######    assert abs(r[-1] - 2) < 0.002

#def test_con_exp_sin():
#    L = 2.0
#    u_exact = lambda x, t: 0.5*(np.exp((x+t)**2)*np.sin(x+t) + np.exp((x-t)**2)*np.sin(x-t)  )
##
#    r = convergence_rates(
#        u_exact=u_exact,
#        I=lambda x: u_exact(x, 0),
#        V=lambda x: 0,
#        f=lambda x, t:0,
#        c=1,
#        L=L,
#        dt0=0.1,
#        num_meshes=6,
#        C=0.9,
#        T=1)
#    print( 'rates sin(x+-)*exp(x+-t^2) solution:', \
#          [round(r_,2) for r_ in r])
######    assert abs(r[-1] - 2) < 0.002
###

def test_convrate_quadratic():
    L = 1.0
    u_exact = lambda x, t: x*(L-x)*(1 + 0.5*t)
    r = convergence_rates(
        u_exact=u_exact,
        I=lambda x: u_exact(x, 0),
        V=lambda x: 0.5*u_exact(x, 0),
        f=lambda x, t : 2*(1 + 0.5*t), #*c**2,
        c=1,
        L=L,
        dt0=0.1,
        num_meshes=6,
        C=0.9,
        T=5)
    print( 'rates x(L-x)(1+1/2t)  solution:', \
          [round(r_,2) for r_ in r])
 


if __name__ == '__main__':
    print('see')

