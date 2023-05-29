#Implicit scheme check convergence rate

import time
import numpy as np
import matplotlib.pyplot as plt
import utils_implicit as utils
import matplotlib as mpl
import h5py as h5


global_mu = 20.0  #20.0 # 50.0

def draw_frame(x, y, yt, yx, time, tn):
    """
    To make plots of u 
    and its first derivatives for space 
    and time

    """
    y2 = np.zeros_like(x)
    yt2 = np.zeros_like(x)
    yx2 = np.zeros_like(x)
    for it in range(len(x)):
        y2[it] = exact_sol(x[it], tn)/x[it]
        yx2[it] = dexact_dt(x[it], tn)/x[it]
        yt2[it] = dexact_dr(x[it], tn)/x[it]
    err = np.abs(y - y2)
    dterr = np.abs(yt - yt2)
    fig, axs = plt.subplots(2,1,figsize=(16,9), gridspec_kw={'height_ratios': [2, 1]})
    axs[0].plot(x, y, 'r', lw =2, label='numeric-U')
    axs[0].plot(x, y2, 'k--', label='exact-U')
    axs[0].plot(x, yt, color='cyan', lw=2, label='numeric-dtU')
    axs[0].plot(x, yx, color='orange', lw=2, label='numeric-drU')
    axs[0].plot(x, yt2, color='magenta', linestyle='--' ,label='exactdtU')
    axs[0].plot(x, yx2, color='brown', linestyle='--' ,label='exactdrU')
    axs[0].set_ylabel("u at t= {0:.3f}".format(time))
    axs[0].set_ylim(-1, 1)
    axs[0].legend()
    axs[0].set_xlabel("x")
    axs[1].plot(x, err, label='u')
    #axs[1].plot(x, dterr, label='d_t u')
    axs[1].plot(x, dterr, label='d_r u')
    axs[1].set_ylabel("|exact - numeric| at t= {0:.3f}".format(time))
    axs[1].semilogy()
    axs[1].legend()
    axs[1].set_ylim(ymin=1e-9, ymax=1.0)
    axs[1].set_xlabel("x")
    

def err_xi(u, x, tn):
    """
    absolute error 
    on a grid of x values at a fixed
    time
    """
    err_grid = np.zeros_like(x)
    for it in range(len(x)):
        exact_i = exact_sol(x[it], tn)/x[it]
        err_grid[it]= np.abs(exact_i - u[it])
    return err_grid


def spatial_Sq_sum_error(u, x, tn):
    #compute the exact solution
    #at tn
    #exact = np.zeros_like(x)
    sumSqerr = 0.0
    for it in range(len(x)):
        exact_i = exact_sol(x[it], tn)/x[it]
        sumSqerr += (exact_i - u[it])**2
    return sumSqerr

##### Solution from Enrico's Mathematica Notebook #####
def s0(x, x0=global_mu):
    if x > 0:
        return x*np.exp(-0.5*(x-x0)**2)
    else:
        return x*np.exp(-0.5*(-x-x0)**2)

def exact_sol(r, t):
    r0 = global_mu
    return 1.0/2.0*(s0(-t+r, x0=r0) + s0(t+r, x0=r0))

def dsdtm_t(r, t):
    r0 = global_mu
    if -t+r > 0:
        return np.exp(-0.5*(-t + r - r0)**2)*(-1.0 + (r - t)*(r - t - r0))
    else:
        return np.exp(-0.5*(t - r - r0)**2)*(-1.0 - (r - t)*(t - r - r0))

def dsdtp_t(r, t):
    r0 = global_mu
    if t+r > 0:
        return np.exp(-0.5*(t + r - r0)**2)*(1.0 - (r + t)*(t + r - r0))
    else:
        return np.exp(-0.5*(-t - r - r0)**2)*(1.0 - (r + t)*(t + r + r0))


def dexact_dt(r, t):
    return 1.0/2.0*(dsdtm_t(r, t) + dsdtp_t(r, t))


def dsdtm_r(r, t):
    r0 = global_mu
    if -t+r > 0:
        return np.exp(-0.5*(-t + r - r0)**2)*(1.0 - (r - t)*(r - t - r0))
    else:
        return np.exp(-0.5*(t - r - r0)**2)*(1.0 + (r - t)*(t - r - r0))

def dsdtp_r(r, t):
    r0 = global_mu
    if t+r > 0:
        return np.exp(-0.5*(t + r - r0)**2)*(1.0 - (r + t)*(t + r - r0))
    else:
        return np.exp(-0.5*(-t - r - r0)**2)*(1.0 - (r + t)*(t + r + r0))

def dexact_dr(r, t):
    return 1.0/2.0*(dsdtm_r(r, t) + dsdtp_r(r, t))


def init(r):
    r0 = global_mu
    return np.exp(-0.5*(r-r0)**2)

def diff_init(x):
    return 0.0

def source(r, t):
    return  0.0 

def wavespeed(a, alp):
    """
    we need this factor for wave solution 
    """
    speed = np.zeros_like(a)
    for i in range(len(a)):
        speed[i] = alp[i]/a[i]
    return speed**2

def numpy_deriv(ratio, dr):
    """
    numpy.gradient seems reasonable?
    but we need the dr =fixed scalar val
    """
    return np.gradient(ratio, dr)
    

def non_linear_source(a, alp, Dudt, Dudt, Da_over_alp_dt, Dalp_over_da_dr):
    """
    adding more terms on rhs of wave equation
    """
    return alp/a *(-Dudt*D_a_over_alp_dt + Dudr*D_alp_over_a_dr)


def solver(I, V, f, c, L, dt, C, T, xmin=0,   user_action=None):
    """
    Mitchell scheme with source term
    """
    Nt = int(round(T/dt))
    t = np.linspace(0, Nt*dt, Nt+1)
    dx = dt*c/C
    # fix dx and chnage C here to see
    #C = dt*c/dx # now C is changed here
    Nx = int(round(L/dx))
    print("before dx, dt , CFL= ", dx, dt, C)
    #foutputh5 = h5.File('outputimplicit_dt_{0}.hfd5'.format(dt), 'w')

    #Crucial to use Staggered Grid  
    #ghostpoints = 2
    x = np.zeros(Nx+1)  #np.linspace(0.0, L, Nx+1)
    for i in range(Nx+1):
        x[i] = (i+0.5)*dx
    #foutputh5.create_dataset('x_grid', data=x)

    print("grid ist, 2nd and last points = ", x[0], x[1], x[-1])
    
    C2 = C**2
    #check dx and dt are compatible:
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    print("after dx, dt , CFL= ", dx, dt, C)
    if f is None or f == 0:
        f = lambda x, t: 0
    if V is None or V == 0:
        V = lambda x: 0

    u = np.zeros(Nx+1, dtype=np.float64)
    u_n = np.zeros(Nx+1, dtype=np.float64)
    u_e = np.zeros(Nx+1, dtype=np.float64)
    u_nm1 = np.zeros(Nx+1, dtype=np.float64)
    u_nm2 = np.zeros(Nx+1, dtype=np.float64)
    u_nm3 = np.zeros(Nx+1, dtype=np.float64)
    # We also need derivatives of scalar field for 
    #constraints equation
    Dudt = np.zeros(Nx+1, dtype=np.float64) #PHI
    Dudr = np.zeros(Nx+1, dtype=np.float64) #PI

    
    Errt_array = np.zeros_like(t)
    n = 0 # to get u at n=1 use zeros for
    
    d0 = np.zeros(Nx+1, dtype=np.float64) 
    for i in range(1, Nx):
        d0[i] = I(x[i+1]) + I(x[i-1]) - 2.0*(1.0 - 2.0/C2)*I(x[i]) \
                - dt*(V(x[i+1]) + V(x[i-1]) - 2.0*(1.0 + 2.0/C2)*V(x[i]))\
                + 2.0*dt**2/(dx*((i+0.5)*dx)*C2)*(I(x[i+1]) - I(x[i-1])) \
                + 2.0*dt**2/C2*f(x[i], t[n])\

    d0[0] =  2.0*I(x[1]) - 2.0*(1.0 - 2.0/C2)*I(x[0]) \
            - dt*(V(x[1]) + V(x[0]) - 2.0*(1.0 + 2.0/C2)*V(x[0]))\
            + 2.0* dt**2/C2*f(x[0], t[n])

    d0[Nx] = 2.0*dt**2/C2*f(x[Nx], t[n]) - 0.5*dx/dt*u_nm3[Nx] \
                + 0.5*(2.0 - 8.0/(2.0*Nx +1.0))*dx/dt*u_nm2[Nx] \
                - dt*(V([Nx]) + V(x[Nx-1]) - 2.0*(1.0 + 2.0/C2)*V(x[Nx]))\
                - 2.0*dx*(3.0 + 16.0/(2.0*Nx + 1.0))*V(x[Nx]) \
                + 2.0*I(x[Nx-1]) - 2.0*I(x[Nx]) \
                + 4.0/C2*I(x[Nx]) + dx/dt*(1.0 - 12.0/(2.0*Nx + 1.0))*I(x[Nx])

    a0 = np.ones(Nx) * (-1.0)
    c0 = np.ones(Nx) * (-1.0)
    b0 = np.ones(Nx+1)*(2.0 + 4.0/C2)
    #BCds 
    c0[0] = -2.0
    a0[-1] = -2.0 
    b0[Nx] +=  3.0*dx/dt + 2*dx/x[Nx]
    for i in range(1, Nx):
       u_nm1[i] = I(x[i])

    if user_action is not None:
        user_action(u_nm1, x, t, 0)
    u_n[:] = utils.TDMAsolver(a0, b0, c0, d0)
   
    #exact solution
    for it in range(len(x)):
        u_e = exact_sol(x[it], t[0])/x[it]

    if user_action is not None:
        print("tridiagnol solving here!")
        user_action(u_n, x, t, 1)
    for i in range(1, Nx):
        u_nm2[i] = u_n[i] - 2.0*dt*V(x[i])
    #foutputh5.create_dataset('u_at_t_{0:04}'.format(t[0]), data=u_n)
    #foutputh5.create_dataset('exact_at_t_{0:04}'.format(t[0]), data=u_e)
    for n in range(1, int(Nt)):
        timeval = (n+1)*dt
        a0 = np.ones(Nx)*(-1.0)
        c0 = np.ones(Nx)*(-1.0)
        b0 = np.ones(Nx+1)*(2.0 + 4.0/C2)
        d0 = np.zeros(Nx+1)

        c0[0] = -2.0
        a0[-1]= -2.0 
        b0[Nx] += 3.0*dx/dt + 2*dx/x[Nx]
        for i in range(1, Nx):
            d0[i] = 2.0*((u_n[i+1] + u_n[i-1]) - 2.0*(1.0 - 2.0/C2)*u_n[i]) \
                    +((u_nm1[i+1] + u_nm1[i-1]) - 2.0*(1.0 + 2.0/C2)*u_nm1[i])\
                    +4.0*dx**2/(dx*(x[i]))*(u_n[i+1] - u_n[i-1]) \
                    +4.0*dx**2/(c**2)*f(x[i], t[n])\
        # using u_n[i-1]  = u_n[i] for all n

        d0[0] =  2.0*(2*u_n[1] - 2.0*(1.0 - 2.0/C2)*u_n[0]) \
                    + 2.0*u_nm1[1] - 2.0*(1.0 + 2.0/C2)*u_nm1[0]\
                    + 4.0* dt**2/C2*f(x[0], t[n]) 
        
        # using u_n[i+1] = u_n[i-1] -2dx*(u_n[i]/x[i] +(3/2dt u_n[i]  -2/dt u_nm1[i] + 1/2dt u_nm2[i])) for all n

        d0[Nx] = 4.0*dx**2/(c*c)*f(x[Nx], t[n]) - dx/dt*u_nm3[Nx] \
                - 4.0*dx**2/dt*u_nm2[Nx]/x[Nx] + 2.0*dx/dt*u_nm2[Nx] \
                + 2.0*u_nm1[Nx-1] - 2.0*u_nm1[Nx] \
                - 4.0*dx**2/(c**2*dt**2)*u_nm1[Nx] \
                + 4.0*dx/dt*u_nm1[Nx] - 2.0*dx/x[Nx]*u_nm1[Nx]\
                + 16.0*dx**2/dt*u_nm1[Nx]/x[Nx] + 4.0*u_n[Nx-1] \
                - 4.0*u_n[Nx] + 8.0*dx**2/c**2*u_n[Nx]/dt**2 \
                - 2.0*dx/dt*u_n[Nx] - 8.0*dx**2/x[Nx]**2*u_n[Nx]\
                - 4.0*dx/x[Nx] *u_n[Nx] - 12.0*dx**2/dt*u_n[Nx]/x[Nx]


       
        u[:] = utils.TDMAsolver(a0, b0, c0, d0)
        #exact solution
        for it in range(len(x)):
            u_e = exact_sol(x[it], timeval)/x[it]
        
        #foutputh5.create_dataset('u_at_t_{0:04}'.format(timeval), data=u)
        #foutputh5.create_dataset('exact_at_t_{0:04}'.format(timeval), data=u_e)
        Dudt[:] = 1.0/dt*(3.0/2.0*u[:] - 2.0*u_n[:] + 1.0/2.0*u_nm1[:])
        Dudr = np.gradient(u, dx)
        Dudr[0] = Dudr[1]
        Dudr[Nx] = u[Nx-1] -2.0*dx*(Dudt[Nx]- u[Nx]/x[Nx])
        #draw_frame(x, u, Dudt, Dudr, timeval, t[n+1])


        #For L2 norm errors
        #sum_sq_spatial = spatial_Sq_sum_error(u, x, t[n+1])
        #Errt_array[n] =  sum_sq_spatial
        #Err_t += sum_sq_spatial #spatial_Sq_sum_error(u, x, t[n+1]) 
        # For Movie making
        #filename='_temp%05d.png'%n
        #draw_frame(x, u, Dudt,Dudr, timeval, t[n+1])
        #plt.savefig(filename)
        #plt.clf()
        #plt.close()

        if user_action is not None:
            if user_action(u, x, t, n+1):
                break
        #switch 
        u_nm3[:] = u_nm2 
        u_nm2[:] = u_nm1
        u_nm1[:] = u_n
        u_n[:] = u
        #dudt[:] = 1.0/dt*(3.0/2.0*u_n[:] - 2.0*u_nm1[:] + 1.0/2.0*u_nm2[:])

    foutputh5.close()
    #Err exact - numeric for each x after time t[n]
    Errgrid = err_xi(u, x, timeval)#t[n+1])
    #draw_frame(x, u, Dudt, timeval, t[n+1])
    #plt.show()


    #import os
    #os.system("/usr/bin/ffmpeg  -framerate 30 -pattern_type glob -i '*.png' -c:v libx264 -pix_fmt yuv420p movie_dudt_{0}_test70.mp4".format(dt))
    #os.system("rm _temp*.png")

    #return u, x, t

    return x, t, Errgrid


### Error and Convergence. 
### Error at a fixed time for each grid point
dt1 = 0.01 # dt and C are fixed
CFL = 1.0
#dxfix = 0.05
x1, t1, E1 = solver(init, diff_init, source, 1, 50, dt1, CFL, 5)
x2, t2, E2 = solver(init, diff_init, source, 1, 50, dt1/2, CFL, 5)

# Interpolating solution on same grid point as dt/2
x2n = x2[0::2]
E2n = E2[0::2]

E1 = np.interp(x2, x1, E1) 
plt.plot(x2, E1, 'k', lw=2, label='err with dt')
plt.plot(x2, E2, 'r', label='err with 1/2dt')
plt.plot(x2, 4*E2, color='orange', ls='--',label='4*err with 1/2dt')
#plt.semilogy()
plt.legend()
plt.show()
quit()
#c=1, xmax =100, dt =0.05/2, cfl=1, time=2, 
#solver(init, diff_init, source, 1, 100, 0.05/2.0, 1.0, 70, user_action=None)
#quit()

#%%dt0 = 0.1 
#%%Totaltime = 25
#%%d = []
#%%E = []
#%%Ei = []
#%%for i in range(6):
#%%    dt1, E1 = solver(init, diff_init, source, 1, 60, dt0, 0.9, Totaltime)
#%%    d.append(dt1)
#%%    E.append(E1)
#%%    Ei.append((i+1)*4*E1)
#%%    dt0 = dt0/2.0
#%%
#%%np.savetxt("dtvsL2norm_afterT_25.txt", np.c_[d, E])
#%%plt.plot(d, E)
#%%plt.semilogy()
#%%plt.figure()
#%%plt.plot(d, Ei)
#%%plt.show()
#%%quit()
#for Totaltime in [20]:
#Totaltime  = 5
dt1 = 0.01 # dt and C are fixed
CFL = 1.5
#dxfix = 0.05
x1, t1, E1 = solver(init, diff_init, source, 1, 60, dt1, dxfix, Totaltime)
x2, t2, E2 = solver(init, diff_init, source, 1, 60, dt1/2, dxfix, Totaltime)


#print(len(t1), len(t2))
#print(t1[-1], t2[-1])

quit()
#for i in range(len(x1)):
#    print(t1[i], t2[i])
#quit()
#x2n = x2[0::2]
#E2n = E2[0::2]

#E1 = np.interp(t2, t1, E1) 
#plt.plot(x2, E1, label='err with dt')
#plt.plot(x2, E2, label='4*err with 1/2dt')
#%%plt.semilogy()
#plt.title("T = {0}".format(Totaltime))
#plt.xlabel("r", fontsize=15)
#plt.ylabel("|u_exact - u_numeric|", fontsize=15)
#plt.legend()
#plt.ylim(ymax=8)

#plt.figure()
#plt.plot(x2, np.abs(np.log2(E2/E1)))
#%%plt.plot(x2, 4.0*E2, label='1/2dt')
#%%plt.legend()
#plt.title("T = {0}".format(Totaltime))
#plt.xlabel("r", fontsize=15)
#plt.ylabel(r"$log2[E_{0.5h} / E_{h}]", fontsize=15)
#plt.ylim(0, 1e2)
#plt.show()
#quit()
#x2n = x2.copy()
#E2n = E2.copy()
#%%err = np.zeros_like(x1) 
#%%for i in range(len(x1)):
#%%    Eh2,Eh = E2n[i], E1[i]
#%%    print(Eh2, Eh)
#%%    if Eh2 < 1e-20 or Eh < 1e-20:
#%%        err[i] = 0.0
#%%    else:
#%%        err[i] = abs(np.log2(Eh2/Eh))
#%%
#%%err[np.isnan(err)]=0.0
#%%
#%%plt.figure(figsize=(12, 7))
#%%plt.plot(x1, err,'r' )
#%%plt.xlabel("x")
#%%plt.ylabel("log(E_{h/2} / E_h)")
#%%plt.title("total_time = {0}".format(Totaltime))
#%%#plt.savefig("convergence_after_T={0}".format(Totaltime))
#%%plt.show()
#%%quit()
#%%#plt.figure(figsize=(8,5))
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
        print("n =", n, len(x))
        uexact = np.zeros_like(x)
        for i in range(len(x)):
            uexact[i] = u_exact(x[i], t[n])/x[i]
        if n == 0:
            error = 0
        else:
            #error = max(error, np.abs(u[int(len(x)/2.0)]- u_exact(x[int(len(x)/2.0)], t[n])).max())
            error = max(error, np.abs(u[:len(x)] - uexact[:]).max())
        if plot_u ==True:

            if n==10 or n%100==0 or n==2 or n==int(NTval-2):
                plt.plot(x, u[:len(x)], label='numeric')
                plt.plot(x, uexact, 'r--', alpha=0.8, label='exact')
                #plt.plot(x, u - u_exact(x, t[n]), 'k+')
                plt.legend()
                plt.xlabel('x')
                #plt.xlim(0, 60)
                plt.ylabel('u(x, t={0:.3e})'.format(t[n]))
                plt.title('n={0}'.format(n))
                plt.pause(0.01)
                plt.title("Max_abs_error={0:.5e}".format(error))
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




def test_convrate_quadratic():
    L = 60.0
    Amp =1.0
    sigma= 1.0#0.5
    r0 = 20.0
    def us0(x, x0=20):
        if x > 0:
            return x*np.exp(-0.5*(x-x0)**2)
        else:
            return x*np.exp(-0.5*(-x-x0)**2)

    def u_exact(r, t):
        r0 = 20
        return 1.0/2.0*(us0(-t+r, x0=r0) + us0(t+r, x0=r0))

    def V(x):
        return 0.0
#        if x==0.0:
#           return 0
#        else:
            #return Amp/(x**2 * sigma**2)*(x*r0 -x**2 -sigma**2)*np.exp(-0.5*((x- r0)/sigma)**2)
#            return Amp/(x*sigma**2)*(x - r0)*np.exp(-0.5*((x- r0)/sigma)**2)
    #u_exact = lambda x, t: Amp/x * np.exp(-0.5*((x- r0-t)/sigma)**2) 
    r = convergence_rates(
        u_exact=u_exact,
        I=lambda x: np.exp(-0.5*(x-r0)**2),
        V=V#lambda x,t: lambda x: Amp/(x**2 * sigma**2)*(x*r0 -x**2 -sigma**2)*np.exp(-0.5*((x- r0)/sigma)**2)
,
        f = lambda x, t : 0.0, 
        c = 1,
        L = L,
        dt0 = 0.1,
        num_meshes = 6,
        C = 1.5,
        T = 5)
    print( 'rates spherical wave  solution:', \
          [round(r_,2) for r_ in r])
 


if __name__ == '__main__':
    main()
    print('see')

