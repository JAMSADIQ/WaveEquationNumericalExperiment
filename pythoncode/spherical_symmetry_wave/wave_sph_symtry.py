#Implicit scheme check convergence rate in both dt and dx
#python wave_sph_symtry.py    --outputfilename implicitB_60

import time
import numpy as np
import matplotlib.pyplot as plt
import utils_implicit as utils
import utils_plots as uplot
import matplotlib as mpl
import h5py as h5
import scipy
from scipy.interpolate import interp1d, splev, splrep
import argparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--outputfilename', help='h5 and mp4 file names for saving data ', default='outfile', type=str)
parser.add_argument('--path', default='./', help='path for saving data and movie', type=str)
opts = parser.parse_args()


global_mu = 20.0  
#def draw_frame(x, y, time, tn):
#    """
#    To make plots of u 
#    if we want we can also
#    make plot of  first derivatives for space 
#    and time
#    """
#    y2 = np.zeros_like(x)
#    #yt2 = np.zeros_like(x)
#    #yx2 = np.zeros_like(x)
#    for it in range(len(x)):
#        y2[it] = exact_sol(x[it], tn)/x[it]
#        #yx2[it] = dexact_dt(x[it], tn)/x[it]
#        #yt2[it] = dexact_dr(x[it], tn)/x[it]
#    err = np.abs(y - y2)
#    #dterr = np.abs(yt - yt2)
#    fig, axs = plt.subplots(2,1,figsize=(16,9), gridspec_kw={'height_ratios': [2, 1]})
#    axs[0].plot(x, y, 'r', lw =2, label='numeric-U')
#    axs[0].plot(x, y2, 'k--', label='exact-U')
#    #axs[0].plot(x, yt, color='cyan', lw=2, label='numeric-dtU')
#    #axs[0].plot(x, yx, color='orange', lw=2, label='numeric-drU')
#    #axs[0].plot(x, yt2, color='magenta', linestyle='--' ,label='exactdtU')
#    #axs[0].plot(x, yx2, color='brown', linestyle='--' ,label='exactdrU')
#    axs[0].set_ylabel("u at t= {0:.3f}".format(time))
#    axs[0].set_ylim(-1, 1)
#    axs[0].legend()
#    axs[0].set_xlabel("x")
#    axs[1].plot(x, err, label='u')
#    #axs[1].plot(x, dterr, label='d_t u')
#    #axs[1].plot(x, dterr, label='d_r u')
#    axs[1].set_ylabel("|exact - numeric| at t= {0:.3f}".format(time))
#    axs[1].semilogy()
#    axs[1].legend()
#    axs[1].set_ylim(ymin=1e-9, ymax=1.0)
#    axs[1].set_xlabel("x")
#    return 0    
#

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
    #compute the exact solution at tn
    sumSqerr = 0.0
    for it in range(len(x)):
        exact_i = exact_sol(x[it], tn)/x[it]
        sumSqerr += (exact_i - u[it])**2

    return sumSqerr

def L2norm_dx(x, numeric_u, exact_u, tn):
    """
    get L2 norm error at tn 
    given x grid, numeric and exact solu at tn
    """
    dx = x[1] - x[0]
    sumSqerr = 0.0
    for it in range(len(x)):
        exact_i = exact_u(x[it], tn)/x[it]
        sumSqerr += (exact_i - numeric_u[it])**2
        # for L2norm we need sqrt and
        L2norm = np.sqrt(sumSqerr*dx)
    return L2norm

def Max_Abs_error(x, numeric_u, exact_u, tn):
    """
    get L2 norm error at tn 
    given x grid, numeric and exact solu at tn
    """
    for i in range(len(x)):
        uexact[i] = exact_u(x[i], t[n])/x[i]
    error = max(0, np.abs(numeric_u - uexact).max())
    return error


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


#### Numerical case with initial conditions
def init(r):
    r0 = global_mu
    return np.exp(-0.5*(r-r0)**2)

def diff_init(x):
    return 0.0

def source(r, t):
    return  0.0 



def solver(I, V, f, c, L, dt, dx, T, xmin=0, user_action=None, save_data=False, save_movie=False, exact_sol=exact_sol):
    """
    Input params 
        I = initial value of U
        V = initial value of dU/dt 
        f = source term
        L = length of domain (grid) [0 to L]
        dt = time spacing
        dx = space spacing
        T = total duration of simulation
        kwargs
            xmin = 0 
                    starting point [we use staggered grid]
                    and x-grid is dependent on xmin and dx
            user_action = None
               to get numerical and exact solution difference
            save_data = False   
               self explanatory
            save_plot = False
               self explanatory
            exact_sol = given above

    returns
        numerical solution of wave equation
        and convergence with a given exact solution

    Mitchell scheme with source term
    parametrized for time (dt) and spatial (dx) spacing
    CFL is dependent on dt and dx
    """
    # time steps
    Nt = int(round(T/dt))
    t = np.linspace(0, Nt*dt, Nt+1)
    #spatial steps
    Nx = int(round(L/dx))
    x = np.zeros(Nx+1)
    #staggered grid
    for i in range(Nx+1):
        x[i] = (i+0.5)*dx  #try with Nx+2 and start with ghost x =-0.5*dx
    C = c*dt/dx

    if save_data==True:
        foutputh5 = h5.File(opts.outputfilename+'implicit_dx_dt_{0}.hfd5'.format(dt), 'w')
        foutputh5.create_dataset('x_grid', data=x)
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
    
    L2_err = np.zeros_like(t)

    n = 0  #t[0] 
    # using initial conditions u(x, t=0) = I  and du/dt(x, t=0) = V
    #Tridiagonal Solver  RHS 
    d0 = np.zeros(Nx+1, dtype=np.float64) 
    
    for i in range(1, Nx):
        d0[i] = I(x[i+1]) + I(x[i-1]) - 2.0*(1.0 - 2.0/C2)*I(x[i]) \
                - dt*(V(x[i+1]) + V(x[i-1]) - 2.0*(1.0 + 2.0/C2)*V(x[i]))\
                + 2.0*dt**2/(dx*((i+0.5)*dx)*C2)*(I(x[i+1]) - I(x[i-1])) \
                + 2.0*dt**2/C2*f(x[i], t[n])\
    # BCd at x = 0 (indx =0):   du/dr=0  => u-1 = u+1 No 2/r term
    d0[0] =  2.0*I(x[1]) - 2.0*(1.0 - 2.0/C2)*I(x[0]) \
            - dt*(V(x[1]) + V(x[0]) - 2.0*(1.0 + 2.0/C2)*V(x[0]))\
            + 2.0* dt**2/C2*f(x[0], t[n])
    # BCd at x = L (index =Nx): du/dt + c(du/dr + u/r) = 0 
    # we have a complicated expressions see mathematica notebook
    d0[Nx] = 2.0*dt**2/C2*f(x[Nx], t[n]) - 0.5*dx/dt*u_nm3[Nx] \
                + 0.5*(2.0 - 8.0/(2.0*Nx +1.0))*dx/dt*u_nm2[Nx] \
                - dt*(V([Nx]) + V(x[Nx-1]) - 2.0*(1.0 + 2.0/C2)*V(x[Nx]))\
                - 2.0*dx*(3.0 + 16.0/(2.0*Nx + 1.0))*V(x[Nx]) \
                + 2.0*I(x[Nx-1]) - 2.0*I(x[Nx]) \
                + 4.0/C2*I(x[Nx]) + dx/dt*(1.0 - 12.0/(2.0*Nx + 1.0))*I(x[Nx])
 
    #LHS matrix in tridiagonal Solver [-(u[i+1, n+1] + u[i-1, n+1]) + 2(1 +2/C2)u[i, n+1]]

    a0 = np.ones(Nx) * (-1.0)
    c0 = np.ones(Nx) * (-1.0)
    b0 = np.ones(Nx+1)*(2.0 + 4.0/C2)
    #BCds 
    c0[0] = -2.0 # at x=0 (indx=0): u-1 = u+1
    a0[-1] = -2.0 # at x = L (indx =Nx): Complicated
    b0[Nx] +=  3.0*dx/dt + 2*dx/x[Nx] #complicated BCd see mathematica
    u_n[:] = utils.TDMAsolver(a0, b0, c0, d0)
      

    #For du/dt at n=1 or t = dt we need
    for i in range(1, Nx):
       u_nm1[i] = I(x[i])
       u_nm2[i] = u_n[i] - 2.0*dt*V(x[i])


    #L2-norm at t=0 mustbe zero 
    #note in the convergence we can have 0/0
    #so when compute convergence ignore first vals
    tn = 0
    L2_err[0] = L2norm_dx(x, u_nm1, exact_sol, tn)

    #exact solution at first time step
    
    for it in range(len(x)):
        u_e[it] = exact_sol(x[it], t[1])/x[it]
    #uplot.waveplot_fixed_t(x, u_n, u_e, t=dt, drf=None, fymin=-2, fymax=2) 
  

    #Errors save for whole grid and 
    #save the abs or l2 error at this time step
    tn = dt # first time 
    L2_err[1] = L2norm_dx(x, u_n, exact_sol, dt)

    
    if save_data == True:
        foutputh5.create_dataset('u_at_t_{0:04}'.format(t[0]), data=u_n)
        foutputh5.create_dataset('exact_at_t_{0:04}'.format(t[0]), data=u_e)

    # for n > 1 # next time steps
    for n in range(1, int(Nt)):
        timeval = (n+1)*dt
        print("timeval = ", timeval)
        a0 = np.ones(Nx)*(-1.0)
        c0 = np.ones(Nx)*(-1.0)
        b0 = np.ones(Nx+1)*(2.0 + 4.0/C2)
        d0 = np.zeros(Nx+1)
        #BCs 
        c0[0] = -2.0
        a0[-1]= -2.0 
        b0[Nx] += 3.0*dx/dt + 2*dx/x[Nx]
        for i in range(1, Nx):
            d0[i] = 2.0*((u_n[i+1] + u_n[i-1]) - 2.0*(1.0 - 2.0/C2)*u_n[i]) \
                    +((u_nm1[i+1] + u_nm1[i-1]) - 2.0*(1.0 + 2.0/C2)*u_nm1[i])\
                    +4.0*dx**2/(dx*(x[i]))*(u_n[i+1] - u_n[i-1]) \
                    +4.0*dx**2/(c**2)*f(x[i], t[n])\

        # using u_n[-1]  = u_n[+1] for all n# non du/dr term
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
            u_e[it] = exact_sol(x[it], timeval)/x[it]
        
        
        #L2norm
        L2_err[n+1] = L2norm_dx(x, u, exact_sol, timeval)

        if save_data==True:
            foutputh5.create_dataset('u_at_t_{0:04}'.format(timeval), data=u)
            foutputh5.create_dataset('exact_at_t_{0:04}'.format(timeval), data=u_e)

      


        # For Movie making
        if save_movie==True:
            filename='_temp%05d.png'%n
            uplot.draw_frame(x, u, u_e, timeval, t[n+1])
            plt.savefig(filename)
            plt.clf()
            plt.close()

        if user_action is not None:
            if user_action(u, x, t, n+1):
                break
        #switch 
        u_nm3[:] = u_nm2 
        u_nm2[:] = u_nm1
        u_nm1[:] = u_n
        u_n[:] = u

    if save_data==True:
        foutputh5.close()
    #Plot at final time
    for it in range(len(x)):
        u_e[it] = exact_sol(x[it], t[n])/x[it]
    #uplot.waveplot_fixed_t(x, u_n, u_e, t=t[n+1], drf=None, fymin=-2, fymax=2)

    Errgrid_finalT = err_xi(u, x, timeval)


    if save_movie ==True:
        import os
        os.system("/usr/bin/ffmpeg  -framerate 30 -pattern_type glob -i '*.png' -c:v libx264 -pix_fmt yuv420p test_movie_du_dt_{0}_dx_{1}.mp4".format(dt, dx))
        os.system("rm _temp*.png")

    print("we get x grid, t-grid, u(x) at Tend, l2err(t-grid),  absErr[numeric-exact](x) at Tend")
    return x, t, u, u_e, L2_err, Errgrid_finalT


### First we just make a movie for fixed dt and dx
#dt_h = 0.1
#dx_h = 0.08
#Totaltime = int(35)
wavespeed = 1.0
#Lval =  60 #right boundary

#xg, tg, ug, u_exg, l2error, abserr = solver(init, diff_init, source, wavespeed, Lval, dt_h, dx_h, Totaltime, xmin=0, save_data=False, save_movie=True)
#quit()


### Error and Convergence 
Total_time = 30
dt1 = 0.01 #
dx = 0.01
Lval = 60

# First test dt convergence numeric plus exact way
x1, t1, u1, u_e1, l2error1, abserr1 = solver(init, diff_init, source, wavespeed, Lval, dt1, dx, Total_time, xmin=0, save_data=False, save_movie=False)
x2, t2, u2, u_e2,  l2error2, abserr2 = solver(init, diff_init, source, wavespeed, Lval, dt1/2.0, dx, Total_time, xmin=0, save_data=False, save_movie=False)
x3, t3, u3, u_e3,  l2error3, abserr3 = solver(init, diff_init, source, wavespeed, Lval, dt1/4.0, dx, Total_time, xmin=0, save_data=False, save_movie=False)






plt.figure()
plt.plot(x1, abserr1, label='low')
plt.plot(x2, abserr2, label='med')
plt.plot(x3, abserr3, label='high')
plt.plot(x2, 4.0*abserr2, ls='--',label='4*med')
plt.legend()
#plt.show()

#numeric error
low_err = abs(u2 - u1) 
high_err = abs(u3 - u2) 

plt.figure()
plt.plot(x1, low_err, label='low')
plt.plot(x2, high_err, label='high')
plt.plot(x2, 4.0*high_err, ls='--',label='4*high')
plt.legend()
plt.show()

quit()
#get numerical convergence ratehr using exact one to check

f1 = interp1d(x1, u1, kind='cubic', fill_value="extrapolate")
u1interp = f1(x2)
Err1 = np.abs(u1interp - u2)

f2 = interp1d(x2, u2, kind='cubic', fill_value="extrapolate")
u2interp = f2(x3)

Err2 = np.abs(u2interp- u3)

plt.figure()
plt.plot(x2, Err1, 'k', label='h-h2')
plt.plot(x3, Err2, 'r', label='h2-h4')
plt.plot(x3, 4*Err2, color = 'orange', ls='--', label='4*h2-h4')
plt.xlabel('x')

#for fixed dx:check difference  abs error(x) at last time no interpolation need for this
plt.figure()
plt.plot(x1, abserr1, 'k', label='low')
plt.plot(x2, abserr2, 'cyan', ls='-.', label ='middle')
#plt.plot(x3, abserr3, 'r', label ='high')
#plt.plot(x4, abserr4, 'g', label ='highest')
plt.plot(x2, 4.0*abserr2,ls='--', color='orange', label ='4*middle')
#plt.plot(x3, 4.0*abserr3,ls='--', color='cyan', label ='2*high')
#plt.plot(x4, 4.0*abserr4,ls='--', color='purple', label ='4*highest')
plt.legend()
plt.show()

quit()
#for fixed dt we need interpolation
abserr_interp = splrep(x1, abserr1, k=3) # cubic
interp_abs =  splev(x2, abserr_interp)
abserr1 = interp_abs
plt.figure()
plt.title("Max abs error(x) at final time")
plt.plot(x2, abserr1, 'k', lw=2, label='err with dt')
plt.plot(x2, abserr2, 'r', label='err with 1/2dt')
plt.plot(x2, 4*abserr2, color='orange', ls='--',label='4*err with 1/2dt')
plt.plot(x2, 2*abserr2, color='cyan', ls='--',label='2*err with 1/2dt')
#plt.semilogy()
plt.legend()
plt.show()

# Interpolate at t2 time the l2-norm error for each step
#l2err_interp = splrep(t1, l2error1, k=3) # cubic
#interp_l2 =  splev(t2, l2err_interp)
interp_l2 = l2error1
print("error at first grid vals=", l2error1[0], l2error2[0])
# Interpolating solution on same grid point as dt/2

plt.plot(t2,interp_l2 , 'k', lw=2, label='err with dt')
plt.plot(t2, l2error2, 'r', label='err with 1/2dt')
plt.plot(t2, 4*l2error2, color='orange', ls='--',label='4*err with 1/2dt')
plt.plot(t2, 2*l2error2, color='cyan', ls='--',label='2*err with 1/2dt')
#plt.semilogy()
plt.legend()
plt.show()



#without interpolation
t2n = t2[0::2]
E2n = l2error2[0::2]

plt.plot(t1, l2error1, 'k', lw=2, label='err with dt')
plt.plot(t2n, E2n, 'r', label='err with 1/2dt')
plt.plot(t2n, 4*E2n, color='orange', ls='--',label='4*err with 1/2dt')
plt.plot(t2n, 2*E2n, color='cyan', ls='--',label='2*err with 1/2dt')
#plt.semilogy()
plt.legend()
plt.show()


