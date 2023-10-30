import numpy as np
import matplotlib.pyplot as plt

import utils_implicit as utils
from test_constraint import constraint_solver

from utils_diagnostics import *
from utils_plot import draw_frame 
import matplotlib as mpl
import h5py as h5
import os
import scipy
from scipy import interpolate


import argparse
parser = argparse.ArgumentParser()
#we want to change parameters of initial data [a0, sigma0] and dx dt stepsizes
parser.add_argument('--a0', type=float, default=0.14, help='amplitude of pulse in initial dat')
parser.add_argument('--sigma0', type=float, default=1.5, help='width of pulse in initial data')
parser.add_argument('--hx', type=float, default=0.25, help='stepsize in x')
parser.add_argument('--ht', type=float, default=0.5, help='stepsize in t')
parser.add_argument('--finalT', type=int, default=80, help='final time to end simulation')
# will add more options like diagonstic plots etc
#parser.add_argument('--savefig', action='store_true')
out = parser.parse_args()


# argparser with choice of initial data params and use 
#######################Main Code ###########################

#Initial Data for phi or scalar field Miguel
def init(r, a0=0.01, r0=55.0, sigma0=2.0):
    """
    Miguels initial data 
    """
    return a0*np.exp(-0.5*(r-r0)**2/sigma0**2)*np.sin((r-r0)/sigma0/np.sqrt(2.0))

def drdphi(phi_func, r):
    """
    derivative  w r t r for scalar field
    it will also be used in constriant solution 
    at initial time    
    r = rgrid
    phi_func = derivative of init function

    """
    return np.gradient(phi_func, r)


### Initial Data for BIGphi by Laura Bernard paper
def init_lauraPHI(phi, r, r0=55.0, Case='A'):
     """   
     caseA a0=0.02, sig = 15.0
     caseB a0 = 0.14, sig= 1.5
     caseC a0 =0.045, sig =15.0
     """
     if Case == 'A':
         a0 = 0.02; sigma0=15.0
     elif Case == 'B':
         a0 = 0.14;  sigma0=1.5
     else:
         Case == 'C'
         a0=0.045; sigma0=15.0

     return a0*np.exp(-(r-r0)**2/sigma0**2)*np.cos(np.pi/10*r)

### Only if we need to solve Lauras equation or use PHI initial data 
### we need initial phi(r)  via integration for constraints
def rk4_init(r_min, r_max, r_grid, f):
        """
        given f(Phi=dphi/dr, x)
        and min and max of domain x
        and our initial staggered grid length
        [if we want to have staggered grid values
         we will use more finer grid such that our 
         rk-4 resulting grid have ourt actual grid values]
        get phi(x) on x grid based on
        a dx which is say finer than the staggered grid in our problem
        and than get interpolant that return
        value of initial function given  any point of staggered grid
        """
        a = r_min
        b = r_max
        dr = r_grid[1] - r_grid[0]
        #print("r_grid = ", r_grid)
        h = dr/2.0
        #if  we use staggered grid starting with -dr/2 we need to modify this
        tpoints = np.arange(a,b+4*h, h)
        print("rk grid = ", tpoints)
        xpoints = []
        x = 0.0 # initial value of dphi/dr = 0
        for t in tpoints:
            xpoints.append(x)
            k1 = h*f(x,t)
            k2 = h*f(x+0.5*k1,t+0.5*h)
            k3 = h*f(x+0.5*k2,t+0.5*h)
            k4 = h*f(x+k3,t+h)
            x += (k1+2*k2+2*k3+k4)/6.0
####Its better to get values using interpolation 
        staggered_tp = tpoints[1:-1:2]
        #check if r_grid == staggered_tp
        for i in range(len(r_grid)):
            if r_grid[i] == staggered_tp[i]:
                print('ok')
                phi_data = xpoints[1: -1: 2]
                phi_data_on_staggered_grid = np.array(phi_data)
            else:
                print('fix the grid, lent, lenx = ', len(r_grid), len
                        (staggered_tp))
                finterp = interpolate.interp1d(tpoints, xpoints)
                phi_data_on_staggered_grid =  finterp(r_grid)
        return phi_data_on_staggered_grid


def diff_init(x):
    """
    dphi/dt at t=0 which is 
    alpha/a * PI  but its zero in 
    out case at t=0 
    """
    return 0.0 


def source(r, t):
    return  0.0 

def wavespeed(a, alp):
    """
    we need this factor for wave solution 
    """
    speed = alp/a 
    return speed**2


#def Source_wave_part2overr(r_val, a_val, alp_val, bigPHI):
#    """
#    second term in wave in spherical symmetry
#    (alpha/a)^2  * 2/r dphi/dr
#    bigPHI = dphi/dr
#    """
#    return (alp_val/a_val)**2*2.0/r_val*bigPHI
#


def Source1(a_val, alp_val, bigPI, Da_over_alp_dt):
    """
    term associated with dtphi
    """
    return -(alp_val/a_val)**2*bigPI*Da_over_alp_dt

def Source2(a_val, alp_val, bigPHI, Dalp_over_da_dr):
    """
    term associated with drphi
    """
    return (alp_val/a_val)*bigPHI*Dalp_over_da_dr


def G_Source(rval, Aval, Alphaval, PHIval, PIval, drAval, drPHIval, drPIval, g=1.0):
    """
    From Laura's Paper g terms in eq 
    """
    fac_ratio = Alphaval/Aval
    factor = 2.*g/(Aval**2 + g*(PHIval**2 - 3.0*PIval**2))*Alphaval/Aval
    T1 = (PHIval**2 + PIval**2) * drPHIval
    T2 = -2.0*PHIval*PIval*drPIval
    T3 = (rval/4.0*PIval**2 - drAval/Aval)*(PHIval**2 - PIval**2)*PHIval
    T4 = g*rval/(4.0*Aval**2)*(PHIval**2 - PIval**2)**2 *PHIval*PIval**2
    T5 = 2.0/rval*PHIval*PIval**2
    
    return fac_ratio*factor*(T1+T2+T3+T4+T5)


def sum_square_val_sqrt(fx, dx, type='midpoint'):
    """
    return sum of square of one[/all] values and sqrt of that sum
    type= midpoint, start or all
    """
    if type =='midpoint':
        fval = fx[int(len(fx)/2)]
    elif type == 'start':
        fval = fx[0]
    elif type=='all':
        fval = fx
    else:
        print('choose option from [midpoint, start, all]')
    return np.sqrt(np.sum(fval**2) *dx)



def solver(I, V, f, c, L, dt, dx, T, xmin=0, FLAG1=1.0, FLAG2=1.0, FLAG3=1.0, user_action=None, makemovie=False, savedata=False):
    """
    Mitchell scheme with source term
    Inputs:
    I is initial fucntion or scalar field
    V is dI/dt  at t=0
    f is source term need to chnage this
    L is max of space domain 
    dt time step
    dx space step
    T is total time
    xmin is min of space domain
    FLAG1extra termI see notes  dtphi
    FLAG2 extra tremII drphi
    FLAG3 g terms  or extra nonj BOX wave terms
    other kwargs about convergence and data  and plots
    """
    Nt = int(round(T/dt))
    t = np.linspace(0, Nt*dt, Nt+1)
    Nx = int(round(L/dx))
    #ghostpoints = 2 # if we start
    x = np.zeros(Nx+1)  #np.linspace(0.0, L, Nx+1)

    for i in range(Nx+1):
        x[i] = (i+0.5)*dx
    C = c*dt/dx
    C2 = C**2 #
    #check dx and dt are compatible:
    dx = x[1] - x[0]
    dt = t[1] - t[0]

    print("after dx, dt", dx, dt)

    # to be safe if f and V are not given use 0 by default
    if f is None or f == 0:
        f = lambda x, t: 0
    if V is None or V == 0:
        V = lambda x: 0
 

    if savedata==True:
        outf = h5.File('fixN_gp1_laura_caseA_out_file_npgraddx_{0}_dt_{1}.h5'.format(dx, dt), 'w')
        outf.create_dataset('r_grid', data=x)
        outf.create_dataset('t_grid', data=t)
        agrp = outf.create_group("Ar")
        alpgrp = outf.create_group("Alpr")
        ugrp = outf.create_group("ur")
        PHIgrp = outf.create_group("BigPHI")
        PIgrp = outf.create_group("BigPI")
        drPHIgrp = outf.create_group("DrBigPHI")
        drPIgrp = outf.create_group("DrBigPI")
        dtPHIgrp = outf.create_group("DtBigPHI")
        dtPIgrp = outf.create_group("DtBigPI")
        drAgrp = outf.create_group("DrA")
        dtAgrp = outf.create_group("DtA")

    Ix = rk4_init(min(x), max(x), x, init_lauraPHI)
    plt.plot(x, Ix, label='numeric phi')
    #implicit scheme?
    I = interpolate.interp1d(x, Ix, kind='cubic')
    #print(I(x[0]), I(x[-2]))
    newI = I(x[:]) 
    #plt.plot(x, newI, label='from interp')
    #plt.legend()
    #plt.show()
    #quit()
    ##just to check numpy gradient we do have our own code
    #PHI_grid = utils.FD_second(x, I(x))
    PHI_grid = np.gradient(Ix, x)
    #plt.plot(x, PHI_grid, 'k',label='numeric PHI')
    #plt.plot(x, init_lauraPHI(0, x),'r--', label='True')
    #plt.xlabel('r')
    #plt.legend()
    #plt.figure()
    #plt.title("actual difference")
    #plt.plot(x, np.abs(PHI_grid - init_lauraPHI(0, x)))
    #plt.semilogy()
    #plt.show()
    #quit()
    #PHI_grid = drdphi(I(x), x) #init_lauraPHI(0.0, x) 
    # we checked this is second order accurate FD
    PI_grid = np.zeros_like(x)

    ##### Getting  a(r) and alpha(r) from constriants 
    xgrid, A, Alp = constraint_solver(x, PHI_grid, PI_grid, ratio=False)
    # we may plot convergence here at each step
    print("initially maxCFL is = ", max(Alp/A*C))
    rho_n = Alp/A   #need this array in evolution every step
    rho_sq = rho_n**2  
    ratio_AoverAlpha_nm2 = np.zeros_like(x)
    ratio_AoverAlpha_nm1 = A/Alp  #a./alpha is not rho or speed
    # we also need dtA so we use same
    A_nm2 = np.zeros_like(x)
    temp = A.copy()
    A_nm1 = temp # to avoid issues of python
    
    # we need T-deriv of A/alpha  and A
    Dtratio_AoverAlpha = np.zeros_like(x) 
    DtA = np.zeros_like(x) 
    DtPHI = np.zeros_like(x) 
    DtPI = np.zeros_like(x) 

    #we also need space-derivative of Alp/A or rho_n
    #Dr_over_ratio_rho = np.zeros_like(x)  #if g!=0 we must use this
    #Dr_rho = utils.FD_second(x, rho_n)
    Dr_rho = np.gradient(rho_n, x)

    # for g terms we need these derivs too
    #drA = utils.FD_second(x, A)
    #drPHI_grid = utils.FD_second(x, PHI_grid)
    #drPI_grid = utils.FD_second(x, PI_grid)
    drA = np.gradient(A, x)
    drPHI_grid = np.gradient(PHI_grid, x )
    drPI_grid = np.gradient(PI_grid, x)
    ## we need dt PHI and PI as well for tau_tr diagonastic
    PHI_nm2 = np.zeros_like(x)
    tempPHI = PHI_grid.copy()
    PHI_nm1 = tempPHI
    PI_nm2 = np.zeros_like(x)
    tempPI = PI_grid.copy()
    PI_nm1 = tempPHI
    
    Saved_sqrt_sum_sq_u_at_t = []

    Saved_sqrt_sum_sq_u_at_t.append(sum_square_val_sqrt(I(x), dx, type='all'))
    u = np.zeros(Nx+1, dtype=np.float64)
    u_n = np.zeros(Nx+1, dtype=np.float64)
    u_nm1 = np.zeros(Nx+1, dtype=np.float64)
    u_nm2 = np.zeros(Nx+1, dtype=np.float64)
    u_nm3 = np.zeros(Nx+1, dtype=np.float64)

    #Lets start evolution
    n = 0       #to get u at n=1 use zero
    ###LHS without Boundary condition


    d0 = np.zeros(Nx+1, dtype=np.float64) 

    #RHS matrix d of   [Ax=d] equations
    #using vectorization if it works
    
    d0[1:Nx] = rho_sq[1:Nx]*(I(x[1+1:Nx+1]) + I(x[1-1:Nx-1])) - 2.0*(rho_sq[1:Nx] - 2.0/C2)*I(x[1:Nx]) \
            - dt*(rho_sq[1:Nx]*(V(x[1+1:Nx+1]) + V(x[1-1:Nx-1])) - 2.0*(rho_sq[1:Nx] + 2.0/C2)*V(x[1:Nx]))\
            + rho_sq[1:Nx]*(2.0*dx**2*2.0/(2.0*dx*x[1:Nx])*(I(x[1+1:Nx+1]) - I(x[1-1:Nx-1])))\
            + 2.0*dx**2*(f(x[1:Nx], t[n]) + FLAG1*(Source1(A[1:Nx], Alp[1:Nx], PI_grid[1:Nx], Dtratio_AoverAlpha[1:Nx]) + Source2(A[1:Nx], Alp[1:Nx], PHI_grid[1:Nx], Dr_rho[1:Nx])) + FLAG2*G_Source(x[1:Nx], A[1:Nx], Alp[1:Nx], PHI_grid[1:Nx], PI_grid[1:Nx], drA[1:Nx], drPHI_grid[1:Nx], drPI_grid[1:Nx], g=1.0))

    # BCd at  r =0
    d0[0] =  rho_sq[0]*2.0*I(x[1]) - 2.0*(rho_sq[0] - 2.0/C2)*I(x[0])\
            - dt*(rho_sq[0]*(V(x[1]) + V(x[0])) - 2.0*(rho_sq[0] + 2.0/C2)*V(x[0]))\
            + 2.0*dx**2*(f(x[0], t[n]) +  FLAG1*(Source1(A[0], Alp[0], PI_grid[0], Dtratio_AoverAlpha[0]) + Source2(A[0], Alp[0], PHI_grid[0], Dr_rho[0])) + FLAG2*G_Source(x[0], A[0], Alp[0], PHI_grid[0], PI_grid[0], drA[0], drPHI_grid[0], drPI_grid[0], g=1.0) )

    #BCd at r --> 100 fix it  for correct V(x terms )
    d0[Nx] = 0.5*(rho_n[Nx]*(-dx/dt)*u_nm3[Nx] \
                + rho_n[Nx]*(2.0*dx/dt - 4.0*dx**2/(dt*x[Nx]))*u_nm2[Nx] \
                + (-2.0*dt)*(2.0*rho_sq[Nx]*V(x[Nx-1]) - 2.0*(rho_sq[Nx] + 2.0/C2)*V(x[Nx])) \
                + (6.0*rho_n[Nx]*dx/dt - 2.0*rho_sq[Nx]*dx/x[Nx] \
                + 16.0*rho_n[Nx]*dx**2/(x[Nx]*dt))*-2.0*dt*V(x[Nx])
                + 4.0*rho_sq[Nx]*I(x[Nx-1]) - 4.0*(rho_sq[Nx] - 2.0/C2)*I(x[Nx]) \
                + (-10.0*rho_n[Nx]*dx/dt -12.0*rho_n[Nx]*dx**2/(x[Nx]*dt) \
                - 8.0*rho_sq[Nx]*dx**2/((x[Nx])**2) - 4.0*rho_sq[Nx]*dx/x[Nx])*I(x[Nx]))\
                + 2.0*dx**2*(f(x[Nx], t[n]) + FLAG1*(Source1(A[Nx], Alp[Nx], PI_grid[Nx], Dtratio_AoverAlpha[Nx]) + Source2(A[Nx], Alp[Nx], PHI_grid[Nx], Dr_rho[Nx])) + FLAG2*G_Source(x[Nx], A[Nx], Alp[Nx], PHI_grid[Nx], PI_grid[Nx], drA[Nx], drPHI_grid[Nx], drPI_grid[Nx], g=1.0)  )


#### LHS matrix or Tridaigonal matrix entries
    a0= -rho_sq
    c0 = -rho_sq
    b0 = (2.0*rho_sq + np.ones_like(rho_sq)*4.0/C2)
    #BCd at t=0
    c0[0] = -2.0*rho_sq[0]
    a0[-1] = -2.0*rho_sq[-1]
    #fixing Bcd with velocity factor A/alpha 
    b0[Nx] += rho_sq[Nx]*(2.0*dx/x[Nx]) + rho_n[Nx]*(3.0*dx/dt)
    
    for i in range(1, Nx):
       u_nm1[i] = I(x[i])
    if user_action is not None:
        user_action(u_nm1, x, t, 0)
        
    u_n[:] = utils.TDMAsolver(a0, b0, c0, d0)
     
    Saved_sqrt_sum_sq_u_at_t.append(sum_square_val_sqrt(u_n, dx, type='all'))
    #save hdf data
    if savedata==True:
        agrp.create_dataset('Ar_iter{0:04}'.format(n), data=A) 
        alpgrp.create_dataset('Alpr_iter{0:04}'.format(n), data=Alp) 
        ugrp.create_dataset('Ur_iter{0:04}'.format(n), data=u_n) 
        PHIgrp.create_dataset('PHI_iter{0:04}'.format(n), data=PHI_grid) 
        PIgrp.create_dataset('PI_iter{0:04}'.format(n), data=PI_grid) 
     
    if user_action is not None:
        print("tridiagnol solving here!")
        user_action(u_n, x, t, 1)
    for i in range(1, Nx):
        u_nm2[i] = u_n[i] - 2.0*dt*V(x[i])
    Tarr = []
    VpMax = []
    VmMax = []
    VpMin = []
    VmMin = []
    AlpMax = []
    Tau_tr = []
    for n in range(1, int(Nt)):
        #Dudt can use utils.T_derivsecond_order(u_n, u_nm1, u_mn2, dt)\
        PI_grid = ratio_AoverAlpha_nm1*(3./2.*u_n - 2.0*u_nm1 + 1.0/2.0*u_nm2)/dt
        #PHI_grid = utils.FD_second(x, u_n)
        PHI_grid = np.gradient(u_n, x)
        #PI_grid = np.zeros_like(x)
        xgrid, A, Alp = constraint_solver(x, PHI_grid, PI_grid, ratio=False)
        rho_n = Alp/A # need this array in evolution every step
        rho_sq = rho_n**2
        ratio_AoverAlpha_n = A/Alp
        #Dr_rho = utils.FD_second(x, rho_n)
        Dr_rho = np.gradient(rho_n, x)
        # need for g-terms
        Dtratio_AoverAlpha = (3./2.*ratio_AoverAlpha_n - 2.0*ratio_AoverAlpha_nm1 + 1./2.*ratio_AoverAlpha_nm2)/dt
        DtA = (3./2.*A - 2.0*A_nm1 + 1./2.*A_nm2)/dt
        DtPHI = (3./2.*PHI_grid - 2.0*PHI_nm1 + 1./2.*PHI_nm2)/dt
        DtPI = (3./2.*PI_grid - 2.0*PI_nm1 + 1./2.*PI_nm2)/dt
        

        #drA = utils.FD_second(x, A)
        #drPHI_grid = utils.FD_second(x, PHI_grid)
        #drPI_grid = utils.FD_second(x, PI_grid)
        drA = np.gradient(A, x)
        drPHI_grid = np.gradient(PHI_grid, x)
        drPI_grid = np.gradient(PI_grid, x)
 

        # for next time  
        timeval = (n+1)*dt
        print("CFL, t = ", max(Alp/A*C), timeval)
        ### Tridiagonal solver RHS
        
        d0[1:Nx] = 2.0*(rho_sq[1:Nx]*(u_n[1+1:Nx+1] + u_n[1-1:Nx-1]) \
                - 2.0*(rho_sq[1:Nx] - 2.0/C2)*u_n[1:Nx]) \
                + rho_sq[1:Nx]*(u_nm1[1+1:Nx+1] + u_nm1[1-1:Nx-1])\
                - 2.0*(rho_sq[1:Nx] + 2.0/C2)*u_nm1[1:Nx] \
                + 4.0*dx**2*rho_sq[1:Nx]*(1.0/(x[1:Nx]*dx)*(u_n[1+1:Nx+1] - u_n[1-1:Nx-1])) \
                +4.0*dx**2*(f(x[1:Nx], t[n]) + FLAG1*(Source1(A[1:Nx], Alp[1:Nx], PI_grid[1:Nx], Dtratio_AoverAlpha[1:Nx]) + Source2(A[1:Nx], Alp[1:Nx], PHI_grid[1:Nx], Dr_rho[1:Nx])) + FLAG2*G_Source(x[1:Nx], A[1:Nx], Alp[1:Nx], PHI_grid[1:Nx], PI_grid[1:Nx], drA[1:Nx], drPHI_grid[1:Nx], drPI_grid[1:Nx], g=1.0))

        #Bcd at r = 0 check it and fix scheme
        d0[0] =  2.0*(rho_sq[0]*(2.0*u_n[1]) - 2.0*(rho_sq[0] - 2.0/C2)*u_n[0]) \
                    + rho_sq[0]*(2.0*u_nm1[1]) - 2.0*(rho_sq[0] + 2.0/C2)*u_nm1[0]\
                    + 0.0 + 4.0* dx**2*(f(x[0], t[n]) + FLAG1*(Source1(A[0], Alp[0], PI_grid[0], Dtratio_AoverAlpha[0]) + Source2(A[0], Alp[0], PHI_grid[0], Dr_rho[0])) + FLAG2*G_Source(x[0], A[0], Alp[0], PHI_grid[0], PI_grid[0], drA[0], drPHI_grid[0], drPI_grid[0], g=1.0))
       ####This need to be verified carefully 
       #Bcd at r = Nx ot -1?check it
        d0[Nx] = rho_n[Nx]*(-dx/dt)*u_nm3[Nx] \
                + rho_n[Nx]*(2.0*dx/dt - 4.0*dx**2/(dt*x[Nx]))*u_nm2[Nx] \
                + 2.0*rho_sq[Nx]*u_nm1[Nx-1] - 2.0*(rho_sq[Nx] + 2.0/C2)*u_nm1[Nx] \
                + (6.0*rho_n[Nx]*dx/dt - 2.0*rho_sq[Nx]*dx/x[Nx] \
                + 16.0*rho_n[Nx]*dx**2/(x[Nx]*dt))*u_nm1[Nx] + 4.0*rho_sq[Nx]*u_n[Nx-1] - 4.0*(rho_sq[Nx] - 2.0/C2)*u_n[Nx] \
                + (-10.0*rho_n[Nx]*dx/dt -12.0*rho_n[Nx]*dx**2/(x[Nx]*dt) \
                - 8.0*rho_sq[Nx]*dx**2/((x[Nx])**2) - 4.0*rho_sq[Nx]*dx/x[Nx])*u_n[Nx] \
                + 4.0*dx**2*(f(x[Nx], t[n]) + FLAG1*(Source1(A[Nx], Alp[Nx], PI_grid[Nx], Dtratio_AoverAlpha[Nx]) + Source2(A[Nx], Alp[Nx], PHI_grid[Nx], Dr_rho[Nx])) + FLAG2*G_Source(x[Nx], A[Nx], Alp[Nx], PHI_grid[Nx], PI_grid[Nx], drA[Nx], drPHI_grid[Nx], drPI_grid[Nx], g=1.0) )

#######Tridiagonal Matrix
        a0 = -rho_sq      # since rho is already an array so a0 is an arry
        c0 = -rho_sq
        b0 =  2.0*rho_sq + np.ones_like(rho_sq)*4.0/C2
        #BCds here
        c0[0]  *= 2.0
        a0[-1] *= 2.0
        #fixed with adding velocity factor in Bcd at large r
        b0[Nx] +=  rho_sq[Nx]*(2.0*dx/x[Nx]) + rho_n[Nx]*(3.0*dx/dt)
        
        u[:] = utils.TDMAsolver(a0, b0, c0, d0)
                
        Saved_sqrt_sum_sq_u_at_t.append(sum_square_val_sqrt(u, dx, type='all'))

        if savedata==True:
            agrp.create_dataset('Ar_iter{0:04}'.format(n), data=A) 
            alpgrp.create_dataset('Alpr_iter{0:04}'.format(n), data=Alp) 
            ugrp.create_dataset('Ur_iter{0:04}'.format(n), data=u)  
            PHIgrp.create_dataset('PHI_iter{0:04}'.format(n), data=PHI_grid) 
            PIgrp.create_dataset('PI_iter{0:04}'.format(n), data=PI_grid) 

        #For Movie making
        if makemovie==True:
            filename='_temp%05d.png'%n
            draw_frame(x, u, PI_grid, PHI_grid, A, Alp, timeval, t[n+1])
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
        ratio_AoverAlpha_nm2[:] = ratio_AoverAlpha_nm1
        ratio_AoverAlpha_nm1[:] = ratio_AoverAlpha_n
        A_nm2[:] = A_nm1
        A_nm1[:] = A 
        PHI_nm2[:] = PHI_nm1
        PHI_nm1[:] = PHI_grid
        PI_nm2[:] = PI_nm1
        PI_nm1[:] = PI_grid
        Tarr.append(t[n])

        Tau_tr.append(max(abs(twist_tau_t_r(A, Alp, PHI_grid, PI_grid, drA, DtA, drPHI_grid, DtPHI, drPI_grid, DtPI))))
        Vplus, Vminus = Vpm(A, Alp, 1.0, PHI_grid, PI_grid)
        VpMax.append(max(abs(Vplus)))
        VmMax.append(max(abs(Vminus)))
        VpMin.append(min(Vplus))
        VmMin.append(min(Vminus))
        AlpMax.append(max(abs(Alp)))

    if savedata==True:
        outf.close()
    #for movie
    if makemovie==True:
        os.system("/usr/bin/ffmpeg  -framerate 30 -pattern_type glob -i '*.png' -c:v libx264 -pix_fmt yuv420p  new_laura_with_g_np_gradient_source_sigma15__movie_dudt_{0}_test80.mp4".format(dt))
        os.system("rm _temp*.png")
    plt.figure()
    plt.plot(Tarr, Tau_tr)
    plt.xlabel('r')
    plt.ylabel('tautr')
    plt.semilogy()
    plt.figure()
    plt.plot(Tarr, VpMax, label='max')
    plt.plot(Tarr, VpMin, label='min')
    plt.xlabel('r')
    plt.ylabel('V+')
    plt.legend()
    plt.ylim(0, 5)
    plt.figure()
    plt.plot(Tarr, VmMax, label='max')
    plt.plot(Tarr, VmMin, label='min')
    plt.xlabel('r')
    plt.ylabel('V-')
    plt.legend()
    plt.ylim(-5, 0)
    plt.figure()
    plt.plot(Tarr, AlpMax)
    plt.xlabel('r')
    plt.ylabel('max_lapse')
    plt.ylim(0.5, 5)
    plt.show()

    return u, x, t, np.asarray(Saved_sqrt_sum_sq_u_at_t)



### Error and Convergence. 
### Error at a fixed time for each grid point
dt1 = 0.05# dt and C are fixed
dx = 0.05

# for movie and saving data try
Totaltime = 65
L =100 #xmax
c = 1 #speed=> CFL = c dt/dx
solver(init, diff_init, source, c, L, dt1, dx, Totaltime, xmin=0, FLAG1=1.0, FLAG2=1.0, FLAG3=1.0, user_action=None, makemovie=True, savedata=True)
# For convergence
quit()
Total_time = int(60)
u1, x1, t1, P1 = solver(init, diff_init, source, 1, 100, dt1, dx, Total_time)#, FLAG1=1.0, FLAG2=1.0, FLAG3=1.0)
u2, x2, t2, P2 = solver(init, diff_init, source, 1, 100, dt1, dx/2.0, Total_time)#, FLAG1=1.0, FLAG2=1.0, FLAG3=1.0)
#
u4, x4, t4, P4 = solver(init, diff_init, source, 1, 100, dt1, dx/4.0, Total_time)#, FLAG1=1.0, FLAG2=1.0, FLAG3=1.0)
###
#### for dt = fixed
nc_t = np.zeros_like(t1)
for i in range(len(t1)-1):
    nc_t[i] = np.log2(np.abs(P1[i] - P2[i]) / np.abs(P2[i] - P4[i]))
plt.figure()
plt.title("Convergence dx^2")
plt.plot(t1, nc_t, '--bo')
plt.xlabel("t with fixed deltaT")
plt.ylabel("sqrt {Sum[u**2] *dx}")
plt.show()
#
####
print(utils.numerical_resolution_convergence(x1, u1, x2, u2, x4, u4, saveplot=False, savedata=False, totaltime=Total_time))
####
#quit()

dx = 0.5
dt1 = 0.25
#for dx = fixed and varying dt we need interpolated values for P1 and P2 
u1, x1, t1, P1 = solver(init, diff_init, source, 1, 100, dt1, dx, Total_time, FLAG1=1.0, FLAG2=1.0, FLAG3=1.0)
print(len(t1), len(P1))
u2, x2, t2, P2 = solver(init, diff_init, source, 1, 100, dt1/2.0, dx, Total_time, FLAG1=1.0, FLAG2=1.0, FLAG3=1.0)
u4, x4, t4, P4 = solver(init, diff_init, source, 1, 100, dt1/4.0, dx, Total_time, FLAG1=1.0, FLAG2=1.0, FLAG3=1.0)

P1interp = utils.get_intep_result(t1, P1, t4, order=4)
P2interp = utils.get_intep_result(t2, P2, t4, order=2)

nc_t = np.zeros_like(t4)
for i in range(len(t4)-1):
    nc_t[i] = np.log2(np.abs(P1interp[i] - P2interp[i]) / np.abs(P2interp[i] - P4[i]))
plt.figure()
plt.title("Convergence dt^2")
plt.plot(t4, nc_t)
plt.xlabel("t for finest resolution dt")
plt.ylabel("Log_2 sqrt {Sum[u**2] *dx}")
plt.show()
#also for final numerical convergence  now we cannot use numerical convergence we have due to different time step evolution although dx is fixed now 

print(utils.numerical_resolution_convergence(x1, u1, x2, u2, x4, u4, saveplot=True, savedata=False, totaltime=Total_time))


quit()

import scipy
from scipy.interpolate import lagrange, interp1d, splev, splrep

def get_intep_result(xval, yval, interpX, order=4):
    f_h_interp = splrep(xval, yval, k=order)
    interp_sol =  splev(interpX, f_h_interp)
    return interp_sol
### see plots and check what convergence to use
plt.figure()
plt.plot(x1, u1, 'k')
plt.plot(x2, u2, 'b--', alpha=0.8)
plt.plot(x4, u4, 'r-.', alpha=0.5)
#plt.show()

for i in range(len(x1)):
    print(x1[i], x4[i])
#if x1==x4:
#    print('same grid')
#u1in = get_intep_result(x1, u1, x4, order=4)
#u2in = get_intep_result(x2, u2, x4, order=4)
plt.figure()
plt.plot(x4, u1, 'ko')
plt.plot(x4, u2 , 'bo', alpha=0.8)
plt.plot(x4, u4, 'ro-', alpha=0.5)
#Interpolation of
plt.show()


############# Call this from utils
ah_interp = get_intep_result(x1, u1, x4, order=4)
ah2_interp =get_intep_result(x2, u2, x4, order=2)
a_diff_h2vsh = np.abs(ah2_interp - ah_interp)
a_diff_h4vsh2 = np.abs(u4 - ah2_interp)
#np.savetxt('diffdata4.txt', np.c_[x4, a_diff_h2vsh, a_diff_h4vsh2])
plt.figure()
#plt.ylabel(r'$\Delta \phi (r)$')
#plt.title('CFL ={0}, dt_h={1}'.format(CFL, dt1))
#plt.xlabel(r'$r$')
plt.plot(x4, a_diff_h2vsh, label='hvsh/2')
plt.plot(x4, a_diff_h4vsh2, label='h/2vsh/4', alpha=0.5)
plt.plot(x4, 4.0*a_diff_h4vsh2, ls='--', label='4* h/2vsh/4')
plt.semilogy()
plt.legend()
plt.show()
#plt.figure()
#plt.title('CFL {0}, dt_1 {1}'.format(CFL, dt1))
#plt.ylabel(r'$\Delta \phi (r)$')
#plt.xlabel(r'$r$')
#plt.plot(x4, a_diff_h2vsh, label='hvsh/2')
#plt.plot(x4, 4.0*a_diff_h4vsh2, label='4* h/2vsh/4')
#plt.semilogy()
#plt.legend()
#plt.show()
#print(dt1)
