# A generic RK-4 scheme for constriants eqs

import numpy as np
import matplotlib.pyplot as plt

#g = 0.0 case
def f1(R, r, Phi, Pi, g=0.0):
    Aaa = R
    drAaa = Aaa/(8.0*r)*(4.0*(1.0 - Aaa**2) + r**2 *(Phi**2 + Pi**2))\
            + g*r/(16.0*Aaa)*((Phi**2 + Pi**2)**2 - 4.0*Pi**4)
    return drAaa

def f2(R, r, Phi, Pi, Aaa, g=0.0):
    alp = R
    dralp = alp/(8.0*r)*(4.0*(Aaa**2 -  1.0) + r**2 *(Phi**2 + Pi**2))\
            - g*r*alp/(16.0*Aaa**2)*((Phi**2 + Pi**2)**2 - 4.0*Phi**4)
    return dralp

def Initial_func(r, r0=20.0, sigma0=5.0):
    """
    gievn that r is a grid [regular or staggered]
    get the initial functions
    """ 
    for i in range(len(r)):
        PI[i] = 0.0
        PHI[i] = np.exp(-0.5* (r[i] -  r0)**2/sigma0**2) *np.cos(np.pi/10*r[i])
    return PI, PHI

def Initial_conditions(rval, Phi):
    """ 
    Based on analytic approx 
    of expansion of drA and drAlpha
    equations
    Taylor expanded soln upto o(r^4)
    term
    """
    a_init = 1.0 + (rval * Phi)**2/24.0
    alpha_init = 1.0 + (rval * Phi)**2/12.0
    return a_init, alpha_init

def get_r_grid(dr, rmin, rmax):
    Nx = int(round((rmax-rmin)/dr))
    grid_vals = np.zeros(Nx+1) #+2 if -
    for i in range(Nx+1):
        grid_vals[i] = (i + 0.5*dr)
    return grid_vals

#### Crucial RK-4 scheme with numeric data
def standard_rk4(dx, rhs, rmin=0, rmax=100):
    #create a staggered grid
    Nx = int(round((rmax-rmin)/dx))
    x_grid = np.zeros(Nx+2)
    for i in range(Nx+2):
        x_grid[i] = (i - 0.5)*dx
    a_val = initial_condition(x_grid[0])
    print("initial value =", a_val)
    Aarr = [a_val]
    for i in range(1, len(x_grid)):
        r = x_grid[i]
        Aarr.append(a_val)
        k1 = dx*rhs(a_val, r)
        k2 = dx*rhs(a_val + 1.0/2.0*k1, r + 1.0/2.0*dx)
        k3 = dx*rhs(a_val + 1.0/2.0*k2, r + 1.0/2.0*dx)
        k4 = dx*rhs(a_val + 1.0*k3, r + 1.0*dx)
        a_val += (k1 + 2.0*k2  + 2.0*k3 + k4)/6.0

    return np.array(x_grid), np.array(Aarr)

### to be tested
def rk4_scheme(r_grid, init_a, init_alp, rhs_a, rhs_alp):
    """
    #used prepared interpolation for Phi and Pi
    #in equations
    given r_grid [regular staggered]
    with initial values 
    get RK4 integral of d_a/dr and d_alp/dr
    """
    f_Phi_interp = interpolate.interp1d(r, Phi, kind='cubic', fill_value="extrapolate")
    f_Pi_interp = interpolate.interp1d(r, Pi, kind='cubic', fill_value="extrapolate")
    alpPts = []
    AaaPts = []
    #alpha and A are zero at r=0 as regularity
    R = 1.0+(r[0]*Phi[0])**2/12.0 #init_a
    for i in range(len(r)):
        AaaPts.append(R)
        k1 = dr*f1(R, r[i], Phi[i], Pi[i])
        # we need Phi not at i but Phi(r+0.5*dr) for that we will need 
        #iterpolated Phi 
        k2 = dr*f1(R+0.5*k1, r[i]+0.5*dr, f_Phi_interp(r[i]+0.5*dr), f_Pi_interp(r[i]+0.5*dr))
        k3 = dr*f1(R+0.5*k2, r[i]+0.5*dr, f_Phi_interp(r[i]+0.5*dr), f_Pi_interp(r[i]+0.5*dr))
        k4 = dr*f1(R+k3, r[i]+dr, f_Phi_interp(r[i]+dr), f_Pi_interp(r[i]+ dr))
        R += (k1 + 2.0*k2 + 2.0*k3 + k4)/6.0
        print(i, R)
    ### we will also need Aaa interpolated points 
    ### here I think we can use linear interpolation?
    f_Aaa_interp = interpolate.interp1d(r, AaaPts, kind='cubic', fill_value="extrapolate")
    #Now compute lapse with backward loop
    init_alpha = 1.0/AaaPts[-1] #1.a[r_max]
    R = init_alpha
    for i in range(len(r)-1, -1, -1):
        alpPts.append(R)
        k1 = dr*f2(R, r[i], Phi[i], Pi[i], AaaPts[i])
        k2 = dr*f2(R+0.5*k1, r[i]-0.5*dr, f_Phi_interp(r[i]-0.5*dr), f_Pi_interp(r[i]-0.5*dr), f_Aaa_interp(r[i]-0.5*dr))
        k3 = dr*f2(R+0.5*k2, r[i]-0.5*dr, f_Phi_interp(r[i]-0.5*dr), f_Pi_interp(r[i]-0.5*dr), f_Aaa_interp(r[i]-0.5*dr))
        k4 = dr*f2(R+k3, r[i]-dr, f_Phi_interp(r[i]-dr), f_Pi_interp(r[i]-dr), f_Aaa_interp(r[i]-dr))
        R += (k1 + 2.0*k2 + 2.0*k3 + k4)/6.0
    #alpPts are backward integrated so we need alpPts[::-1] order

    if make_plot==True:
        plt.plot(r, alpPts[::-1]/alpPts[-1], label ='lapse')
        plt.plot(r, AaaPts, label='a')
        #plt.plot(r, exact_sol, label='analytic')
        plt.xlabel('r')
        plt.ylabel(r'$a, \alpha$ at t={0:.3f}'.format(time))
        plt.legend()
        plt.show()

    return r_grid, AaaPts, alpPts


import scipy
from scipy.interpolate import splev, splrep
def interpolate(required_grid, input_x, input_y, order=4):
    prepare_interp = splrep(input_x, input_y, k=order)
    return splev(required_grid, prepare_interp)

def check_numerical_convergence(dr):
    """
    given dr 
    compute rk4 solution 
    for a and alpha 
    at dr, dr/2, dr/4
    and compute numerical
    convergence point by point
    with appropriate interpolation
    due to staggered grid
    """
    #usind dr get r_grid
    r_grid = get_r_grid(dr, rmin=0.0, rmax=100.0)
    r_grid2 = get_r_grid(dr/2.0, rmin=0.0, rmax=100.0)
    r_grid4 = get_r_grid(dr/4.0, rmin=0.0, rmax=100.0)
    r_grid_dr, A_rk_sol_dr, Alp_rk_sol_dr = rk4_scheme(r_grid, init_a, init_alp)
    r_grid_dr2, A_rk_sol_dr2, Alp_rk_sol_dr2 = rk4_scheme(r_grid2, init_a, init_alp)
    r_grid_dr4, A_rk_sol_dr4, Alp_rk_sol_dr4 = rk4_scheme(r_grid4, init_a, init_alp)
    #4th order spline interpolation
    interp_A_dr = interpolate(r_grid_dr4, r_grid_dr, A_rk_sol_dr, order=4)
    interp_A_dr2 = interpolate(r_grid_dr4, r_grid_dr2, A_rk_sol_dr2, order=4)

    interp_Alp_dr = interpolate(r_grid_dr4, r_grid_dr, Alp_rk_sol_dr, order=4)
    interp_Alp_dr2 = interpolate(r_grid_dr4, r_grid_dr2, Alp_rk_sol_dr2, order=4)

    #Error_dr_dr2
    Err_Adr =  np.abs(interp_A_dr - interp_A_dr2)
    Err_Adr2 =  np.abs(interp_A_dr2 - A_rk_sol_dr4)
    #Error_dr2_dr4

    #plots
    plt.figure(figsize=(8, 7))
    plt.plot(r_grid_dr4, Err_Adr,'k', lw=3, label='diff[dr vs dr/2]')
    plt.plot(r_grid_dr4, Err_Adr2, 'r', label='diff[dr/2 vs dr/4]')
    plt.plot(r_grid_dr4, 2**4*Err_Adr2, color= 'yellow',ls='--', label='16* diff[dr/2 vs dr/4]')
    plt.legend()
    plt.xlabel('r')
    plt.ylabel('Abs(Error)')
    plt.show()
    plt.close()
    return 0
