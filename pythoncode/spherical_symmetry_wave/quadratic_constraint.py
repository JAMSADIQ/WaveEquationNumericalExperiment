import numpy as np
import matplotlib.pyplot as plt
from utils_interpolation import *

gval =  1.0

def Fa(x, A, interpPHI, interpPI, g=gval, b=0.0):
    """
    given Interpolator of PHI/PI function get rhs of
    a eq at any x grid point """    

    gterm = x/(16.0*A)*((interpPHI**2 +  interpPI**2)**2 - 4.0*interpPI**4)
    bterm = x/32.0/A**3*((interpPI**2 - interpPHI**2)**2 + (interpPHI**2 + 5.0*interpPI**2))
    return A/8.0/x *((4.0*(1.0 - A**2)) + x**2 *(interpPHI**2 +  interpPI**2)) + g*gterm + b*bterm


def Falpha(x, Alp, interpA, interpPHI, interpPI, g=gval, b=0.0):
    gterm = x*Alp/(16.0*interpA**2)*((interpPHI**2 +  interpPI**2)**2 - 4.0*interpPHI**4)
    bterm = x*Alp/32.0/interpA**4 *((interpPI**2 - interpPHI**2)**2 + (5.0*interpPHI**2 + interpPI**2))
    return Alp/8.0/x *(4.0*(interpA**2 - 1.0) + x**2*(interpPHI**2 +  interpPI**2)) - g*gterm


#STEP I we must be given a grid, PHI and PI on that grid so compute init_a 
def constraint_solver(x_grid, PHI_grid, PI_grid, ratio=False):
    """
    x_grid is an staggered grid (i + 1/2)*dx
    given x, PHI, PI
    get a and alpha
    """
    #get dx  
    dx = x_grid[1] - x_grid[0]
    p0 = 0.0  #PHI[0] #dphi/dr=0 regularity
    ### dPHI/dr #note here big PHI or its d^2 phi/dr^2
    p1 = (-3.0*PHI_grid[0] + 4.0*PHI_grid[1] - PHI_grid[2])/(2.0*dx)#use dPHI/dr finitie diff (O(dr^2)) formula
    p2 = (2.0*PHI_grid[0] - 5.0*PHI_grid[1] + 4.0*PHI_grid[2] - PHI_grid[3])/(dx**3)  #used second derivative of PHI fd formula
    # same for PI[r]
    q0 = PI_grid[0]
    q1 = (-3.0*PI_grid[0] + 4.0*PI_grid[1] - PI_grid[2])/(2.0*dx)
    q2 = (2.0*PI_grid[0] - 5.0*PI_grid[1] + 4.0*PI_grid[2] - PI_grid[3])/(dx**3)

    #compute initial a and alpha at first grid point
    #init_A = 1.0 + (x_grid[0]**2*(PHI_grid[0]**2 + PI_grid[0]**2))/24.0
    #init_Alp = 1.0 + (x_grid[0]**2*(PHI_grid[0]**2 + PI_grid[0]**2))/12.0
    init_A = 1.0 + x_grid[0]**2*(p0**2 + q0**2)/24.0 + x_grid[0]**3*(p0*p1 + q0*q1)/16.0 + x_grid[0]**4*(p0**4 + 96.0*p0*p2 + 2.0*p0**2*q0**2 + q0**4 + 48.0*(p1**2 + q1**2) + 96.0*q0*q2)/1920.0 
    init_Alp = 1.0 + x_grid[0]**2*(p0**2 + q0**2)/12.0 + 1./2880.*(300.0*(p0*p1 + q0*q1)*x_grid[0]**3 + (11.0*p0**4 + 216.0*p0*p2 + 22*p0**2*q0**2 + 11.0*q0**4 + 108.0*(p1**2 + q1**2) + 216*q0*q2)*x_grid[0]**4)


    Apoints = [init_A]  #before start of domain ghost
    aval = init_A
    for i in range(1, len(x_grid)): # note we are starting with 1 not 0 index 
        x = x_grid[i]
        Apoints.append(aval) #symmetric if we use -dr/2 as first grid point
        Interpolator_PHI1 = local_interp(x_grid, PHI_grid, x, order=4)
        Interpolator_PI1 = local_interp(x_grid, PI_grid, x, order=4)
        k1 = dx*Fa(x, aval,  Interpolator_PHI1, Interpolator_PI1)

        Interpolator_PHI2 = local_interp(x_grid, PHI_grid, x+1.0/2.0*dx, order=4)
        Interpolator_PI2 = local_interp(x_grid, PI_grid, x+1.0/2.0*dx, order=4)
        k2 = dx*Fa(x+1.0/2.0*dx, aval+1.0/2.0*k1, Interpolator_PHI2, Interpolator_PI2)
        k3 = dx*Fa(x+1.0/2.0*dx, aval+1.0/2.0*k2, Interpolator_PHI2, Interpolator_PI2)
        Interpolator_PHI3 = local_interp(x_grid, PHI_grid, x+dx, order=4)
        Interpolator_PI3 = local_interp(x_grid, PI_grid, x+dx, order=4)
        k4 = dx*Fa(x + dx, aval+ k3, Interpolator_PHI3, Interpolator_PI3)
        aval +=  (k1 + 2.0*k2 + 2.0*k3 + k4)/6.0
        #Apoints.append(aval) #if we use dr/2 as first grid point uncomment this and commone above
    
    ## Now integrate Alpha with interpolation of A vals
    A_grid = np.array(Apoints)
    Alppoints = [init_Alp]  # before start of domain ghost
    alpval = init_Alp
    for i in range(1, len(x_grid)):
        x = x_grid[i]
        Alppoints.append(alpval) #symmetric
        Interpolator_A1 = local_interp(x_grid, A_grid, x, order=4)
        Interpolator_PHI1 = local_interp(x_grid, PHI_grid, x, order=4)
        Interpolator_PI1 = local_interp(x_grid, PI_grid, x, order=4)
        k1 = dx*Falpha(x, alpval,  Interpolator_A1, Interpolator_PHI1, Interpolator_PI1)


        Interpolator_A2 = local_interp(x_grid, A_grid, x+1.0/2.0*dx, order=4)
        Interpolator_PHI2 = local_interp(x_grid, PHI_grid, x+1.0/2.0*dx, order=4)
        Interpolator_PI2 = local_interp(x_grid, PI_grid, x+1.0/2.0*dx, order=4)
        k2 = dx*Falpha(x+1.0/2.0*dx, alpval+1.0/2.0*k1, Interpolator_A2, Interpolator_PHI2, Interpolator_PI2)
        k3 = dx*Falpha(x+1.0/2.0*dx, alpval+1.0/2.0*k2, Interpolator_A2, Interpolator_PHI2, Interpolator_PI2)

        Interpolator_A3 = local_interp(x_grid, A_grid, x+dx, order=4)
        Interpolator_PHI3 = local_interp(x_grid, PHI_grid, x+dx, order=4)
        Interpolator_PI3 = local_interp(x_grid, PI_grid, x+dx, order=4)
        k4 = dx*Falpha(x + dx, alpval+ k3, Interpolator_A3, Interpolator_PHI3, Interpolator_PI3)
        alpval +=  (k1 + 2.0*k2 + 2.0*k3 + k4)/6.0
        #Alppoints.append(alpval) #last grid point
    Alp_grid = np.array(Alppoints)
    Alp_grid[-1] = 1.0/A_grid[-1]

    ratio_Alpha_by_A = Alp_grid/A_grid
    if ratio==True:
        return x_grid, A_grid, Alp_grid, ratio_Alpha_by_A
    return x_grid, A_grid, Alp_grid


# Test Constriant Convergence if 4th oder or not
#Initial PHI and PI given x_grid
def dr_phi_dt_phi(r_grid, r0=55.0, Case='B'):
    if Case == 'A':
        print('Case A')
        a0 = 0.02; sig0 =15.0
    elif Case == 'B':
        print('Case B')
        a0 = 0.14; sig0 = 1.5
    elif  Case == 'C':
        print('Case = C')
        a0 = 0.045; sig0 = 15.0
    else:
        print('check if you use wrong case')
    phigrid = np.zeros_like(r_grid)
    pigrid = np.zeros_like(r_grid)
    for i in range(len(r_grid)):
        phigrid[i] = a0*np.exp(-(r_grid[i] - r0)**2/sig0**2)*np.cos(np.pi/10*r_grid[i])
    return phigrid, pigrid



def get_solution(dx, x0=0.0, xmax=100.0, testcase='C'):
    """
    given dx  compute xgrid and rk4 results
    on that grid
    """
    Nx = int((xmax - x0)/dx)
    X_grid = np.zeros(Nx+1)
    for i in range(Nx+1):
        X_grid[i] = (i + 0.5)*dx
    PHI_grid, PI_grid = dr_phi_dt_phi(X_grid, Case=testcase)
    Xarr, Aarr, Alparr = constraint_solver(X_grid, PHI_grid, PI_grid, ratio=False)
    return  Xarr, Aarr, Alparr



def get_intep_result(xval, yval, interpX, order=4):
    import scipy 
    from scipy.interpolate import lagrange, interp1d, splev, splrep
    f_h_interp = splrep(xval, yval, k=order)
    interp_sol =  splev(interpX, f_h_interp)
    return interp_sol


def get_convergence(dxval):
    """
    get a and alpha at dx, dx/2, and dx/4
    """
    th, ah, alph = get_solution(dxval)
    th2, ah2, alph2 = get_solution(dxval/2.0)
    th4, ah4, alph4 = get_solution(dxval/4.0)
    plt.figure()
    plt.plot(th4, ah4, label='a_4')
    plt.plot(th4, alph4, label='alpha_4')

    plt.title('g={0}'.format(gval))
    plt.legend()
    # interpolate h, h2 onto h4 grid
    ah_interp = get_intep_result(th, ah, th4, order=4)
    ah2_interp = get_intep_result(th2, ah2, th4, order=4)
    alph_interp = get_intep_result(th, alph, th4, order=4)
    alph2_interp = get_intep_result(th2, alph2, th4, order=4)
    a_diff_h2vsh = np.abs(ah2_interp - ah_interp)
    a_diff_h4vsh2 = np.abs(ah4 - ah2_interp)
    alp_diff_h2vsh = np.abs(alph2_interp - alph_interp)
    alp_diff_h4vsh2 = np.abs(alph4 - alph2_interp)
    plt.figure()
    plt.plot(th4, a_diff_h2vsh, label='|r-r2|')
    plt.plot(th4, 16.0*a_diff_h4vsh2, label='16* r2-r4')
    #plt.plot(th4, 8.0*a_diff_h4vsh2, 'k--', label='8* r2-r4')
    plt.title(r'$a(r),\, g={0}$'.format(gval), fontsize=14)
    plt.xlabel('r', fontsize=14)
    plt.legend(fontsize=14)
    plt.figure()
    plt.plot(th4, alp_diff_h2vsh, label='r-r2')
    plt.plot(th4, 16.0*alp_diff_h4vsh2, label='16* r2-r4')
    #plt.plot(th4, 8.0*alp_diff_h4vsh2, 'k--' , label='8* r2-r4')
    plt.title(r'$\alpha (r) \, g={0}$'.format(gval), fontsize=14)
    plt.xlabel('r', fontsize=14)
    plt.legend(fontsize=14)
    plt.figure()
    plt.plot(th4, a_diff_h2vsh, label='r-r2')
    plt.plot(th4, 16.0*a_diff_h4vsh2, label='16* r2-r4')
    #plt.plot(th4, 8.0*a_diff_h4vsh2, 'k--', label='8* r2-r4')
    plt.title(r'$a(r)$', fontsize=14)
    plt.xlabel('r', fontsize=14)
    plt.semilogy()
    plt.legend(fontsize=14)
    plt.figure()
    plt.plot(th4, alp_diff_h2vsh, ls='--', label='r-r2')
    plt.plot(th4, 16.0*alp_diff_h4vsh2, label='16* r2-r4')
    #plt.plot(th4, 8.0*alp_diff_h4vsh2, 'k--' , label='8* r2-r4')
    plt.title(r'$\alpha(r)$', fontsize=14)
    plt.xlabel('r', fontsize=14)
    plt.semilogy()
    plt.legend(fontsize=14)
    plt.show()
    return th4, a_diff_h2vsh, a_diff_h4vsh2,  alp_diff_h2vsh, alp_diff_h4vsh2

#get_convergence(0.1)
