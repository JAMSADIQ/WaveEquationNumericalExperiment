# Module which upon call will provide an phi values at x_grid

import numpy as np

#Laura based Initial data will need RK-integrationi

def rk4_init(r_min, r_max, r_grid, f):
    """
    f is dphi/dr function at r_grid 
    one need to provide initial phi(r=rmin)
    start of integration to integrate
    """
    a = r_min
    b = r_max
    dr = r_grid[1] - r_grid[0]
    #print("r_grid = ", r_grid)
    #high resolution result
    h = dr/2.0

    tpoints = np.arange(a, b+4*h, h)
    xpoints = []
    x = 0.0 # initial value of dphi/dr = 0
    # note we have staggered grid so we may need 
    #corrected values at x=dr/2 using Taylor expansion
    for t in tpoints:
        xpoints.append(x)
        k1 = h*f(x,t)
        k2 = h*f(x+0.5*k1,t+0.5*h)
        k3 = h*f(x+0.5*k2,t+0.5*h)
        k4 = h*f(x+k3,t+h)
        x += (k1+2*k2+2*k3+k4)/6.0

    # get values at x_grid via 
    #interpolation 
    import scipy
    from scipy import interpolate
    finterp = interpolate.interp1d(tpoints, xpoints)
    phi_data_on_staggered_grid =  finterp(r_grid)
    return phi_data_on_staggered_grid


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

def miguels_phi(r, a0=0.01, r0=55.0, sigma0=2.0):
    """
    Miguels initial data 
    """
    return a0*np.exp(-0.5*(r-r0)**2/sigma0**2)*np.sin((r-r0)/sigma0/np.sqrt(2.0))
