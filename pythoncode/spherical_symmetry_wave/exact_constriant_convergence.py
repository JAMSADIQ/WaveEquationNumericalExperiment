# From Enrico's mathematica notebook
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.interpolate import splev, splrep

def exact_sol(t):
    return np.sin(t)/t

#rhs of eq
def f(x, t):
    return -x + (t*np.cos(t) + (-1 + t)*np.sin(t))/t**2

def initial_condition(t):
    return 1.0 - t**2/6.0

# we have grid [dx/2, 3dx/2, ...]  # -dx/2, dx/2 is not working 
def get_grid_init_val(dx, x0=0.0, xmax=10.0):
    """
    return initial value at first point of grid
    and grid itself
    """
    #which have same value as first point in staggered grid
    Nx = int(round((xmax-x0)/dx))
    t_grid = np.zeros(Nx+2)
    for i in range(Nx+2):
        t_grid[i] = (i - 0.5)*dx
    xi = initial_condition(t_grid[0]) #1.0 - t_grid[0]**6/6.0  #np.cos(t_grid[0])
    return xi, t_grid


def RK4(t_grid, init_x, dx=0.01):
    xpoints = [init_x]  # before start of domain ghost
    x = init_x
    for i in range(1, len(t_grid)):
        t = t_grid[i] 
        xpoints.append(x) # symmetric
        k1 = dx*f(x, t)
        k2 = dx*f(x+1.0/2.0*k1, t+1.0/2.0*dx)
        k3 = dx*f(x+1.0/2.0*k2, t+1.0/2.0*dx)
        k4 = dx*f(x+k3, t+dx)
        x +=  (k1 + 2.0*k2 + 2.0*k3 + k4)/6.0 
    return t_grid, xpoints

### Checxk RK-4 convergence
x0 = 0
xmax = 10.0     #will try till 100 as well to see
dxval= 0.1/2.0     #change this for convergence
#create a grid and check
xinit, tgrid_h = get_grid_init_val(dxval, x0=x0, xmax=xmax)
#print(xinit, tgrid_h)
# get RK4 results 
tgrid_h, rk4_h = RK4(tgrid_h, xinit, dx=dxval)

## We want to check convergence I will use dx/2 and dx/4 grids and we compute convergence on most finer grids
xinit2, tgrid_h2 = get_grid_init_val(dxval/2.0, x0=x0, xmax=xmax)
tgrid_h2, rk4_h2 = RK4(tgrid_h2, xinit2, dx= dxval/2.0)

#create interpolater for h and h/2 maybe also h/4?
f_h_interp = splrep(tgrid_h, rk4_h, k=4)
interp_rk4_h =  splev(tgrid_h2, f_h_interp)


#plot exact sol with dx and dx/2 results
plt.figure(figsize=(8, 7))
plt.plot(tgrid_h2, exact_sol(tgrid_h2),'o-', label='exact')
plt.plot(tgrid_h2, interp_rk4_h, '^-', label='interp-rk4-dx')
plt.plot(tgrid_h2, rk4_h2, '*-', label='rk4-dx2')
plt.legend()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()

# Now lets compute the error of exact with dx and dx/2 based numeric and compute convergence
exact_h2 = exact_sol(tgrid_h2)
diffinterp_h = np.abs(exact_h2 - interp_rk4_h)
diff_h2 = np.abs(exact_h2 - rk4_h2)
#plot point by point to check convergence
plt.figure(figsize=(8, 7))
plt.plot(tgrid_h2, diffinterp_h,'k', label='err_interp-dx')
plt.plot(tgrid_h2, 16.0*diff_h2, 'r--', label='16* err_dx/2')
plt.legend()
plt.xlabel('x')
plt.ylabel('abs|error|')
plt.semilogy()
plt.show()
