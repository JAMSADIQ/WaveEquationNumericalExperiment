#simple vib string solution numerical explicit scheme
import numpy as np
import matplotlib.pyplot as plt

Lval = 1.0
c = 1.0  #wavephase
def exact_sol(x, t, L=Lval, Mm=2):
    return np.cos(Mm*np.pi/L*t)*np.sin(Mm*np.pi/L*x)


def I(x):
    return exact_sol(x, 0)

def V(x, L=Lval):
    return 0.0

def f(x, t):
    return 0.0 

def explicit_scheme(t, xmax=Lval, CFLval=0.9):
    dt = t[1] - t[0]
    #for convergence test try fix cfl
    CFL = CFLval
    dx = c*dt/CFL
    x = np.arange(0, xmax, dx)
    #CFL = c*dt/dx
    print("dt, dx, Nxpoints =", dt, dx, len(x))
    #quit()
    Nt = len(t) -1
    Nx = len(x) -1
    sqCFL = CFL**2
    u_n = np.zeros_like(x) 
    u_nm1 = np.zeros_like(x) 
    u = np.zeros_like(x) 
    #set initial condition
    for i in range(0, Nx+1):
        u_n[i] = I(x[i])

    #First step scheme
    n = 0 # need fot t[0] from tarray = 0
    for i in range(1, Nx):
        u[i] = u_n[i] + dt * V(x[i]) + 0.5*sqCFL*(u_n[i+1] - 2*u_n[i] + u_n[i-1]) + 0.5*dt**2*f(x[i], t[n])
        #u[i] = u_n[i] + 0.5*sqCFL*(u_n[i+1] - 2*u_n[i] + u_n[i-1])
    #Boundary conditions
    u[0] = 0
    u[Nx] = 0
    #switch varibales before next step
    u_nm1[:] = u_n
    u_n[:] = u
    #Now we use scheme at 2nd and next steps
    for n in range(1, Nt):
        for i in range(1, Nx):
            u[i] = 2*u_n[i] - u_nm1[i] + sqCFL*(u_n[i+1] - 2*u_n[i] + u_n[i-1]) + dt**2*f(x[i], t[n])
        #Bcds
        u[0] = 0
        u[Nx] = 0
        #switch variables
        u_nm1[:]= u_n
        u_n[:] = u
        #if n%10==0:
            #exact_t = exact_sol(x, t[n+1])
            #diff = np.abs(u - exact_t).max()
    
    exact_t1 = exact_sol(x, t[Nt-1]) 
    exact_t2 = exact_sol(x, t[Nt]) 
    diff1 = np.abs(u - exact_t1).max()
    diff2 = np.abs(u - exact_t2).max()
    plt.figure()
    plt.plot(x, u, label='numeric_at_t={0:.2f}'.format(n*dt))
    plt.plot(x, exact_t1, label='exact_t_{0:.2f}'.format(t[n]))
    plt.title('error = {0:.3e}, {1:.3e}'.format(diff1, diff2))
    plt.legend()
    plt.show()
    
    return dt , diff2

dtlist = []
difflist =[]

#test convergence
dt0 = 0.1
for i in range(7):
    #t = np.linspace(0, 10, npoints)
    t = np.arange(0, 1+dt0, dt0)
    ndt, ndiff =  explicit_scheme(t, xmax=Lval, CFLval=0.9) 
    print("dt, diff = ", ndt, ndiff)
    dtlist.append(ndt)
    difflist.append(ndiff)
    dt0 = dt0/2.0

for i in range(1, len(dtlist), 1):
    r = np.log(difflist[i]/difflist[i-1])/np.log(dtlist[i]/dtlist[i-1])
    print(dtlist[i], round(r, 2))


