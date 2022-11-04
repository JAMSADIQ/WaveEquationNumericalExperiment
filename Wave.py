import numpy as np
import matplotlib.pyplot as plt

# we want to solve u_tt = c^2 u_xx 
#Initial conditons: u(x,0) = sin(n*pi*x/L), u_t(x,0)=0
#Boundary conditions u(0,t) = u(L,t) = 0 for t>=0

#ICnds functions
def f(x, L=1):
    return np.sin(4*np.pi*x/L)

def g(x):
    return 0.0*x

#domains x: [0, L=1], t: [0,T*Nt], T=periods =10, Nt= T/dt
#stepsize: dt=0.0025, c=1, CFL=cdt/dx=1/4=0.25<1 to get dx = cdt/CFL

L = 1.0 
T = 10 
dt = 0.0025
Nt = int(round(T/dt))
t = np.linspace(0, Nt*dt, Nt+1)
c = 1.0
coeff = 1.001 #<1
dx = dt*c/coeff     #coeff=CFL condition (test it to see stability)
Nx = int(round(L/dx))
x = np.linspace(0, L, Nx+1)

#Explicit Scheme u[t=n+1] = f(u[t=n], u[t=n-1])
u = np.zeros(Nx+1)
u_n = np.zeros(Nx+1) #1 step back value
u_nm1 = np.zeros(Nx+1) #2 steps back value 

#Initial condition 1: u[x, t= 0] = f[x]
for i in range(0, Nx+1):
    u_n[i] = f(x[i], L=1.0)

#check with plot
plt.figure()
plt.plot(x, u_n, 'r--', lw=2)

# Formula at t = 0, see notes 
n = 0
#inside domain [x: i==1 to i=Nx-1]
for i in range(1, Nx):
    u[i] = dt*g(x[i]) + (1 - coeff**2)*u_n[i] + 0.5*coeff**2*(u_n[i-1] + u_n[i+1])
#Bounday values u(x=0,T) = 0.0 = u(x=L,T)
u[0] = 0.0
u[Nx] = 0.0
#Now we switch variable before next time step
u_nm1[:] = u_n
u_n[:] = u
#for n in range(1, 30):
for n in range(1, Nt):
    #for each xi grid values
    for i in range(1, Nx):
        u[i] = -u_nm1[i] + 2*(1 - coeff**2)*u_n[i]+coeff**2*(u_n[i+1] + u_n[i-1])
    #boundary values
    u[0] = 0
    u[Nx] = 0
    #switch variables
    u_nm1[:] = u_n
    u_n[:] = u
    plt.plot(x, u, label='time={0}'.format(n))

plt.xlabel('x')
plt.ylabel('u')
plt.show()    
#plot c, t, u it will be threeD plot

#Impicit Scheme:
def TDMAsolver(a, b, c, d):
    '''
    TDMA solver, a b c d can be NumPy array type or Python list type.
    refer to http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
    and to http://www.cfd-online.com/Wiki/Tridiagonal_matrix_algorithm_-_TDMA_(Thomas_algorithm)
    '''
    nf = len(d) # number of equations
    ac, bc, cc, dc = map(np.array, (a, b, c, d)) # copy arrays
    for it in range(1, nf):
        mc = ac[it-1]/bc[it-1]
        bc[it] = bc[it] - mc*cc[it-1]
        dc[it] = dc[it] - mc*dc[it-1]

    xc = bc
    xc[-1] = dc[-1]/bc[-1]

    for il in range(nf-2, -1, -1):
        xc[il] = (dc[il]-cc[il]*xc[il+1])/bc[il]

    return xc

#Implicit scheme as in notes
u = np.zeros(Nx+1)
u_n = np.zeros(Nx+1) #1 step back value
u_nm1 = np.zeros(Nx+1) #2 steps back value
#initially
n = 0
for i in range(1, Nx):
    u[i] = f(x[i])
#Bcnds
u[0] = 0
u[Nx] = 0
plt.figure()
plt.plot(x, u, 'r--', lw=2)
n=1

a = np.ones(Nx)*(-0.5*coeff**2)
b = np.ones(Nx+1)*(1.0 +coeff**2)
c = np.ones(Nx)*(-0.5*coeff**2)
d = np.zeros(Nx+1)
for i in range(1, Nx, 1):
    d[i] = f(x[i])+ dt * g(x[i])

u_n = TDMAsolver(a, b, c, d)
u_n[0] = 0
u_n[Nx] = 0 
plt.plot(x, u_n)
#replace steps n-1==>n, n==>n+1
u_nm1[:] = u_n
u_n[:] = u
#evolve for Nt time
for n in range(1, Nt):
    #for each xi grid values
    d = np.zeros(Nx+1)
    u = np.zeros(Nx+1)
    for i in range(1,Nx, 1):
        d[i] = 2*u_n[i] - u_nm1[i]
        u = TDMAsolver(a, b, c, d)
    u_nm1[:] = u_n
    u_n[:] = u
    plt.plot(x, u)
plt.show()
