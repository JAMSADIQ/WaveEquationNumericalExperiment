#Implicit scheme check convergence rate
#need 1: tridiagonal solver for initial guess than use in iterative scheme
#or try inverse matrix 
# First order scheme verses 2nd order scheme
import numpy as np
import matplotlib.pyplot as plt
import utils_implicit as utils

def exact_sol(x, t, L=2.5, c=1.5, A=1):
    #return A*np.sin(4*np.pi*x/L)*np.cos(np.pi*c*t/L)
    #return x*(L - x)*np.sin(t)
    return x*(L - x)*(1 + 0.5*t)

c = 1.5

def I(x):
    #exact solution at t = 0
    return exact_sol(x, 0)

def V(x):
    return 0.5*exact_sol(x, 0)

def f(x, t):
    return 2*(1 + 0.5*t)*c**2


def implicit(x, t):
    dx = x[1] - x[0] 
    dt = t[1] - t[0]
    Nx = len(x) - 1
    Nt = len(t) - 1
    c = 1.5
    CFL = c*dt/dx
    print("dx, dt, cfl =", dx, dt, CFL)
    C2  =  CFL**2 # (cdt/dx)^2
    size_of_a_matrix = len(x)
    diagonalVal = 2
    diagonalBelowVal = -1
    diagonalAboveVal = -1
    Kmatrix = utils.TridiagonalMatrix(size_of_a_matrix, diagonalVal, diagonalAboveVal, diagonalBelowVal)
    Identitymatrix = np.identity(size_of_a_matrix)
    bvec = np.zeros_like(x)
    bvec[0] = -1
    bvec[Nx] = -1
    un = np.zeros_like(x) 
    n = 0
    # we need to solve tridiagonal system
    # we first try direct solver nad then we use this solver in an iterative scheme to get convergent solution
    #finally we will try inverse matrix method
    unm1 = np.zeros_like(x)
    Fvals = np.zeros_like(x)
    for i in range(len(x)):
        unm1[i]  = I(x[i])
        Fvals[i] = f(x[i], t[n])
    mat1 = np.linalg.inv(4.0/C2*Identitymatrix + Kmatrix) 
    mat2 = 4.0/C2*Identitymatrix - Kmatrix
    prod1 = np.matmul(mat2, unm1) - 2*bvec + 0.5*dt**2*Fvals
    prod2 = np.matmul(mat1, prod1)
    print(type(prod1), prod1.shape)
    for i in range(len(x)):
        un[i] = prod2[i] + dt*V(x[i]) 
    un[0] = 0.0
    un[-1] = 0.0
    #Now compare result at this step
    plt.plot(x, exact_sol(x, t[1]), label='exact')
    plt.plot(x, un , label='numeric')
    plt.legend()
    plt.figure()
    diff = un - exact_sol(x, t[1])
    plt.plot(x, diff)
    plt.title("{0}".format(abs(diff).max()))
    plt.show()
    print("diff = ", abs(un - exact_sol(x, t[0])).max()) 
    
    #n=1  and so 
    u = np.zeros_like(x)
    for n in range(1, 3):
        mat1 = np.linalg.inv(4.0/C2*Identitymatrix + Kmatrix)
        mat2 = 2.0* (4.0/C2*Identitymatrix - Kmatrix)
        prod1 = np.matmul(mat2, unm1) - 4.0*bvec + dt**2*Fvals
        prod2 = np.matmul(mat1, prod1)
        for i in range(len(x)):
            u[i] = prod2[i] + 2.0*dt*un[i]
        u[0] = 0.0
        u[-1] = 0.0
        #switch
        unm1[:] = un
        un[:] = u

        #if n%10==0:
        plt.plot(x, exact_sol(x, t[n]), label='exact')
        plt.plot(x, u, label='numeric_at t={0:.3f}'.format(t[n]))
        plt.legend()
        plt.show()
    return x, t, u 

x = np.linspace(0, 2.5, 100)
t =  np.linspace(0, 1, 100)

implicit(x, t)


