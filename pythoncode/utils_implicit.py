#extra functions like tridiagonal solver and Jacobi or gauss siedel solver or inverse matrix function

import numpy as np

def TDMA(a,b,c,d):
    n = len(d)
    w= np.zeros(n-1,float)
    g= np.zeros(n, float)
    p = np.zeros(n,float)
    
    w[0] = c[0]/b[0]
    g[0] = d[0]/b[0]

    for i in range(1,n-1):
        w[i] = c[i]/(b[i] - a[i-1]*w[i-1])
    for i in range(1,n):
        g[i] = (d[i] - a[i-1]*g[i-1])/(b[i] - a[i-1]*w[i-1])
    p[n-1] = g[n-1]
    for i in range(n-1,0,-1):
        p[i-1] = g[i-1] - w[i-1]*p[i]
    return p


def TDMAsolver(a, b, c, d):
    """
    a b c d can be NumPy array type or Python list type.
    d and  b have same lengths as # of eqs
    a and c are off diagnol so have len(d) -1 terms 
     http://www.cfd-online.com/Wiki/\
     Tridiagonal_matrix_algorithm_-_TDMA_(Thomas_algorithm)
    """
    nf = len(d) #number of equations
    ac, bc, cc, dc = map(np.array, (a, b, c, d)) #copy arrays
    for it in range(1, nf):
        mc = ac[it - 1]/bc[it - 1]
        bc[it] = bc[it] - mc*cc[it - 1]
        dc[it] = dc[it] - mc*dc[it - 1]

    xc = bc
    xc[-1] = dc[-1]/bc[-1]

    for il in range(nf-2, -1, -1):
        xc[il] = (dc[il] - cc[il]*xc[il + 1])/bc[il]

    return xc

#test TDMA
#A = np.array([[10, 2, 0, 0],[3, 10, 4, 0], [0, 1, 7, 5], [0, 0, 3, 4]], dtype=float)
#d = np.array([3, 4, 5, 6.])
##solve via linear algebra
#sol1 = np.linalg.solve(A, d)
##Now TDMA solver
#a = np.array([3., 1., 3.]) #below the diagonal
##b = np.array([10., 10., 7., 4.])
#c = np.array([2., 4., 5.]) #above the diagonal
#sol2 = TDMAsolver(a, b, c, d)
#sol3 = TDMA(a, b, c, d)
#print(sol1, sol2, sol3)
#quit()
#
#Inverse first we need tridaigonal Matrix and its easy to get its invers and use A-1*d with np.dot should give what solver do

def TridiagonalMatrix(size_of_a_matrix, diagonalVal, diagonalAboveVal, diagonalBelowVal):
    diagonal = np.ones(size_of_a_matrix) *diagonalVal
    diagonalAbove = np.ones(size_of_a_matrix-1) *diagonalAboveVal
    diagonalBelow = np.ones(size_of_a_matrix-1) *diagonalBelowVal
    matrix = [[0 for j in range(size_of_a_matrix)]
              for i in range(size_of_a_matrix)]
      
    for k in range(size_of_a_matrix-1):
        matrix[k][k] = diagonal[k]
        matrix[k][k+1] = diagonalAbove[k]
        matrix[k+1][k] = diagonalBelow[k]
      
    matrix[size_of_a_matrix-1][size_of_a_matrix - 1] = diagonal[size_of_a_matrix-1]
   
  
    return matrix # "this is the tridiagonal matrix"

#print(tridiagonal(5, 6, -2, -1))
#scipy.linalg.solve_triangular(a, b)
