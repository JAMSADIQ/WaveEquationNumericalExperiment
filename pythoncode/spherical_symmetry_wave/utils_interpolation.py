### Some interpolation functions for RK4 integrartion of our problems
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange, interp1d, splev, splrep
from numpy.polynomial.polynomial import Polynomial


def k_order_spline_interp(xval, yval, interpX, order=4):
    """
    k=4
    by default
    changer order to get correct order spline interpolator
    returned triand function use it to get value on a
    given x val.
    """
    prepare_interp = splrep(xval, yval, k=order)
    interp_sol =  splev(interpX, prepare_interp)
    return interp_sol

def local_lagrange_interp(xarray, yarray, x_interp, order=4):
    """
    using scipy and numpy to locally interpolate
    on xarray, yarray data and get result on x_interp
    array
    order is same at boundary as in between 
    but can be change the order easily changed
    """
    y_interp = np.zeros_like(x_interp)
    order = order + 2 # to make sure order is ok for intermediate points
    for i in range(len(x_interp)):
        #find the index in xarray that is closest to x_interp[i]
        indx = np.searchsorted(xarray, x_interp[i])
        if indx <= order:
            x = xarray[indx : indx+order]
            y = yarray[indx : indx+order]
            #print("start order", i,  len(x))
        elif indx >= len(xarray)-1:
            x = xarray[indx-order : indx]
            y = yarray[indx-order : indx]
            #print("end order", i,  len(x))
        else:
            x = xarray[indx - int(order/2): indx + int(order/2)]
            y = yarray[indx - int(order/2): indx + int(order/2)]
        #create polynomial or order via scipy
        poly = lagrange(x, y)
        #evaluate valuer at xinterp #using numpy here to get value
        y_interp[i] = Polynomial(poly.coef[::-1])(x_interp[i])
    return y_interp


###########My code for lagrange interpolation ########## not using scipy
def lagrange(x, y, xp):
    """
     order of lagrange
     polynomial depend
     on input data x and y lengths
    """
    yf = np.zeros(len(x))
    for i in range(len(x)):
        p = 1.0
        for j in range(len(x)):
            if (j != i):
                p *= (x[i] - x[j])
        yf[i] = y[i]/p

    yk = 0.0
    for i in range(len(x)):
        p = 1e0
        for j in range(len(x)):
            if(j != i):
                p *= (xp - x[j])
        yk += p* yf[i]

    return yk


def local_interp(x_grid, y, xp, order=5):
    """
    x_grid is all data on which we want interpolate
    xp is point at which interpolated values is needed
    I first need to find four points of x_grid  close to
    xp using index search
    if the indx is
    """
    k = order + 1 # 5 bydeault
    indx = np.searchsorted(x_grid, xp)
    if indx == 0:
        fp = lagrange(x_grid[indx:indx+6], y[indx:indx+6], xp)
    elif indx == 1:
        fp = lagrange(x_grid[indx-1:indx+5], y[indx-1:indx+5], xp)
    elif indx == 2:
        fp = lagrange(x_grid[indx-2:indx+4], y[indx-2:indx+4], xp)

    elif indx == len(x_grid)-1:
        fp = lagrange(x_grid[indx-6:indx], y[indx-6:indx], xp)

    elif indx == len(x_grid)-2:
        fp = lagrange(x_grid[indx-5:indx+1], y[indx-5:indx+1], xp)

    elif indx == len(x_grid)-3:
        fp = lagrange(x_grid[indx-4:indx+2], y[indx-4:indx+2], xp)

    else:
        fp = lagrange(x_grid[indx-3:indx+3], y[indx-3:indx+3], xp)

    return fp

######################
#need to have test and add main to cehck script
