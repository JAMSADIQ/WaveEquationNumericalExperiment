#Plots for wave solutions
import numpy as np
import matplotlib.pyplot as plt
#fixed Style
from matplotlib import rcParams
rcParams["text.usetex"] = True
rcParams["font.serif"] = "Computer Modern"
rcParams["font.family"] = "Serif"
rcParams["xtick.labelsize"]=14
rcParams["ytick.labelsize"]=14
rcParams["xtick.direction"]="in"
rcParams["ytick.direction"]="in"
rcParams["legend.fontsize"]=16
rcParams["axes.labelsize"]=16
rcParams["axes.grid"] = True
rcParams["grid.color"] = 'grey'
rcParams["grid.linewidth"] = 1.
rcParams["grid.linestyle"] = ':'
rcParams["grid.alpha"] = 0.8
#from matplotlib.pyplot import cm
#color = iter(cm.plasma(np.linspace(0, 0.9, 6)))

#Plot of numeric and analytic solution of 1D wave at fix time

def waveplot_fixed_t(xgrid, f_numeric, f_exact, t=0, drf=None, fymin=-1, fymax=1):
    """
    given  x-grid  and solutions [Numerix/exact]
    on x-grid get plot at given time
    If drf is not None plot numeric and exact derivs
    as well
    also get the residual of analytic and numeric result
    with time and  max error as title 
    """
    fig, axs = plt.subplots(2,1,figsize=(16,9), gridspec_kw={'height_ratios': [2, 1]})
    axs[0].plot(xgrid, f_numeric, 'r', lw =2, label='numeric')
    axs[0].plot(xgrid, f_exact, 'k--', label='analytice')
    if drf !=None  and dtf !=None:
        axs[0].plot(xgrid, drf, color='cyan', lw=2, label='numeric-drU')
        axs[0].plot(xgrid, np.gradient(f_exact, xgrid) , color='orange', lw=2, label='analytic-drU', ls='--')
    axs[0].set_ylabel("u at t= {0:.3f}".format(t))
    axs[0].set_ylim(fymin, fymax)
    axs[0].legend()
    axs[0].set_xlabel("x")
    residual = np.abs(f_exact - f_numeric)
    axs[1].plot(xgrid, residual, label='residual')
    axs[1].set_ylabel("|exact - numeric| at t= {0:.3f}".format(t))
    axs[1].semilogy()
    axs[1].legend()
    axs[1].set_ylim(ymin=1e-9, ymax=1.0)
    axs[1].set_xlabel("x")
    axs[1].set_title("max Err = {0}".format(max(residual)))
    plt.show()
    return 0



def draw_frame(x, y, y2, time, tn):
    """
    To make plots of u 
    if we want we can also
    make plot of  first derivatives for space 
    and time
    """
    #y2 = np.zeros_like(x)
    #yt2 = np.zeros_like(x)
    #yx2 = np.zeros_like(x)
    #for it in range(len(x)):
    #    y2[it] = exact_sol(x[it], tn)/x[it]
        #yx2[it] = dexact_dt(x[it], tn)/x[it]
        #yt2[it] = dexact_dr(x[it], tn)/x[it]
    err = np.abs(y - y2)
    #dterr = np.abs(yt - yt2)
    fig, axs = plt.subplots(2,1,figsize=(16,9), gridspec_kw={'height_ratios': [2, 1]})
    axs[0].plot(x, y, 'r', lw =2, label='numeric-U')
    axs[0].plot(x, y2, 'k--', label='exact-U')
    #axs[0].plot(x, yt, color='cyan', lw=2, label='numeric-dtU')
    #axs[0].plot(x, yx, color='orange', lw=2, label='numeric-drU')
    #axs[0].plot(x, yt2, color='magenta', linestyle='--' ,label='exactdtU')
    #axs[0].plot(x, yx2, color='brown', linestyle='--' ,label='exactdrU')
    axs[0].set_ylabel("u at t= {0:.3f}".format(time))
    axs[0].set_ylim(-1, 1)
    axs[0].legend()
    axs[0].set_xlabel("x")
    axs[1].plot(x, err, label='u')
    #axs[1].plot(x, dterr, label='d_t u')
    #axs[1].plot(x, dterr, label='d_r u')
    axs[1].set_ylabel("|exact - numeric| at t= {0:.3f}".format(time))
    axs[1].semilogy()
    axs[1].legend()
    axs[1].set_ylim(ymin=1e-9, ymax=1.0)
    axs[1].set_xlabel("x")
    return 0

