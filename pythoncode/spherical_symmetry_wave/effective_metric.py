import numpy as np

def Gamma_tt(a, alpha, g, Phi, Pi):
    return -1.0/alpha**2*(1.0 - g*2.0*Pi**2/(a**2 + g*(Phi**2 - Pi**2)))

def Gamma_rr(a, alpha, g, Phi, Pi):
    return -1.0/a**2*(1.0 + g*2.0*Phi**2/(a**2 + g*(Phi**2 - Pi**2)))

def Gamma_tr(a, alpha, g, Phi, Pi):
    return -g*2.0*Pi*Phi/(a*alpha*(a**2 +g*(Phi**2 - Pi**2)))



#def lambdapm(Ytt, Yrr, Ytr):
def lambdapm(a, alpha, g, Phi, Pi):
    Ytt = Gamma_tt(a, alpha, g, Phi, Pi)
    Yrr = Gamma_rr(a, alpha, g, Phi, Pi)
    Ytr = Gamma_tr(a, alpha, g, Phi, Pi)
    detY = Ytt*Yrr - Ytr**2
    lmdaplus = (Ytt + Yrr)/2.0 + np.sqrt((0.5*(Ytt + Yrr))**2- detY)
    lmdaminus = (Ytt + Yrr)/2.0 - np.sqrt((0.5*(Ytt + Yrr))**2- detY)
    return lmdaplus, lmdaminus

#def Vpm(Ytt, Yrr, Ytr):
def Vpm(a, alpha, g, Phi, Pi):
    Ytt = Gamma_tt(a, alpha, g, Phi, Pi)
    Yrr = Gamma_rr(a, alpha, g, Phi, Pi)
    Ytr = Gamma_tr(a, alpha, g, Phi, Pi)
    detY = Ytt*Yrr - Ytr**2
    Vplus = -Ytr/Ytt + np.sqrt(-detY/Ytt**2)
    Vplus = -Ytr/Ytt - np.sqrt(-detY/Ytt**2)
    return Vplus, Vminus


#def Principal_matrix(Ytt, Yrr, Ytr):
def Principal_matrix(a, alpha, g, Phi, Pi):
    Ytt = Gamma_tt(a, alpha, g, Phi, Pi)
    Yrr = Gamma_rr(a, alpha, g, Phi, Pi)
    Ytr = Gamma_tr(a, alpha, g, Phi, Pi)
    M00 = 0.0
    M01 = alpha/a
    M10 = -a*Yrr/(alpha*Ytt)
    M11 = -2.0*Ytr/Ytt
    return np.array([[M00, M10],[M01, M11]])
