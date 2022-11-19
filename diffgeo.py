#added a comment
from __future__ import print_function
import numpy as np
from numpy import sin, cos

try:
  xrange
except NameError:
  xrange = range

dg_prec = np.float128
fd_h = dg_prec(0.001)

# 8th order derivative operator
def diff(f, coord, extra = None):
  delta = np.identity(len(coord))
  out = []
  for i in xrange(len(coord)):
    f1=f(coord + 1 * fd_h * delta[i], extra)
    f2=f(coord + 2 * fd_h * delta[i], extra)
    f3=f(coord + 3 * fd_h * delta[i], extra)
    f4=f(coord + 4 * fd_h * delta[i], extra)
    fm1=f(coord - 1 * fd_h * delta[i], extra)
    fm2=f(coord - 2 * fd_h * delta[i], extra)
    fm3=f(coord - 3 * fd_h * delta[i], extra)
    fm4=f(coord - 4 * fd_h * delta[i], extra)
    out.append((3*fm4 - 32*fm3 + 168*(fm2 - 4*fm1 + 4*f1 -f2) + 32*f3\
      - 3*f4)/(840 * fd_h))
  return np.array(out)

# 8th order second derivative operator. Usess diff()
def ddiff(f, coord, extra = None):
  delta = np.identity(len(coord))
  out = []
  for i in xrange(len(coord)):
    f1=diff(f,coord + 1 * fd_h * delta[i], extra)
    f2=diff(f,coord + 2 * fd_h * delta[i], extra)
    f3=diff(f,coord + 3 * fd_h * delta[i], extra)
    f4=diff(f,coord + 4 * fd_h * delta[i], extra)
    fm1=diff(f,coord - 1 * fd_h * delta[i], extra)
    fm2=diff(f,coord - 2 * fd_h * delta[i], extra)
    fm3=diff(f,coord - 3 * fd_h * delta[i], extra)
    fm4=diff(f,coord - 4 * fd_h * delta[i], extra)
    out.append((3*fm4 - 32*fm3 + 168*(fm2 - 4*fm1 + 4*f1 -f2) + 32*f3\
        - 3*f4)/(840 * fd_h))
  return np.array(out)

def kmet(coord, extra = None):
  t,r,theta,phi = coord
  M = dg_prec(1.0)
  a_spin = dg_prec(0.9)

  gtt = -((a_spin**4 - 2*a_spin**2*M**2 + (M**2 - 4*r**2)**2 +\
    8*a_spin**2*r**2*cos(2*theta))/(a_spin**4 + (M + 2*r)**4 -\
      2*a_spin**2*M*(M + 4*r) + 8*a_spin**2*r**2*cos(2*theta)))

  gtr = 0

  gtth = 0

  gtph = (8*a_spin*M*r*(a_spin**2 - (M +\
    2*r)**2)*sin(theta)**2)/(a_spin**4 + (M + 2*r)**4 -\
        2*a_spin**2*M*(M + 4*r) + 8*a_spin**2*r**2*cos(2*theta))

  grr = (a_spin**4 + (M + 2*r)**4 - 2*a_spin**2*M*(M + 4*r) +\
      8*a_spin**2*r**2*cos(2*theta))/(16.*r**4)

  grth = 0

  grph = 0

  gthth = (M + ((-a_spin + M)*(a_spin + M))/(4.*r) + r)**2 +\
      a_spin**2*cos(theta)**2

  gthph=0

  gphph = (sin(theta)**2*((a_spin**2 + (M + ((-a_spin + M)*(a_spin +\
    M))/(4.*r) + r)**2)**2 - (a_spin**2*(a_spin**2 - M**2 +\
      4*r**2)**2*sin(theta)**2)/(16.*r**2)))/((M + ((-a_spin +\
        M)*(a_spin + M))/(4.*r) + r)**2 + a_spin**2*cos(theta)**2)

  return np.array([[gtt, gtr, gtth, gtph],\
                   [gtr, grr, grth, grph],\
                   [gtth, grth, gthth, gthph],\
                   [gtph, grph, gthph, gphph]], dtype=dg_prec)


  
def met(coord, extra = None):
  t,r,theta,phi = coord
  m=dg_prec(1.0)
  return np.array([[-1 + 2 * m / r, 0,0,0],\
                   [0,1/(1-2 * m / r),0,0],\
                   [0,0,r**2,0],[0,0,0,r**2*sin(theta)**2]], dtype=dg_prec)


def metric_to_inv_metric(gab, dgab):
  dim = len(gab)

  # linalg does not work on float128, so we drop precision
  ginv = np.linalg.inv(np.float64(gab))

  dginv = np.zeros(shape=(dim,dim,dim), dtype=dg_prec)

  for l in xrange(dim):
    for i in xrange(dim):
      for j in xrange(dim):
        dginv[l][i][j] = 0
        for a in xrange(dim):
          for b in xrange(dim):
            dginv[l][i][j] += - ginv[i][a] * ginv[j][b] * dgab[l][a][b]
  return ginv, dginv
 

def metric_to_gamma(ginv, dgab):
  dim = len(ginv)
  downgamma = np.zeros(shape=(dim,dim,dim), dtype=dg_prec)
  Gamma = np.zeros(shape=(dim,dim,dim), dtype=dg_prec)

  for c in xrange(dim):
    for a in xrange(dim):
      for b in xrange(dim):
        downgamma[c][a][b] = (dgab[a][b][c] + dgab[b][a][c] - dgab[c][a][b])/2

  for a in xrange(dim):
    for b in xrange(dim):
      for c in xrange(dim):
        for d in xrange(dim):
          Gamma[c][a][b] += ginv[c][d] * downgamma[d][a][b]


  return Gamma

def metric_to_dgamma(ginv, dginv, dgab, ddgab):
  dim = len(ginv)
  downgamma = np.zeros(shape=(dim,dim,dim), dtype=dg_prec)
  ddowngamma = np.zeros(shape=(dim,dim,dim,dim), dtype=dg_prec)
  dGamma = np.zeros(shape=(dim,dim,dim,dim), dtype=dg_prec)

  for c in xrange(dim):
    for a in xrange(dim):
      for b in xrange(dim):
        downgamma[c][a][b] = (dgab[a][b][c] + dgab[b][a][c] - dgab[c][a][b])/2

  for l in xrange(dim):
    for c in xrange(dim):
      for a in xrange(dim):
        for b in xrange(dim):
          ddowngamma[l][c][a][b] = (ddgab[l][a][b][c] + ddgab[l][b][a][c] - ddgab[l][c][a][b])/2


  for l in xrange(dim):
    for a in xrange(dim):
      for b in xrange(dim):
        for c in xrange(dim):
          for d in xrange(dim):
            dGamma[l][c][a][b] += dginv[l][c][d] * downgamma[d][a][b] +\
                 ginv[c][d] * ddowngamma[l][d][a][b]


  return dGamma


def Riemann_dn_dn_dn_up(gamma, dgamma):
  dim = len(gamma)
  out = np.zeros(shape=(dim,dim,dim,dim),dtype=dg_prec)

  for mu in xrange(dim):
    for nu in xrange(dim):
      for rho in xrange(dim):
        for sigma in xrange(dim):
          out[mu][nu][rho][sigma] = dgamma[nu][sigma][mu][rho] -\
              dgamma[mu][sigma][nu][rho]
          for alpha in xrange(dim):
            out[mu][nu][rho][sigma] = out[mu][nu][rho][sigma]  + \
                gamma[alpha][mu][rho]*gamma[sigma][alpha][nu] - \
                gamma[alpha][nu][rho]*gamma[sigma][alpha][mu]

  return out

def Riemann_dn_dn_dn_dn(metric, Rdddu):
  dim = len(metric)
  out = np.zeros(shape=(dim,dim,dim,dim),dtype=dg_prec)

  for mu in xrange(dim):
    for nu in xrange(dim):
      for rho in xrange(dim):
        for sigma in xrange(dim):
          out[mu][nu][rho][sigma] = 0
          for alpha in xrange(dim):
            out[mu][nu][rho][sigma] +=\
                metric[alpha][sigma] * Rdddu[mu][nu][rho][alpha]
  return out

def Ricci(Rdddu):
  dim = len(Rdddu)
  out = np.zeros(shape=(dim,dim),dtype=dg_prec)
  for mu in xrange(dim):
    for nu in xrange(dim):
      out[mu][nu] = 0
      for sigma in xrange(dim):
        out[mu][nu] += Rdddu[mu][sigma][nu][sigma]
  return out

def threericci_to_threeriemann(gab, Rab, R):
  out = np.zeros(shape=(3,3,3,3), dtype=dg_prec)

  for a in xrange(3):
    for b in xrange(3):
      for c in xrange(3):
        for d in xrange(3):
          out[a][b][c][d] = \
            +gab[a][c]*Rab[d][b] \
            -gab[a][d]*Rab[c][b] \
            -gab[b][c]*Rab[d][a] \
            +gab[b][d]*Rab[c][a] \
            -0.5 * R * gab[a][c]*gab[d][b]\
            +0.5 * R * gab[a][d]*gab[c][b]
  return out

def fourmetric_to_threemetric(gmunu, dgmunu, ddgmunu):
  betadown = np.zeros(shape=(3), dtype=dg_prec)
  dalpha = np.zeros(shape=(3), dtype=dg_prec)
  betaup = np.zeros(shape=(3), dtype=dg_prec)
  dbetadown = np.zeros(shape=(3,3), dtype=dg_prec)
  dbetaup = np.zeros(shape=(3,3), dtype=dg_prec)
  ddbetadown = np.zeros(shape=(3,3,3), dtype=dg_prec)
  Kab = np.zeros(shape=(3,3), dtype=dg_prec)
  gab = np.zeros(shape=(3,3), dtype=dg_prec)
  dgab = np.zeros(shape=(3,3,3), dtype=dg_prec)
  ddgab = np.zeros(shape=(3,3,3,3), dtype=dg_prec)
  dtgab = np.zeros(shape=(3,3), dtype=dg_prec)
  dtdgab = np.zeros(shape=(3,3,3), dtype=dg_prec)
  dKab = np.zeros(shape=(3,3,3), dtype=dg_prec)
  dginv = np.zeros(shape=(3,3,3), dtype=dg_prec)

  for i in xrange(3):
    betadown[i] =  gmunu[0][i+1]
    for j in xrange(3):
      gab[i][j] = gmunu[i+1][j+1]

  for l in xrange(3):
    for m in xrange(3):
      for i in xrange(3):
        ddbetadown[l][m][i] = ddgmunu[l+1][m+1][0][i+1]
        for j in xrange(3):
          ddgab[l][m][i][j] = ddgmunu[l+1][m+1][i+1][j+1]

  for m in xrange(3):
    for i in xrange(3):
      dbetadown[m][i] = dgmunu[m+1][0][i+1]
      for j in xrange(3):
        dgab[m][i][j] = dgmunu[m+1][i+1][j+1]

  ginv, dginv = metric_to_inv_metric(gab, dgab)


  for i in xrange(3):
    betaup[i] = 0
    for a in xrange(3):
      betaup[i] += ginv[i][a] * betadown[a]

  for l in xrange(3):
    for i in xrange(3):
      dbetaup[l][i] = 0
      for a in xrange(3):
        dbetaup[l][i] += dginv[l][i][a] * betadown[a] +\
                         ginv[i][a] * dbetadown[l][a]

  b2 = np.dot(betadown, betaup)

  a2 = b2 - gmunu[0][0]

  alpha = np.sqrt(a2)

  for l in xrange(3):
    dalpha[l] = -dgmunu[l+1][0][0]
    for a in xrange(3):
      dalpha[l] += dbetaup[l][a]*betadown[a] + betaup[a]*dbetadown[l][a]
    dalpha[l] /= 2 * alpha

  for i in xrange(3):
    for j in xrange(3):
      dtgab[i][j] = dgmunu[0][i+1][j+1]

  for l in xrange(3):
    for i in xrange(3):
      for j in xrange(3):
        dtdgab[l][i][j] = ddgmunu[0][l+1][i+1][j+1]

  for i in xrange(3):
    for j in xrange(3):
      Kab[i][j] = dbetadown[i][j] +  dbetadown[j][i] - dtgab[i][j]
      for a in xrange(3):
        Kab[i][j] -= betaup[a] * (dgab[i][j][a] +\
                     dgab[j][i][a] - dgab[a][i][j])
      Kab[i][j] /= 2 * alpha

  for l in xrange(3):
    for i in xrange(3):
      for j in xrange(3):
        dKab[l][i][j] = ddbetadown[l][i][j] +\
                        ddbetadown[l][j][i] -\
                        dtdgab[l][i][j]
        for a in xrange(3):
          dKab[l][i][j] -= dbetaup[l][a] * (dgab[i][j][a] +\
                           dgab[j][i][a] - dgab[a][i][j]) +\
                           betaup[a] * (ddgab[l][i][j][a] +\
                           ddgab[l][j][i][a] - ddgab[l][a][i][j])

        dKab[l][i][j] /= 2 * alpha
        dKab[l][i][j] -=  Kab[i][j] * dalpha[l]/alpha

  Gamma = metric_to_gamma(ginv, dgab)
  dGamma = metric_to_dgamma(ginv, dginv, dgab, ddgab)

  Rdddu = Riemann_dn_dn_dn_up(Gamma, dGamma)

  Kupdn = np.dot(ginv, Kab)
  Kupup = np.dot(Kupdn, ginv)
  trK = np.trace(Kupdn)
  K2ab = np.dot(Kab, Kupdn)
  Rdd = Ricci(Rdddu)
  DKab = dKab.copy()
  for a in xrange(3):
    for b in xrange(3):
      for c in xrange(3):
        for d in xrange(3):
          DKab[a][b][c] -= Gamma[d][a][b]*Kab[d][c] + Gamma[d][a][c]*Kab[b][d]


  DivKup = np.zeros(shape=(3), dtype=dg_prec)
  GradKdn = np.zeros(shape=(3), dtype=dg_prec)

  for a in xrange(3):
    for b in xrange(3):
      for c in xrange(3):
        GradKdn[a] += ginv[b][c] * DKab[a][b][c]
        for d in xrange(3):
          DivKup[a] += ginv[a][b] * ginv[c][d] * DKab[c][d][b]

  Rnnab = Rdd.copy()

  for a in xrange(3):
    for b in xrange(3):
      Rnnab[a][b] += trK * Kab[a][b] - K2ab[a][b]

  R = np.trace(np.dot(ginv,Rdd))
  Rtest  = threericci_to_threeriemann(gab, Rdd, R)
  Rdddd = Riemann_dn_dn_dn_dn(gab, Rdddu)
  print ("Rdddd test {0:20.16e}".format(np.amax(np.abs(Rtest - Rdddd))))
  print ("Ham = {0:20.16e}".format(R + trK**2 - np.trace(np.dot(Kupup, Kab))))
  print ("Mom = {}".format(DivKup - np.dot(ginv, GradKdn)))







coord = np.array((0,3,4,5), dtype=dg_prec)

np.set_printoptions(precision=16)
metric = kmet(coord)
dmetric = diff(kmet, coord)
ddmetric = ddiff(kmet, coord)
(ginv, dginv) = metric_to_inv_metric(metric, dmetric)

gamma = metric_to_gamma(ginv, dmetric)
dgamma = metric_to_dgamma(ginv, dginv, dmetric, ddmetric)
Rdddu = Riemann_dn_dn_dn_up(gamma, dgamma)
Rdddd = Riemann_dn_dn_dn_dn(metric, Rdddu)
Rdd = Ricci(Rdddu)
print (np.amax(np.abs(Rdd)))


fourmetric_to_threemetric(metric, dmetric, ddmetric)
