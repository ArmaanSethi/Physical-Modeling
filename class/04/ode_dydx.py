import numpy as np
import globalvar

#==============================================================
# function dydx = h2formation(x,y,dx)
#
# Calculates RHS for molecular hydrogen formation network. 
#
# input: 
#   x,y    : time and abundances (A,B,C) for (n(H), n(H2), n(H+))
#
# global:
#   reaction rate coefficients stored in par
# output:
#   dydx    : vector of results as in y'=f(x,y)
#--------------------------------------------------------------

def h2formation(x,y,dx):
    nspec = y.size
    par   = globalvar.get_odepar()
    A     = y[0] # not necessary, but more readable
    B     = y[1]
    C     = y[2]
    #print 'A,B,C,sum = ',A,B,C,A+B+C
    # This is the stoichiometry matrix S
    S     = np.array([[-2.0, 2.0,-1.0, 1.0],
                      [ 1.0,-1.0, 0.0, 0.0],
                      [ 0.0, 0.0, 1.0,-1.0]])
    # This is the reaction rate vector. 
    # Note that the stoichiometry would give par[3]*C, not par[3]*C*C. 
    # However, we need to account for the electrons, whose density is that of C.
    # Also note that B is the abundance of H bound in H_2.
    v     = np.array([par[0]*A*A,par[1]*B,par[2]*A,par[3]*C*C])
    Myr   = 1e6*3.65e2*3.6e3*2.4e1
    dydx  = S.dot(v)
    return dydx

#==============================================================
# function dydx = doubleexp(x,y,dx)
#
# Calculates RHS for stiff test equation
#--------------------------------------------------------------

def doubleexp(x,y,dx):
    par     = globalvar.get_odepar()
    dydx    = np.zeros(2)
    dydx[0] =  998.0*y[0]+1998.0*y[1]
    dydx[1] = -999.0*y[0]-1999.0*y[1]
    return dydx

#===============================================================
def enrightpryce(x,y,dx):
    dydx    = np.zeros(3)
    dydx[0] = -0.013*y[0]-1000.0*y[0]*y[2]
    dydx[1] = -2500.0*y[1]*y[2]
    dydx[2] = -0.013*y[0]-1000.0*y[0]*y[2]-2500.0*y[1]*y[2]
    return dydx

