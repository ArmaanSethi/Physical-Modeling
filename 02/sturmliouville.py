# Solver for a Sturm-Liouville problem (finite string Eigenmodes)
# This can be generalized to any BVP, as long as it's solvable
# by a shooting method.

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import p358utilities as util

#========================================
# Density function for string
# Returns the mass density at given position x.
# Here, we make use of optional arguments:
# The syntax would be (for a flat density):
# a = rho(x,imode=0)
def rho(x,**kwargs):
    for key in kwargs:
        if (key=='imode'):
            imode = kwargs[key]
    if (imode == 0):   # flat
    # ????? from here
        rho = 1
    # ????? to here
    elif (imode == 1): # exponential
    # ????? from here
        rho = 1+np.exp(10*x)
    # ????? to here
    return rho

#========================================
# fRHS for the (non-)uniform string
# Should return an array dydx containing three
# elements corresponding to y'(x), y''(x), and lambda'(x).
def dydx_string(x,y,**kwargs):
    dydx    = np.zeros(3)
    # ????? from here
    dydx[0] = y[1]
    dydx[1] = -(np.pi*y[2])**2 * y[0] * rho(x, **kwargs)
    dydx[2] = 0
    # ????? to here
    return dydx

#========================================
# fLOA for the string problem
# This is the (boundary) loading function.
# Should return the integration boundaries x0,x1, and
# the initial values y(x=0). Therefore, y must be
# an array with three elements. 
# The function takes an argument v, which in our case
# is just the eigenvalue lambda.
# Note that some of the initial values in y can
# (and should!) depend on lambda.
def load_string(v):
    # ????? from here
    y = np.zeros(3)
    x0 = 0
    x1 = 1
    y[0] = 0
    y[1] = np.pi*v
    y[2] = v
    # ????? to here
    return x0,x1,y

#========================================
# fSCO for the string problem
# This is the scoring function. Should return
# the function that needs to be zeroed by the
# root finder. In our case, that's just y(1). 
# The function takes the arguments x and y, where
# y is an array with three elements.
def score_string(x,y):
    # ????? from here
    score = y[0][-1]
    # ????? to here
    return score # displacement should be zero.

#========================================
# Single rk4 step.
# Already provided. Good to go; 2.5 free points :-)
def rk4(fRHS,x0,y0,dx,**kwargs):
    k1 = dx*fRHS(x0       ,y0       ,**kwargs)
    k2 = dx*fRHS(x0+0.5*dx,y0+0.5*k1,**kwargs)
    k3 = dx*fRHS(x0+0.5*dx,y0+0.5*k2,**kwargs)
    k4 = dx*fRHS(x0+    dx,y0+    k3,**kwargs)
    y  = y0+(k1+2.0*(k2+k3)+k4)/6.0
    return y    

#========================================
# ODE IVP driver.
# Already provided. Good to go; 2.5 free points :-)
def ode_ivp(fRHS,fORD,x0,x1,y0,nstep,**kwargs):
    nvar    = y0.size                      # number of ODEs
    x       = np.linspace(x0,x1,nstep+1)   # generates equal-distant support points
    y       = np.zeros((nvar,nstep+1))     # result array 
    y[:,0]  = y0                           # set initial condition
    dx      = x[1]-x[0]                    # step size
    for k in range(1,nstep+1):
        y[:,k] = fORD(fRHS,x[k-1],y[:,k-1],dx,**kwargs)
    return x,y

#=======================================
# A single trial shot.
# Sets the initial values (guesses) via fLOA, calculates 
# the corresponding solution via ode_ivp, and returns 
# a "score" via fSCO, i.e. a value for the rootfinder to zero out.
def bvp_shoot(fRHS,fORD,fLOA,fSCO,v,nstep,**kwargs):
    # ????? from here
    x0,x1,y0 = fLOA(v)
    x,y = ode_ivp(fRHS, fORD, x0, x1, y0, nstep, **kwargs)
    score = fSCO(x,y)
    # ????? to here
    return score # this should be zero, and thus can be directly used.

#=======================================
# The rootfinder.
# The function pointers are problem-specific (see main()). 
# v0 is the initial guess for the eigenvalue (in our case).
# Should return x,y, so that the solution can be plotted.
def bvp_root(fRHS,fORD,fLOA,fSCO,v0,nstep,**kwargs):
    # ????? from here
    def samesign(a, b):
        return a * b > 0.0

    f0 = bvp_shoot(fRHS, fORD, fLOA, fSCO, v0, nstep, **kwargs)
    f1 = bvp_shoot(fRHS, fORD, fLOA, fSCO, v0, nstep, **kwargs)

    #-------------------Bracket the Solution-------------------#
    i = 0
    while( samesign(f0, f1) and (i < nstep) ) :
        f1 = bvp_shoot(fRHS, fORD, fLOA, fSCO, v0*np.power(1.1, i), nstep, **kwargs)
        i+=1

    # v_left = v0*np.power(1.1, 0)
    v_left = v0*np.power(1.1, i-2)
    v_right = v0*np.power(1.1, i-1)

    #-------------------Bisection Method-------------------#
    tol = np.power(10.0, -8)
    print("Bisection")
    v_final = 0
    for k in range(100):
        midpoint = (v_left + v_right) / 2.0

        score_left = bvp_shoot(fRHS,fORD,fLOA,fSCO,v_left,nstep,**kwargs)
        score_midpoint = bvp_shoot(fRHS,fORD,fLOA,fSCO,midpoint,nstep,**kwargs)
        score_right = bvp_shoot(fRHS,fORD,fLOA,fSCO,v_right,nstep,**kwargs)

        print(k,score_midpoint, midpoint)
        if(( abs(score_midpoint) <= tol)):
            v_final = midpoint
            break

        if( samesign(score_left,score_midpoint) ):
            v_left = midpoint
        else:
            v_right = midpoint 
        
        
    x0,x1,y0 = fLOA(v_final)
    x,y = ode_ivp(fRHS, fORD, x0, x1, y0, nstep, **kwargs)
    # ????? to here
    return x,y

def main():

    nstep = 500
    imode = 1
    v0    = np.array([1.5])
    fRHS  = dydx_string 
    fLOA  = load_string
    fSCO  = score_string
    fORD  = rk4
    x,y   = bvp_root(fRHS,fORD,fLOA,fSCO,v0,nstep,imode=imode)

    u = y[0,:] # amplitude
    l = y[2,:] # eigenvalue

    ftsz = 10
    sns.set()
    plt.figure(num=1,figsize=(8,8),dpi=100,facecolor='white')
    plt.subplot(211)
    plt.plot(x,u,linestyle='-',color='black',linewidth=1.0)
    plt.xlabel('x',fontsize=ftsz)
    plt.ylabel('u',fontsize=ftsz)
    util.rescaleplot(x,u,plt,0.05)
    plt.tick_params(labelsize=ftsz)
    plt.subplot(212)
    plt.plot(x,l,linestyle='-',color='black',linewidth=1.0)
    plt.xlabel('x',fontsize=ftsz)
    plt.ylabel('$l$',fontsize=ftsz)
    util.rescaleplot(x,l,plt,0.05)
    plt.tick_params(labelsize=ftsz)

    plt.show()



#=======================================
main()

