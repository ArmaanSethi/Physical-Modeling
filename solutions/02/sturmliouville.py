# Solver for a Sturm-Liouville problem (finite string Eigenmodes)
# This can be generalized to any BVP, as long as it's solvable
# by a shooting method.
import numpy as np
import matplotlib.pyplot as plt
import p358utilities as util

#========================================
# Density function for string
# Returns the mass density at given position x.
# Here, we make use of optional arguments.
def rho(x,**kwargs):
    for key in kwargs:
        if (key=='imode'):
            imode = kwargs[key]
    if (imode == 0):   # flat
        rho = 1.0
    elif (imode == 1): # cosine
        rho = 1.0-0.9*np.cos(2.0*np.pi*x-np.pi)
    elif (imode == 2): # linear
        rho = 1.0+x*10.0
    elif (imode == 3): # tanh
        rho = 1.0+50.0*(1.0+np.tanh((x-0.5)/0.001))
    elif (imode == 4): # exponential
        rho = 1.0+np.exp(10.0*x)
    return rho

#========================================
# fRHS for the (non-)uniform string
# Should return an array dydx containing three
# elements corresponding to y'(x), y''(x), and lambda'(x).
def dydx_string(x,y,**kwargs):
    dydx    = np.zeros(3)
    dydx[0] = y[1]      # displacement of string
    dydx[1] = -np.power(np.pi*y[2],2)*rho(x,**kwargs)*y[0] # slope
    dydx[2] = 0.0       # eigenvalue (number of waves)
    return dydx

#========================================
# fLOA for the string problem
# This is the (boundary) loading function.
# Should return the integration boundaries x0,x1, and
# the initial values y(x=0). Therefore y must be
# an array with three elements. 
# The function takes an argument v which in our case
# is just the eigenvalue lambda.
# Note that some of the initial values in y can
# (and should!) depend on lambda.
def load_string(v):
    x0 = 0.0
    x1 = 1.0
    y  = np.array([0.0,v[0]*np.pi,v[0]])
    return x0,x1,y

#========================================
# fSCO for the string problem
# This is the scoring function. Should return
# the function that needs to be zeroed by the
# root finder. In our case, that's just y(1). 
# The function takes the arguments x and y, where
# y is an array with three elements.
def score_string(x,y):
    return y[0] # displacement should be zero.

#========================================
# Single rk4 step.
def rk4(fRHS,x0,y0,dx,**kwargs):
    k1 = dx*fRHS(x0       ,y0       ,**kwargs)
    k2 = dx*fRHS(x0+0.5*dx,y0+0.5*k1,**kwargs)
    k3 = dx*fRHS(x0+0.5*dx,y0+0.5*k2,**kwargs)
    k4 = dx*fRHS(x0+    dx,y0+    k3,**kwargs)
    y  = y0+(k1+2.0*(k2+k3)+k4)/6.0
    return y    

#========================================
# ODE IVP driver.
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
    x0,x1,y0 = fLOA(v) # set lower boundary values
    x,y      = ode_ivp(fRHS,fORD,x0,x1,y0,nstep,**kwargs)
    score    = fSCO(x[x.size-1],y[:,x.size-1])
    return score # this should be zero, and thus can be directly used.

#=======================================
# The rootfinder.
# The function pointers are problem-specific (see main()). 
# v0 is the initial guess for the eigenvalue (in our case).
# Should return x,y, so that the solution can be plotted.
def bvp_root(fRHS,fORD,fLOA,fSCO,v0,nstep,**kwargs):
    # try to bracket first
    fac    = 1.2
    vlo    = v0
    vhi    = vlo*fac
    flo    = bvp_shoot(fRHS,fORD,fLOA,fSCO,vlo,nstep,**kwargs)
    fhi    = bvp_shoot(fRHS,fORD,fLOA,fSCO,vhi,nstep,**kwargs)
    while (flo*fhi > 0.0):
        vhi   = vhi*fac
        fhi   = bvp_shoot(fRHS,fORD,fLOA,fSCO,vhi,nstep,**kwargs)
    # now do bisection
    i = 0
    while (np.abs(vhi-vlo)/(vlo+vhi) > 1e-5):
        vmd    = 0.5*(vlo+vhi)
        fmd    = bvp_shoot(fRHS,fORD,fLOA,fSCO,vmd,nstep,**kwargs)
        print('%5i %13.5e %13.5e' % (i,vmd,fmd))
        if (flo*fmd > 0.0):
            vlo = vmd
            flo = fmd
        else:
            vhi = vmd
        i = i+1
    x0,x1,y0 = fLOA(vmd)
    x,y      = ode_ivp(fRHS,fORD,x0,x1,y0,nstep,**kwargs)
    return x,y

#=======================================
def main():

    nstep = 500
    imode = 4
    v0    = np.array([1.0])
    fRHS  = dydx_string
    fLOA  = load_string
    fSCO  = score_string
    fORD  = rk4
    x,y   = bvp_root(fRHS,fORD,fLOA,fSCO,v0,nstep,imode=imode)

    u = y[0,:]
    l = y[2,:]

    ftsz = 10
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

