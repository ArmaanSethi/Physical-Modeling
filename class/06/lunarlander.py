#==============================================================
# Test bank for ODE integrators. 
# Containing the functions
#   get_planetdata  : returns basic information about planets
#   get_h2rates     : returns reaction rates for H2 formation network.
#   get_h2times     : returns the corresponding timescales for reaction rates
#   get_rooth2func  : root function for root finder to get H2 equilibrium abundance
#   get_h2eq        : root finder to get H2 equilibrium abundance
#
#   ode_init        : initializes ODE problem, setting functions and initial values
#   ode_check       : performs tests on results (plots, sanity checks)
#   main            : calls the rest. Needs argument iprob. Calling sequence e.g.: ode_test.py 10
#
#==============================================================
# required libraries
import argparse	                 # allows us to deal with arguments to main()
from argparse import RawTextHelpFormatter
import numpy as np               # numerical routines (arrays, math functions etc)
import math
import matplotlib.pyplot as plt  # plotting commands
import p358utilities as util     # for rescaleplot
import globalvar                 # interface for global variables

import ode_integrators as odeint # contains the drivers for ODE integration
import ode_step as step          # the stepper functions
import ode_dydx as dydx          # contains the RHS functions for selected problems.
import ode_bvp  as bvp           # contains the boundary value functions for selected problems.
import ode_jac  as jac           # contains definitions of Jacobians for various problems

#==============================================================
# functions
#==============================================================
# function fRHS,x0,y0,x1 = ode_init(stepper)
#
# Initializes derivative function, parameters, and initial conditions
# for ODE integration.
#
# input: 
#   stepper: euler
#            rk2
#            rk4
#            rk45
# output:
#   fINT   : function handle for integrator (problem) type: initial or boundary value problem (ode_ivp or ode_bvp)
#   fORD   : function handle for integrator order (euler, rk2, rk4, rk45). 
#            Note: symplectic integrators require euler.
#   fRHS   : function handle for RHS of ODE. Needs to return vector dydx of same size as y0.
#   fBVP   : function handle for boundary values in case fINT == ode_bvp.
#   fJAC   : functino handle for Jacobian required for implicit integration. Default = None
#   x0     : starting x
#   y0     : starting y(x0)
#   x1     : end x
#--------------------------------------------------------------

def ode_init(imode):

    fBVP = None # default: IVP, but see below.
    fJAC = None # default: explicit integrators (don't need Jacobian)
    eps  = None  # default: fixed stepsize integrators don't need eps
    fORD = step.rk4
  
    mode = float(imode)

    thrmax  = 2.5e3 # thrmax[N]      : maximum thrust (par[0])
    g       = 1.62  # [m s^(-2)]     : gravitational acceleration (par[2])
    Vnozz   = 2.5e3 # [m s^(-1)      : nozzle gas speed (par[3])
    mship   = 8e2   # [kg]           : ship mass (without fuel)
    # initial conditions
    z0      = 5e2   # z0[m]          : altitude above surface
    v0      = -5.0  # v0[m/s]        : starting velocity
    f0      = 2e2   # f0[kg]         : initial fuel mass 
    thr0    = 0.0   # thr0 [0,1]     : throttle fraction
    # PID control parameter:
    vref    = -0.2  # vref [m s^(-1)]: reference velocity for PID controller
    kp      = 0.0
    ki      = 0.0
    kd      = 0.0 
    par     = np.array([thrmax,mode,g,Vnozz,mship,vref,kp,ki,kd]) 

    nstep   = 100
    x0      = 0.0   # start time (in s)
    x1      = 20.0  # end time (in s). This is just a guess.
    y0      = np.array([z0,v0,f0,thr0]) 
    fINT    = odeint.ode_bvp                   # function handle: IVP or BVP
    fRHS    = dydx.lunarlanding         # function handle: RHS of ODE
    fBVP    = bvp.lunarlanding          # function handle: BVP values
    eps     = 1e-8

    globalvar.set_odepar(par)
    return fINT,fORD,fRHS,fBVP,fJAC,x0,y0,x1,nstep,eps

#==============================================================
# function ode_check(x,y,iprob)
#
# Performs problem-dependent tests on results.
#
# input: 
#   iinteg   : integrator type
#   x    :  independent variable
#   y    :  integration result
#   it   :  number of substeps used. Only meaningful for RK45 (iinteg = 4). 
#--------------------------------------------------------------

def ode_check(x,y,it):
    
    n    = x.size
    par  = globalvar.get_odepar()
    z    = y[0,:] # altitude above ground
    vz   = y[1,:] # vertical velocity
    f    = y[2,:] # fuel mass
    thr  = y[3,:] # throttle

    accG = np.zeros(n) # acceleration in units of g
    for k in range(n):
        accG[k] = ((dydx.lunarlanding(x[k],y[:,k],n/(x[n-1]-x[0])))[1])/9.81 # acceleration

    ftsz = 10
    plt.figure(num=1,figsize=(8,8),dpi=100,facecolor='white')

    plt.subplot(321)
    plt.plot(x,z,linestyle='-',color='black',linewidth=1.0)
    plt.xlabel('t [s]',fontsize=ftsz)
    plt.ylabel('z(t) [m]',fontsize=ftsz)
    util.rescaleplot(x,z,plt,0.05)
    plt.tick_params(labelsize=ftsz)

    plt.subplot(322)
    plt.plot(x,vz,linestyle='-',color='black',linewidth=1.0)
    plt.xlabel('t [s]',fontsize=ftsz)
    plt.ylabel('v$_z$ [m s$^{-1}$]',fontsize=ftsz)
    util.rescaleplot(x,vz,plt,0.05)
    plt.tick_params(labelsize=ftsz)

    plt.subplot(323)
    plt.plot(x,f,linestyle='-',color='black',linewidth=1.0)
    plt.xlabel('t [s]',fontsize=ftsz)
    plt.ylabel('fuel [kg]',fontsize=ftsz)
    util.rescaleplot(x,f,plt,0.05)
    plt.tick_params(labelsize=ftsz)

    plt.subplot(324)
    plt.plot(x,thr,linestyle='-',color='black',linewidth=1.0)
    plt.xlabel('t [s]',fontsize=ftsz)
    plt.ylabel('throttle',fontsize=ftsz)
    util.rescaleplot(x,thr,plt,0.05)
    plt.tick_params(labelsize=ftsz)

    plt.subplot(325)
    plt.plot(x,accG,linestyle='-',color='black',linewidth=1.0)
    plt.xlabel('t [s]',fontsize=ftsz)
    plt.ylabel('acc/G',fontsize=ftsz)
    util.rescaleplot(x,accG,plt,0.05)
    plt.tick_params(labelsize=ftsz)

    plt.tight_layout()

    plt.show()
        
#==============================================================
#==============================================================
# main
# 

def main():

    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument("imode",type=int,default=0,
                        help="mode of operation:\n" 
                             "   0: no control\n"
                             "   1: PID controller (constant velocity)\n"
                             "   2: PID controller (variable velocity)")

    args    = parser.parse_args()
    imode   = args.imode

    fINT,fORD,fRHS,fBVP,fJAC,x0,y0,x1,nstep,eps = ode_init(imode)
    x,y,it                                      = fINT(fRHS,fORD,fBVP,x0,y0,x1,nstep,fJAC=fJAC,eps=eps)

    ode_check(x,y,it)

#==============================================================

main()


