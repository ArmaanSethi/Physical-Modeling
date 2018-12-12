#=============================================================
# Test bank for ODE integrators: Lunar lander.
#  
# Contains the functions:
#
#   set_odepar      : setting global variables for get_dydx()
#   get_odepar      : getting global variables for get_dydx()
#   get_dydx        : the RHS of the ODEs
#   get_bvp         : the boundary values (specific to lunar lander!)
#   ode_init        : initializes ODE problem, setting functions and initial values
#   ode_check       : performs tests on results (plots, sanity checks)
#   main            : calls the rest. Needs argument iprob. Calling sequence e.g.: ode_test.py 10 
#
# Arguments:
#  --stepper [euler,rk2,rk4]
#=============================================================
# required libraries
import argparse	                 # allows us to deal with arguments to main()
from argparse import RawTextHelpFormatter
import numpy as np               # numerical routines (arrays, math functions etc)
import matplotlib.pyplot as plt  # plotting commands
import p358utilities as util     # for rescaleplot

import ode_integrators as odeint # contains the drivers for ODE integration
import ode_step as step          # the stepper functions

#=============================================================
# interface for global variables
#=============================================================
# function set_odepar()
def set_odepar(par):
    global odepar
    odepar = par

#=============================================================
# function get_odepar()
def get_odepar():
    global odepar
    return odepar

#==============================================================
# function dydx = get_dydx(x,y,dx)
#
# Calculates ODE RHS for lunar lander.
#
# input: 
#   x,y    : position and values for RHS
#            Assumes y to have the shape (z,v,m,k), 
#            with z     the position above the surface,
#                 vz    the velocity along z,
#                 fuel  the fuel amount, 
#                 k     the throttle fraction (called "thrfrac" below).
#  dx      : step size. Not needed for fixed-step, but for consistency
#            with adaptive stepsize integrators
# global:
#   globalvar.get_odepar needs to return a parameter vector with
#   the elements
#   thrmax  : maximum thrust (par[0])
#   mode    : 0,1,2 for constant throttle, PID, step function, (par[1])
#   g       : gravitational acceleration (par[2])
#   Vnozz   : nozzle gas speed (par[3])
#   mship   : mass of lander (without fuel)
#   
# output:
#   dydx   : vector of results as in y'=f(x,y)
#
# Note:
#   The throttle fraction will be set to 0 if m <= 0.
#
#--------------------------------------------------------------

def get_dydx(x, y, dx):
    par     = get_odepar()
    dydx    = np.zeros(y.size) # derivatives of z,vz,fuel,thrfrac, to be returned
    thrmax  = par[0]
    mode    = int(par[1])
    g       = par[2]
    Vnozz   = par[3]
    mship   = par[4]
    z       = y[0]
    vz      = y[1]
    fuel    = y[2]
    thrfrac = y[3]

    # The function has to accomplish the following:
    # (1) make sure that if fuel < 0, the throttle is set to 0. 
    # (2) make sure that the total mass is used.
    # (3) calculate the RHS of the coupled 1st order ODEs, i.e. the
    #     time derivatives for z, vz, fuel, and thrfrac.
    # (4) You can set the derivative for the throttle to zero, i.e. dydx[3] = 0.0
    #????????????????? from here
    # limit throttle value. This has no effect on dthrdt, 
    # but is necessary for mode == 1, and for RHS of ODEs.
    thrfrac = np.max(np.array([0.0,np.min(np.array([1.0,thrfrac]))]))
    # make sure fuel can't go negative
    if (fuel <= 0.0):
        fuel    = 0.0
        thrfrac = 0.0
    mtot    = mship+fuel

    # calculate RHS for all ODEs, bc we'll need them if mode == 1
    dzdt    =  vz
    dvzdt   =  thrmax*thrfrac/mtot - g
    dfdt    = -thrmax*thrfrac/Vnozz 
    if (int(mode)==0): # constant throttle
        dthrdt = 0.0
    elif (int(mode)==1): # PID controller. This is a controller with beta = 0, since drdt == 0
                         # Note: dthrdt contains change of accelerations.
        kp     = 0.10
        ki     = 0.60
        kd     = 0.30 
        vref   = -0.2
        dthrdt = (-kp*dvzdt+ki*(vref-vz)+kd*thrmax*thrfrac*dfdt/(mtot*mtot))/(1.0+kd*thrmax/mtot)
        if (dthrdt < 0.0): # limit adjustment of valve to prevent overshoots
            dthrdt = np.max(np.array([dthrdt,-thrfrac/dx])) 
        else: 
            dthrdt = np.min(np.array([dthrdt,(1.0-thrfrac)/dx]))
        # need to check for fuel again to make sure throttle change is 0.
        if (fuel <= 0.0):
            dthrdt = 0.0
    elif (int(mode)==2): # off/on throttle. Note that we need to calculate the throttle change!
                         # this just via guessing
        if (x > 11.5):
            dthrdt = (1.0-thrfrac)/dx
        else:
            dthrdt = 0.0
        if (fuel <= 0.0):
            dthrdt = 0.0

    dydx[0] =  dzdt
    dydx[1] =  dvzdt
    dydx[2] =  dfdt
    dydx[3] =  dthrdt
    return dydx

    #????????????????? to here
    return dydx

#==============================================================
# function sign = get_bvp(y)
#
# Returns sign of y[0], i.e. of the altitude above ground.
# See ode_test.py, specifically ode_init.
#
#--------------------------------------------------------------

def get_bvp(y):
    return np.sign(y[0])

#==============================================================
# function fRHS,x0,y0,x1 = ode_init(iprob,stepper)
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
#   x0     : starting x
#   y0     : starting y(x0)
#   x1     : end x
#--------------------------------------------------------------

def ode_init(stepper):

    fBVP = 0 # default is IVP, but see below.
    if (stepper == 'euler'):
        fORD = step.euler
    elif (stepper == 'rk2'):
        fORD = step.rk2
    elif (stepper == 'rk4'):
        fORD = step.rk4
    elif (stepper == 'rk45'):
        fORD = step.rk45
    else:
        raise Exception('[ode_init]: invalid stepper value: %s' % (stepper))

    print('[ode_init]: initializing lunar lander')
    thrmax  = 2.5e3 # thrmax[N]      : maximum thrust (par[0])
    mode    = 0.0   # imode          : controls throttle value (see get_dydx)
                    #                  0: constant throttle
                    #                  1: PID controller
                    #                  2: step function
    g       = 1.62  # [m s^(-2)]     : gravitational acceleration (par[2])
    Vnozz   = 2.5e3 # [m s^(-1)      : nozzle gas speed (par[3])
    mship   = 9e2   # [kg]           : ship mass (without fuel)
    # initial conditions
    z0      = 5e2   # z0[m]          : altitude above surface
    v0      = -5.0  # v0[m/s]        : starting velocity
    f0      = 1e2   # f0[kg]         : initial fuel mass 
    thr0    = 0.0   # thr0 [0,1]     : throttle fraction
    par     = np.array([thrmax,mode,g,Vnozz,mship]) 

    nstep   = 100
    x0      = 0.0   # start time (in s)
    x1      = 20.0  # end time (in s). This is just a guess.
    y0      = np.array([z0,v0,f0,thr0]) 
    fINT    = odeint.ode_bvp                   # function handle: IVP or BVP
    fRHS    = get_dydx                         # function handle: RHS of ODE
    fBVP    = get_bvp                          # function handle: BVP values

    set_odepar(par)
    return fINT,fORD,fRHS,fBVP,x0,y0,x1,nstep

#==============================================================
# function ode_check(x,y)
#
# input: 
#   iinteg   : integrator type
#   x    :  independent variable
#   y    :  integration result
#   it   :  number of substeps used. Only meaningful for RK45 (iinteg = 4). 
#--------------------------------------------------------------

def ode_check(x,y,it):
    
    n    = x.size
    par  = get_odepar()
    z    = y[0,:] # altitude above ground
    vz   = y[1,:] # vertical velocity
    f    = y[2,:] # fuel mass
    thr  = y[3,:] # throttle

    accG = np.zeros(n) # acceleration in units of g
    for k in range(n):
        accG[k] = ((get_dydx(x[k],y[:,k],n/(x[n-1]-x[0])))[1])/9.81 # acceleration

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

    plt.show()

#==============================================================
#==============================================================
# main
# 
# parameters:

def main():

    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument("stepper",type=str,default='euler',
                        help="stepping function:\n"
                             "   euler: Euler step\n"
                             "   rk2  : Runge-Kutta 2nd order\n"
                             "   rk4  : Runge-Kutta 4th order\n")

    args   = parser.parse_args()

    stepper= args.stepper

    fINT,fORD,fRHS,fBVP,x0,y0,x1,nstep = ode_init(stepper)
    x,y,it                             = fINT(fRHS,fORD,fBVP,x0,y0,x1,nstep)

    ode_check(x,y,it)

#==============================================================

main()

