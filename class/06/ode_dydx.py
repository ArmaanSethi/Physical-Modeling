import numpy as np
import globalvar

#==============================================================
# function dydx = lunarlanding(x,y,dx)
#
# Calculates ODE RHS for lunar landing.
#
# input: 
#   x,y    : position and values for RHS
#            Assumes y to have the shape (z,v,m,k), 
#            with z the position above the surface,
#                 v the velocity along z,
#                 m the mass of the lander.
#                 k the throttle fraction
#  dx      : step size. Not needed for fixed-step, but for consistency
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

def lunarlanding(x,y,dx):
    par     =  globalvar.get_odepar()
    dydx    =  np.zeros(4)
    thrmax  =  par[0]
    mode    =  par[1]
    g       =  par[2]
    Vnozz   =  par[3]
    mship   =  par[4]
    z       =  y[0]
    vz      =  y[1]
    fuel    =  y[2]
    thrfrac =  y[3]

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
    elif (int(mode)==1): # PID controller: constant reference velocity
        vref   = par[5]
        kp     = par[6]
        ki     = par[7]
        kd     = par[8]
        vintdt = z-500.0
        denom  = (1.0+kd*thrmax/mtot)
        numer  = kp*(vref-vz)+ki*(vref*x-vintdt)+kd*g
        ddenom = -kd*thrmax*dfdt/mtot**2 
        dnumer = -kp*dvzdt+ki*(vref-vz)
        dthrdt = (denom*dnumer-numer*ddenom)/denom**2

        if (dthrdt < 0.0): # limit adjustment of valve to prevent overshoots
            dthrdt = np.max(np.array([dthrdt,-thrfrac/dx])) 
        else: 
            dthrdt = np.min(np.array([dthrdt,(1.0-thrfrac)/dx]))
        # need to check for fuel again to make sure throttle change is 0.
        if (fuel <= 0.0):
            dthrdt = 0.0
    elif (int(mode)==2): # PID controller with variable velocity
        kp     = par[6]
        ki     = par[7]
        kd     = par[8]
        vref   = -5.0+(1.0-z/500.0)*4.9
        vintdt = z-500.0
        denom  = (1.0+kd*thrmax/mtot)
        numer  = kp*(vref-vz)+ki*(vref*x-vintdt)+kd*(-4.9/500.0*vz+g)
        ddenom = -kd*thrmax*dfdt/mtot**2
        dnumer = -kp*dvzdt+ki*(vref-vz)
        dthrdt = (denom*dnumer-numer*ddenom)/denom**2

        if (dthrdt < 0.0): # limit adjustment of valve to prevent overshoots
            dthrdt = np.max(np.array([dthrdt,-thrfrac/dx]))
        else:
            dthrdt = np.min(np.array([dthrdt,(1.0-thrfrac)/dx]))
        # need to check for fuel again to make sure throttle change is 0.
        if (fuel <= 0.0):
            dthrdt = 0.0
    else: 
        raise Exception("[dydx.lunarlanding]: invalid imode %i" % (i))

    dydx[0] =  dzdt
    dydx[1] =  dvzdt
    dydx[2] =  dfdt
    dydx[3] =  dthrdt
    return dydx

