#=============================================================
# Test bank for ODE integrators: Kepler problem.
#  
# Contains the functions:
#   get_planetdata  : returns basic information about planets
#   set_odepar      : setting global variables for get_dydx()
#   get_odepar      : getting global variables for get_dydx()
#   get_dydx        : the RHS of the ODEs
#   ode_init        : initializes ODE problem, setting functions and initial values
#   ode_check       : performs tests on results (plots, sanity checks)
#   main            : calls the rest. 
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
# function mass,eps,rap,vorb,torb = get_planetdata(which)
#
# Returns planetary orbit data
#
# input:
#   which: integer array with elements between 1 and 8, with 1: Mercury...8: Neptune
# output:
#   mass: planet mass in kg
#   eps : eccentricity
#   rap : aphelion distance (in km)
#   vorb: aphelion velocity (in km/s)
#   torb: orbital period (in years)
#---------------------------------------------------------------

def get_planetdata(which):
    nplanets             = len(which)
    mass                 = np.array([1.989e30,3.3011e23,4.8675e24,5.972e24,6.41e23,1.89819e27,5.6834e26,8.6813e25,1.02413e26]) 
    eps                  = np.array([0.0,0.205,0.0067,0.0167,0.0934,0.0489,0.0565,0.0457,0.0113])
    rap                  = np.array([0.0,6.9816e10,1.0894e11,1.52139e11,2.49232432e11,8.1662e11,1.5145e12,3.00362e12,4.54567e12])
    vorb                 = np.array([0.0,3.87e4,3.479e4,2.929e4,2.197e4,1.244e4,9.09e3,6.49e3,5.37e3])
    yrorb                = np.array([0.0,0.241,0.615,1.0,1.881,1.1857e1,2.9424e1,8.3749e1,1.6373e2])
    rmass                = np.zeros(nplanets+1)
    reps                 = np.zeros(nplanets+1)
    rrap                 = np.zeros(nplanets+1)
    rvorb                = np.zeros(nplanets+1)
    ryrorb               = np.zeros(nplanets+1)
    rmass [0]            = mass [0]
    rmass [1:nplanets+1] = mass [which]
    reps  [1:nplanets+1] = eps  [which]
    rrap  [1:nplanets+1] = rap  [which]
    rvorb [1:nplanets+1] = vorb [which]
    ryrorb[1:nplanets+1] = yrorb[which]
    return rmass,reps,rrap,rvorb,ryrorb

#==============================================================
# function dydx = get_dydx(x,y,dx)
#
# Calculates ODE RHS for Kepler problem via direct summation.
#
# input: 
#   x,y    : position and values for RHS.
#            If we have three bodies, y has the shape
#            [x1,y1,x2,y2,x3,y3,vx1,vy1,vx2,vy2,vx3,vy3], for bodies 1,2,3.
#            with the cartesian positions (x,y) and their
#            velocities (v_x,v_y).
# global:
#   G      : grav constant (par[0])
#   masses : masses of bodies (par[1:npar])
#   
# output:
#   dydx   : vector of results as in y'=f(x,y)
#--------------------------------------------------------------

def get_dydx(x, y, dx):
    nbodies    = y.size // 4 # per body, we have four variables
    par        = get_odepar()
    npar       = par.size
    gnewton    = par[0]
    masses     = par[1:npar]
    dydx       = np.zeros(y.size)

    # The function needs to accomplish the following:
    # (1) Set the time derivatives of the positions for all objects.
    # (2) Calculate the gravitational force on each object.
    # (3) Set velocity derivatives to the resulting accelerations.
    #?????????????? from here
    indx        = 2*np.arange(nbodies)
    indy        = 2*np.arange(nbodies)+1
    indvx       = 2*np.arange(nbodies)+2*nbodies
    indvy       = 2*np.arange(nbodies)+2*nbodies+1
    dydx[indx]  = y[indvx] # x'=v
    dydx[indy]  = y[indvy]
    for k in range(nbodies):
        gravx = 0.0
        gravy = 0.0
        for j in range(nbodies):
            if (k != j):
                dx    = y[indx[k]]-y[indx[j]]
                dy    = y[indy[k]]-y[indy[j]]
                R3    = np.power(dx*dx+dy*dy,1.5)
                gravx = gravx - gnewton*masses[j]*dx/R3
                gravy = gravy - gnewton*masses[j]*dy/R3
        dydx[indvx[k]] = gravx
        dydx[indvy[k]] = gravy
    return dydx


    #x is time
    # for i in range (nbodies):
    #     dydx[i]   = y[2*nbodies+i]         #x'
    #     dydx[i+1] = y[2*nbodies+1+i]       #y'
    

    # for i in range (nbodies):
    #     sumx = 0
    #     sumy = 0
    #     for j in range(nbodies):
    #         if(i != j):
    #             sumx += masses[j] * (y[i] - y[j]) / abs(y[i] - y[j])**3
    #             sumy += masses[j] * (y[i+1] - y[j+1]) / abs(y[i+1] - y[j+1])**3

    #     dydx[2*nbodies + i] = sumx * -gnewton       #x''
    #     dydx[2*nbodies + i + 1] = sumy * -gnewton   #y''

    #?????????????? to here
    # return dydx

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

    print('[ode_init]: initializing Kepler problem')
    # We set the initial positions, assuming orbit starts at aphel.
    # Units are different here. We set G=1, L=1AU, t=1yr. This results
    # a set scale for the mass, as below.
    AU      = 1.495979e11               # AU in meters
    year    = 3.6e3*3.65e2*2.4e1        # year in seconds
    mass,eps,r_aphel,v_orb,yr_orb = get_planetdata(np.array([3]))
    gnewton = 6.67408e-11
    uLeng   = AU
    uTime   = year
    uVelo   = uLeng/uTime
    uAcce   = uVelo/uTime
    uMass   = uAcce*uLeng*uLeng/gnewton
    masscu  = mass/uMass 
    rapcu   = r_aphel/uLeng
    velcu   = v_orb/uVelo
    # Set initial conditions. All objects are aligned along x-axis, with planets to positive x, sun to negative x.
    rapcu[0]= -np.sum(masscu*rapcu)/masscu[0]
    velcu[0]= -np.sum(masscu*velcu)/masscu[0]

    nstepyr = 100                          # number of steps per year
    nyears  = int(np.ceil(np.max(yr_orb)))
    x0      = 0.0                          # starting at t=0
    x1      = nyears*year/uTime            # end time in years
    nstep   = nyears*nstepyr               # thus, each year is resolved by nstepyr integration steps
    nbodies = mass.size                    # number of objects
    y0      = np.zeros(4*nbodies)
    par     = np.zeros(nbodies+1)          # number of parameters
    par[0]  = 1.0
    for k in range(nbodies):               # fill initial condition array and parameter array
        y0[2*k]             = rapcu[k]
        y0[2*k+1]           = 0.0
        y0[2*(nbodies+k)]   = 0.0
        y0[2*(nbodies+k)+1] = velcu[k]
        par[k+1]            = masscu[k]
    fINT    = odeint.ode_ivp
    fRHS    = get_dydx

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
    
    # for the direct Kepler problem, we check for energy and angular momentum conservation,
    # and for the center-of-mass position and velocity
    color   = ['black','green','cyan','blue','red','black','black','black','black']
    n       = x.size
    par     = get_odepar()
    npar    = par.size
    nbodies = par.size-1
    gnewton = par[0]
    masses  = par[1:npar]
    Egrav   = np.zeros(n)
    indx    = 2*np.arange(nbodies)
    indy    = 2*np.arange(nbodies)+1
    indvx   = 2*np.arange(nbodies)+2*nbodies
    indvy   = 2*np.arange(nbodies)+2*nbodies+1
    E       = np.zeros(n) # total energy
    Lphi    = np.zeros(n) # angular momentum
    R       = np.sqrt(np.power(y[indx[0],:]-y[indx[1],:],2)+np.power(y[indy[0],:]-y[indy[1],:],2))
    Rs      = np.zeros(n) # center of mass position
    vs      = np.zeros(n) # center of mass velocity
    for k in range(n):
        E[k]    = 0.5*np.sum(masses*(np.power(y[indvx,k],2)+np.power(y[indvy,k],2)))
        Lphi[k] = np.sum(masses*(y[indx,k]*y[indvy,k]-y[indy,k]*y[indvx,k]))
        Rsx     = np.sum(masses*y[indx,k])/np.sum(masses)
        Rsy     = np.sum(masses*y[indy,k])/np.sum(masses)
        vsx     = np.sum(masses*y[indvx,k])/np.sum(masses)
        vsy     = np.sum(masses*y[indvy,k])/np.sum(masses)
        Rs[k]   = np.sqrt(Rsx*Rsx+Rsy*Rsy)
        vs[k]   = np.sqrt(vsx*vsx+vsy*vsy)
    for j in range(nbodies):
        for i in range(j): # preventing double summation. Still O(N^2) though.
            dx    = y[indx[j],:]-y[indx[i],:]
            dy    = y[indy[j],:]-y[indy[i],:]
            Rt    = np.sqrt(dx*dx+dy*dy)
            Egrav = Egrav - gnewton*masses[i]*masses[j]/Rt 
    E       = E + Egrav 
    E       = E/E[0]
    Lphi    = Lphi/Lphi[0]
    for k in range(n):
        print('k=%7i t=%13.5e E/E0=%20.12e L/L0=%20.12e Rs=%10.2e vs=%10.2e' 
              % (k,x[k],E[k],Lphi[k],Rs[k],vs[k]))
    Eplot   = E-1.0
    Lplot   = Lphi-1.0
    logE    = np.log10(np.abs(Eplot-1.0))
    logL    = np.log10(np.abs(Lplot-1.0))
    # now plot everything
    # (1) the orbits
    xmin    = np.min(y[indx,:])
    xmax    = np.max(y[indx,:])
    ymin    = np.min(y[indy,:])
    ymax    = np.max(y[indy,:])
    plt.figure(num=1,figsize=(6,6),dpi=100,facecolor='white')
    plt.subplot(111)
    plt.xlim(1.05*xmin,1.05*xmax)
    plt.ylim(1.05*ymin,1.05*ymax)
    for k in range(nbodies):
        plt.plot(y[indx[k],:],y[indy[k],:],color=color[k],linewidth=1.0,linestyle='-')
    plt.axes().set_aspect('equal')
    plt.xlabel('x [AU]')
    plt.ylabel('y [AU]')
    # (2) the checks (total energy and angular momentum)
    plt.figure(num=2,dpi=100,facecolor='white')
    plt.subplot(211)
    plt.plot(x,Eplot,linestyle='-',color='black',linewidth=1.0,label='E')
    plt.plot(x,Lplot,linestyle='--',color='black',linewidth=1.0,label='L')
    plt.xlabel('t [yr]')
    plt.ylabel('E/E0-1, L/L0-1')
    plt.legend()
    plt.subplot(212)
    plt.plot(x,logE,linestyle='-',color='black',linewidth=1.0,label='E')
    plt.plot(x,logL,linestyle='--',color='black',linewidth=1.0,label='L')
    plt.xlabel('t [yr]')
    plt.ylabel('log|E/E0-1|, log|L/L0-1|')
    plt.legend()
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
