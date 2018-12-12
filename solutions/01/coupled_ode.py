#!/usr/bin/python
#==============================================================
# Implicit integrator schemes for stiff coupled ODEs.
# Containing the functions
#
#==============================================================
# required libraries
import argparse	                 # allows us to deal with arguments to main()
from argparse import RawTextHelpFormatter
import numpy as np               # numerical routines (arrays, math functions etc)
import matplotlib.pyplot as plt  # plotting commands
import p358utilities as util     # for rescaleplot

import ode_integrators as odeint # contains the drivers for ODE integration
import ode_step as step          # the stepper functions

#==============================================================
# function dydx = get_dydx(x,y,dx)
#
# Calculates RHS for stiff test equation
#--------------------------------------------------------------

def get_dydx(x,y,dx):
    dydx    = np.zeros(2)
    dydx[0] =  998.0*y[0]+1998.0*y[1]
    dydx[1] = -999.0*y[0]-1999.0*y[1]
    return dydx

#==============================================================
#==============================================================
# main
# 
# parameters:
#   stepper: ODE solver (see help function below)   
#

def main():

    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument("stepper",type=str,default='euler',
                        help="stepping function:\n"
                             "   euler    : Euler step\n"
                             "   rk2      : Runge-Kutta 2nd order\n"
                             "   rk4      : Runge-Kutta 4th order\n"
                             "   backeuler: implicit Euler step\n")
    args   = parser.parse_args()
    if (args.stepper == "euler"):
        fORD = step.euler
    elif (args.stepper == "rk2"):
        fORD = step.rk2
    elif (args.stepper == "rk4"):
        fORD = step.rk4
    elif (args.stepper == "backeuler"):
        fORD = step.backeuler
    else:
        raise Exception("invalid stepper %s" % (args.stepper))

    # initialization
    nstep    = 100
    x0       = 0.0
    x1       = 1.0
    y0       = np.array([1.0,0.0])
    fINT     = odeint.ode_ivp
    fRHS     = get_dydx
    fBVP     = 0

    # solve
    x,y,it   = fINT(fRHS,fORD,fBVP,x0,y0,x1,nstep)

    # Your code should show three plots in one window:
    # (1) the dependent variables y[0,:] and y[1,:] against x
    # (2) the residual y[0,:]-u and y[1,:]-v against x, see homework sheet
    # (3) the number of iterations  against x.
    # from here ??????
    xmin = np.min(x)
    xmax = np.max(x)
    ymin = np.min(y)
    ymax = np.max(y)
    ftsz = 10
    fig  = plt.figure(num=1,figsize=(6,8),dpi=100,facecolor='white')
    u    = 2.0*np.exp(-x)-np.exp(-1e3*x)
    v    = -np.exp(-x)+np.exp(-1e3*x)
    diff = np.zeros((2,x.size))
    diff[0,:] = y[0]-u
    diff[1,:] = y[1]-v

    plt.subplot(311)
    plt.plot(x,y[0,:],linestyle='-',color='blue',linewidth=1.0,label='u')
    plt.plot(x,u,linestyle='--',color='blue',linewidth=1.0,label='u [analytic]')
    plt.plot(x,y[1,:],linestyle='-',color='red',linewidth=1.0,label='v')
    plt.plot(x,v,linestyle='--',color='red',linewidth=1.0,label='v [analytic]')
    plt.xlabel('x',fontsize=ftsz)
    plt.ylabel('y',fontsize=ftsz)
    plt.legend(fontsize=ftsz)
    util.rescaleplot(x,y,plt,0.05)
    plt.tick_params(labelsize=ftsz)
    plt.subplot(312)
    plt.plot(x,diff[0,:],linestyle='-',color='blue',linewidth=1.0,label='residual u')
    plt.plot(x,diff[1,:],linestyle='-',color='red',linewidth=1.0,label='residual v')
    plt.xlabel('x',fontsize=ftsz)
    plt.ylabel('y-y[analytic]',fontsize=ftsz)
    plt.legend(fontsize=ftsz)
    util.rescaleplot(x,diff,plt,0.05)
    plt.tick_params(labelsize=ftsz)
    plt.subplot(313)
    plt.plot(x[1:x.size],np.log10(it[1:x.size]),linestyle='-',color='black',linewidth=1.0,label='iterations')
    plt.xlabel('x',fontsize=ftsz)
    plt.ylabel('log #',fontsize=ftsz)
    plt.legend(fontsize=ftsz)
    util.rescaleplot(x[1:x.size],np.log10(it[1:x.size]),plt,0.05)
    plt.tick_params(labelsize=ftsz)

    plt.show()
    # to here ??????

#==============================================================

main()


