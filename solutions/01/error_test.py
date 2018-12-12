#==============================================================
# Calculates the integration errors on y' = exp(y-2x)+2
#
#==============================================================
# required libraries
import numpy as np               # numerical routines (arrays, math functions etc)
import matplotlib.pyplot as plt  # plotting commands
import p358utilities as util     # for rescaleplot

import ode_integrators as odeint # contains the drivers for ODE integration.
import ode_step as step          # contains the ODE single step functions.

#==============================================================
# function dydx = get_dydx(x,y,dx)
#
# Calculates RHS for error test function
#
# input: 
#   x,y    : 
#
# global:
#   -
# output:
#   dydx    : vector of results as in y'=f(x,y)
#--------------------------------------------------------------

def get_dydx(x, y, dx):
    dydx = np.exp(y[0] - 2.0 * x) + 2.0
    return dydx

#==============================================================
# main
#==============================================================
def main():

#   the function should accomplish the following:
#   Test the 3 fixed stepsize integrators euler,rk2,rk4 by calculating their 
#   cumulative integration errors for increasing step sizes
#   on the function y' = exp(y(x)-2*x)+2. This is given in the function "get_dydx" above.
#   Use the integration interval x=[0,1], and the initial 
#   condition y[0] = -ln(2).
#   (1) define an array containing the number of steps you want to test. Logarithmic spacing
#       (in decades) might be useful.
#   (2) loop through the integrators and the step numbers, calculate the 
#       integral and store the error. You'll need the analytical solution at x=1,       
#       see homework assignment.
#   (3) Plot the errors against the stepsize as log-log plot, and print out the slopes.

    fINT  = odeint.ode_ivp                  # use the initial-value problem driver
    fORD  = [step.euler,step.rk2,step.rk4]  # list of stepper functions to be run
    fRHS  = get_dydx                        # the RHS (derivative) for our test ODE
    fBVP  = 0                               # unused for this problem

    #????????????????? from here
    x0    = 0.0
    x1    = 1.0
    y0    = np.array([-np.log(2.0)])
    nstep = np.array([10,30,100,300,1000,3000,10000,30000,100000])    
    err   = np.zeros((nstep.size,len(fORD)))
    for j in range(len(fORD)):
        for i in range(nstep.size):
            x,y,it   = fINT(fRHS,fORD[j],fBVP,x0,y0,x1,nstep[i]) 
            err[i,j] = np.abs(y[0,nstep[i]]-2.0)
            print('[error_test]: j=%3d i=%3d f=%10s nstep=%7d err=%13.5e' % (j,i,fORD[j].__name__,nstep[i],err[i,j]))

    lnstep = np.log10(nstep)
    lerr   = np.log10(err)

    ftsz = 10
    plt.figure(num=1,figsize=(8,8),dpi=100,facecolor='white')

#   plot the error against step number, and print the slopes.
    plt.subplot(111)
    for j in range(len(fORD)):
        plt.plot(lnstep,lerr[:,j],linestyle='-',linewidth=1.0)
        print('[error_test]: j=%3d f=%10s slope=%10.2e' % (j,fORD[j].__name__,(lerr[1,j]-lerr[0,j])/(lnstep[1]-lnstep[0])))
    plt.xlabel('log steps',fontsize=ftsz)
    plt.ylabel('log err',fontsize=ftsz)
    util.rescaleplot(lnstep,lerr,plt,0.05)
    plt.tick_params(labelsize=ftsz)

    plt.show()

    #????????????????? to here


#==============================================================

main()


