#==============================================================
# Test bank for ODE integrators. 
# Containing the functions
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
# function get_h2rates()
# 
# Returns reaction rates for simplified H2 formation network:
#   (1) H + H + G -> H2 + G        H2 formation on grains
#   (2) H2 + g    -> 2H            photodissociation
#   (3) H + CR    -> H+ + e        ionization by cosmic rays
#   (4) H+ + e    -> H + g         recombination
#   A temperature needs to be assumed.
#
# output:
#   reaction rate coefficients 
#--------------------------------------------------------------

def get_h2rates():
    k    = np.zeros(4)
    k[0] = 3e-17                                          # cm^3 s^(-1): B formation on dust grains: A+A+G -> B+G
    k[1] = 3e-15 # s^(-1)     : photodissociation         : B+nu  -> 2A. Assuming N(H2)>10^18. See Glover & Mac Low 2007, 2.2.1
    k[2] = 1e-17 # s^(-1)     : ionization by cosmic rays : A+CR  -> C+e
    k[3] = 8e-12 # cm^3 s^(-1): recombination (rad+grain) : C+e   -> A
    return k

def get_h2times(y):
    Myr  = 1e6*3.65e2*3.6e3*2.4e1
    k    = get_h2rates()
    t    = 1.0/(np.array([k[0]*y[0],k[1],k[2],k[3]*y[2]])*Myr)
    #print t
    return t

#==============================================================
# Function group for H2 equilibrium abundance. Note that
# arguments need to be logarithmic
# D is the total number of H atoms (whether in H, H+ or H2).
# B is the fraction of number of H atoms bound in n(H2), so B = 2*n(H2).

def get_rooth2func(lB,lD):
    k   = get_h2rates()
    k21 = k[1]/k[0]
    k34 = k[2]/k[3]
    D  = np.power(10.0,lD)
    B  = np.power(10.0,lB)*D
    D2 = D*D
    D4 = D2*D2
    f  = B/D - 0.5*(1.0-np.sqrt(k21*B/D2)-np.sqrt(k34*np.sqrt(k21*B/D4)))
    return f

def get_h2eq(D):
    Blo = -8.0
    Bhi = -0.001
    Bmd = 0.5*(Blo+Bhi)
    flo = get_rooth2func(Blo,D)
    fhi = get_rooth2func(Bhi,D)
    fmd = get_rooth2func(Bmd,D)
    while(np.abs((Bhi-Blo)/Bmd) > 1e-15):
        if (flo*fmd < 0.0):
            Bhi = Bmd
        else:
            Blo = Bmd
            flo = fmd
        Bmd = 0.5*(Blo+Bhi)
        fmd = get_rooth2func(Bmd,D)
    k   = get_h2rates()
    D0  = np.power(10.0,D)
    B0  = np.power(10.0,Bmd)*D0 # convert to particles
    k21 = k[1]/k[0]
    k34 = k[2]/k[3]
    A0  = np.sqrt(k21*B0)
    C0  = np.sqrt(k34*A0)
    return np.array([A0,B0,C0])/D0 # these are fractions with respect to D=A+B+C. 

#==============================================================
# function fRHS,x0,y0,x1 = ode_init(iprob,stepper)
#
# Initializes derivative function, parameters, and initial conditions
# for ODE integration.
#
# input: 
#   iprob  : problem number
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
#   fJAC   : function handle for Jacobian required for implicit integration. Default = None
#   x0     : starting x
#   y0     : starting y(x0)
#   x1     : end x
#--------------------------------------------------------------

def ode_init(iprob,stepper):

    fBVP = None # default: IVP, but see below.
    fJAC = None # default: explicit integrators (don't need Jacobian)
    eps  = None  # default: fixed stepsize integrators don't need eps
    if (stepper == 'euler'):
        fORD = step.euler
    elif (stepper == 'rk2'):
        fORD = step.rk2
    elif (stepper == 'rk4'):
        fORD = step.rk4
    elif (stepper == 'rk45'):
        fORD = step.rk45
    elif (stepper == 'rk5'):
        fORD = step.rk5
    elif (stepper == 'eulersi'):
        fORD = step.eulersi
    elif (stepper == 'kr4si'):
        fORD = step.kr4si
    elif (stepper == 'rb34si'):
        fORD = step.rb34si
    else:
        raise Exception('[ode_init]: invalid stepper value: %s' % (stepper))

    if (iprob == "h2formation"): # chemical reaction network
        print('[ode_init]: initializing chemical network')
        # Simplified molecular hydrogen formation network.
        # Species: A = neutral hydrogen, B = molecular hydrogen, C = ionized hydrogen
        # The reaction rate coefficients are:
        k        = get_h2rates()
        Myr      = 1e6*3.56e2*3.6e3*2.4e1 # 10^6 years in seconds
        nstepMyr = 5
        nMyrs    = 20
        x0       = 0.0
        x1       = nMyrs*Myr
        nstep    = nMyrs*nstepMyr
        nspecies = 3
        D0       = 1e3   # total hydrogen atom number density
        B0       = 1e-8    # molecular hydrogen
        C0       = 9e-1    # ionized hydrogen
        A0       = D0-2.0*B0-C0# atomic hydrogen density
        par      = np.zeros(5)
        par[0:4] = k[:]
        par[4]   = D0
        y0       = np.array([A0,B0,C0])
        fINT     = odeint.ode_ivp
        fRHS     = dydx.h2formation
        fJAC     = jac.h2formation
        eps      = 1e-8
    elif (iprob == "doubleexp"):
        print('[ode_init]: initializing stiff ODE test problem (double exponential)')
        par      = np.zeros(1)
        nstep    = 100
        x0       = 0.0
        x1       = 1.0
        y0       = np.array([1.0,0.0])
        fINT     = odeint.ode_ivp
        fRHS     = dydx.doubleexp
        fJAC     = jac.doubleexp
        eps      = 1e-10
    elif (iprob == "enrightpryce"):
        print('[ode_init]: initializing stiff ODE test problem (Enright-Pryce)')
        par      = np.zeros(1)
        nstep    = 100
        x0       = 0.0
        x1       = 50.0
        y0       = np.array([1.0,1.0,0.0])
        fINT     = odeint.ode_ivp
        fRHS     = dydx.enrightpryce
        fJAC     = jac.enrightpryce
        eps      = 1e-4
    else:
        raise Exception('[ode_int]: invalid iprob %i\n' % (iprob))

    globalvar.set_odepar(par)
    return fINT,fORD,fRHS,fBVP,fJAC,x0,y0,x1,nstep,eps

#==============================================================
# function ode_check(x,y,iprob)
#
# Performs problem-dependent tests on results.
#
# input: 
#   iprob:  problem numberj
#   x    :  independent variable
#   y    :  integration result
#   it   :  number of substeps used. Only meaningful for RK45 (iinteg = 4). 
#--------------------------------------------------------------

def ode_check(x,y,it,iprob):

    if (iprob == "h2formation"): # H2 formation
        color = ['black','blue','red']
        label = [u"H",u"H\u2082",u"H\u207A"]
        tlabel= ['k1','k2','k3','k4']
        Myr   = 1e6*3.6e3*3.65e2*2.4e1
        nspec = (y.shape)[0]
        n     = x.size
        par   = globalvar.get_odepar()
        Ds    = par[4]
        ys    = get_h2eq(np.log10(Ds))
        print('Equilibrium fractions: D=%13.5e A=%13.5e B=%13.5e C=%13.5e total=%13.5e' % (Ds,ys[0],ys[1],ys[2],np.sum(ys)))
        ys    = ys*Ds
        th2   = np.zeros((4,n))
        for i in range(n):
            th2[:,i] = get_h2times(y[:,i])
        th2[2,:] = 0.0 # don't show CR ionization (really slow)
        t     = x/Myr # convert time in seconds to Myr
        tmin  = np.min(t)
        tmax  = np.max(t)
        nmin  = np.min(np.array([np.nanmin(y),np.min(ys)]))
        nmax  = np.max(np.array([np.nanmax(y),np.max(ys)]))
        ly    = np.log10(y)
        lnmin = np.min(np.array([np.nanmin(ly),np.min(np.log10(ys))]))
        lnmax = np.max(np.array([np.nanmax(ly),np.max(np.log10(ys))]))

        # Note that the first call to subplot is slightly different, because we need the
        # axes object to reset the y-labels.
        ftsz = 10
        fig  = plt.figure(num=1,figsize=(8,8),dpi=100,facecolor='white')
        ax   = fig.add_subplot(321)
        for i in range(nspec):
            plt.plot(t,y[i,:],linestyle='-',color=color[i],linewidth=1.0,label=label[i])
            plt.plot([tmin,tmax],[ys[i],ys[i]],linestyle='--',color=color[i],linewidth=1.0)
        plt.xlabel('t [Myr]',fontsize=ftsz)
        plt.ylabel('n$_\mathrm{\mathsf{H}}$',fontsize=ftsz)
        plt.legend(fontsize=10)
        util.rescaleplot(t,y,plt,0.05)
        ylabels     = ax.yaxis.get_ticklocs()
        ylabels1    = ylabels[1:ylabels.size-1]
        ylabelstext = ['%4.2f' % (lb/np.max(ylabels1)) for lb in ylabels1]
        plt.yticks(ylabels1,ylabelstext)
        plt.tick_params(labelsize=ftsz)

        plt.subplot(322)
        for i in range(nspec):
            plt.plot(t,ly[i,:],linestyle='-',color=color[i],linewidth=1.0,label=label[i])
            plt.plot(np.array([tmin,tmax]),np.log10(np.array([ys[i],ys[i]])),linestyle='--',color=color[i],linewidth=1.0)
        plt.xlabel('t [Myr]',fontsize=ftsz)
        plt.ylabel('log n [cm$^{-3}$]',fontsize=ftsz)
        util.rescaleplot(t,ly,plt,0.05)
        plt.tick_params(labelsize=ftsz)

        plt.subplot(323)
        for i in range(nspec):
            plt.plot(t,y[i,:]/ys[i],linestyle='-',color=color[i],linewidth=1.0,label=label[i])
        plt.xlabel('t [Myr]',fontsize=ftsz)
        plt.ylabel('n/n$_{eq}$',fontsize=ftsz)
        util.rescaleplot(t,np.array([0,2]),plt,0.05)
        plt.tick_params(labelsize=ftsz)

        plt.subplot(324)
        for i in range(nspec):
            plt.plot(t,ly[i,:]-np.log10(ys[i]),linestyle='-',color=color[i],linewidth=1.0,label=label[i])
        plt.xlabel('t [Myr]',fontsize=ftsz)
        plt.ylabel('log n/n$_{eq}$',fontsize=ftsz)
        util.rescaleplot(t,np.array([-1,1]),plt,0.05)
        plt.tick_params(labelsize=ftsz)

        plt.subplot(325)
        wplot = np.array([0,1,3])
        for i in range(wplot.size):
            plt.plot(t,np.log10(th2[wplot[i],:]),linestyle='-',linewidth=1.0,label=tlabel[wplot[i]])
        plt.xlabel('t [Myr]',fontsize=ftsz)
        plt.ylabel('log reaction time [Myr]',fontsize=ftsz)
        plt.legend(fontsize=ftsz)
        util.rescaleplot(t,np.log10(th2[wplot,:]),plt,0.05)
        plt.tick_params(labelsize=ftsz)

        plt.subplot(326)
        plt.plot(t[1:t.size],np.log10(it[1:t.size]),linestyle='-',color='black',linewidth=1.0,label='iterations')
        plt.xlabel('t [Myr]',fontsize=ftsz)
        plt.ylabel('log #',fontsize=ftsz)
        plt.legend(fontsize=ftsz)
        util.rescaleplot(t,np.log10(it[1:t.size]),plt,0.05)
        plt.tick_params(labelsize=ftsz)

        print("[ode_test]: integration took %6i iterations" % (np.sum(it)))

        plt.tight_layout()
        plt.show()
    
    elif (iprob == "doubleexp"):
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
 
        print("[ode_test]: integration took %6i iterations" % (np.sum(it)))

        plt.tight_layout()
        plt.show()
 
    elif (iprob == "enrightpryce"):
 
        xmin = np.min(x)
        xmax = np.max(x)
        ymin = np.min(y)
        ymax = np.max(y)

        ftsz = 10
        fig  = plt.figure(num=1,figsize=(6,8),dpi=100,facecolor='white')

        plt.subplot(211)
        for i in range(3):
            plt.plot(x,y[i,:],linestyle='-',linewidth=1.0,label='y[%i]' %(i))
        plt.xlabel('x',fontsize=ftsz)
        plt.ylabel('y',fontsize=ftsz)
        plt.legend(fontsize=ftsz)
        util.rescaleplot(x,y,plt,0.05)
        plt.tick_params(labelsize=ftsz)

        plt.subplot(212)
        plt.plot(x[1:x.size],np.log10(it[1:x.size]),linestyle='-',color='black',linewidth=1.0,label='iterations')
        plt.xlabel('x',fontsize=ftsz)
        plt.ylabel('log #',fontsize=ftsz)
        plt.legend(fontsize=ftsz)
        util.rescaleplot(x[1:x.size],np.log10(it[1:x.size]),plt,0.05)
        plt.tick_params(labelsize=ftsz)

        print("[ode_test]: integration took %6i iterations" % (np.sum(it)))

        plt.tight_layout()
        plt.show()

    else:
        raise Exception('[ode_check]: invalid iprob %i\n' % (iprob))

#==============================================================
#==============================================================
# main
# 
# parameters:

def main():

    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument("iprob",type=str,default=0,
                        help="problem:\n" 
                             "   h2formation (chemical reaction network)\n"
                             "   doubleexp (test for stiff ODEs)\n"
                             "   enrightpryce (test for stiff ODEs)\n")
    parser.add_argument("stepper",type=str,default='euler',
                        help="stepping function:\n"
                             "  explicit updates:\n"
                             "    euler   : Euler step\n"
                             "    rk2     : Runge-Kutta 2nd order\n"
                             "    rk4     : Runge-Kutta 4th order\n"
                             "    rk5     : Runge-Kutta 5th order\n"
                             "    rk45    : RK-Fehlberg\n"
                             "  semi-implicit updates:\n"
                             "    eulersi : Euler step\n"
                             "    kr4si   : Kaps-Rentrop 4th order\n"
                             "    rb34si  : Rosenbrock 3rd order")

    args    = parser.parse_args()
    iprob   = args.iprob
    stepper = args.stepper

    fINT,fORD,fRHS,fBVP,fJAC,x0,y0,x1,nstep,eps = ode_init(iprob,stepper)
    x,y,it                                      = fINT(fRHS,fORD,fBVP,x0,y0,x1,nstep,fJAC=fJAC,eps=eps)

    ode_check(x,y,it,iprob)

#==============================================================

main()


