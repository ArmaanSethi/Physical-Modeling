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
#   main            : calls the rest. Needs argument prob. Calling sequence e.g.: ode_test.py kepler rk4
#
#==============================================================
# required libraries
import argparse	                 # allows us to deal with arguments to main()
from argparse import RawTextHelpFormatter
import numpy as np               # numerical routines (arrays, math functions etc)
import matplotlib.pyplot as plt  # plotting commands
import p358utilities as util     # for rescaleplot
import globalvar                 # interface for global variables

import ode_integrators as odeint # contains the drivers for ODE integration
import ode_step as step          # the stepper functions
import ode_dydx as dydx          # contains the RHS functions for selected problems.
import ode_bvp  as bvp           # contains the boundary value functions for selected problems.

#==============================================================
# functions
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
# function fRHS,x0,y0,x1 = ode_init(prob,stepper)
#
# Initializes derivative function, parameters, and initial conditions
# for ODE integration.
#
# input: 
#   prob:    lunarlander 
#            kepler
#            h2formation 
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

def ode_init(prob,stepper):

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

    if (prob == 'lunarlander'): 
        print('[ode_init]: initializing lunar lander')
        thrmax  = 2.5e3 # thrmax[N]      : maximum thrust (par[0])
        mode    = 0.0   # imode          : controls throttle value (see dydx.lunarlanding)
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
        fRHS    = dydx.lunarlanding         # function handle: RHS of ODE
        fBVP    = bvp.lunarlanding          # function handle: BVP values

    elif (prob == 'kepler'): #  (direct, multiple bodies)
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
        fRHS    = dydx.keplerdirect

    elif (prob == 'h2formation'): # chemical reaction network
        print('[ode_init]: initializing chemical network')
        # Simplified molecular hydrogen formation network.
        # Species: A = neutral hydrogen, B = molecular hydrogen, C = ionized hydrogen
        # The reaction rate coefficients are:
        k        = get_h2rates()
        Myr      = 1e6*3.56e2*3.6e3*2.4e1 # 10^6 years in seconds

        nstepMyr = 5     # steps per Myr.
        nMyrs    = 20
        x0       = 0.0
        x1       = nMyrs*Myr
        nstep    = nMyrs*nstepMyr
        nspecies = 3
        D0       = 1e2   # total hydrogen atom number density
        B0       = 1e-7    # molecular hydrogen
        C0       = 1e-3    # ionized hydrogen
        A0       = D0-2.0*B0-C0# atomic hydrogen density
        par      = np.zeros(5)
        par[0:4] = k[:]
        par[4]   = D0
        y0       = np.array([A0,B0,C0])
        fINT     = odeint.ode_ivp
        fRHS     = dydx.h2formation
    else:
        raise Exception('[ode_int]: invalid prob %s\n' % (prob))

    globalvar.set_odepar(par)
    return fINT,fORD,fRHS,fBVP,x0,y0,x1,nstep

#==============================================================
# function ode_check(x,y,prob)
#
# Performs problem-dependent tests on results.
#
# input: 
#   prob:   lunarlander 
#           kepler
#           h2formation
#   iinteg   : integrator type
#   x    :  independent variable
#   y    :  integration result
#   it   :  number of substeps used. Only meaningful for RK45 (iinteg = 4). 
#--------------------------------------------------------------

def ode_check(x,y,it,prob):
    
    if (prob == 'lunarlander'): # lunar lander problem
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

    elif (prob == 'kepler'):
        # for the direct Kepler problem, we check for energy and angular momentum conservation,
        # and for the center-of-mass position and velocity
        color   = ['black','green','cyan','blue','red','black','black','black','black']
        n       = x.size
        par     = globalvar.get_odepar()
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

        plt.tight_layout()

        plt.show() 

    elif (prob == 'h2formation'): # H2 formation
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

        plt.tight_layout()

        plt.show()

    else:
        raise Exception('[ode_check]: invalid prob %i\n' % (prob))

#==============================================================
#==============================================================
# main
# 
# parameters:
#   prob:   lunarlander 
#           kepler
#           h2formation

def main():

    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument("prob",type=str,default='lunarlander',
                        help="problem number:\n" 
                             "   lunarlander\n"
                             "   kepler\n"
                             "   h2formation")
    parser.add_argument("stepper",type=str,default='euler',
                        help="stepping function:\n"
                             "   euler: Euler step\n"
                             "   rk2  : Runge-Kutta 2nd order\n"
                             "   rk4  : Runge-Kutta 4th order\n")

    args   = parser.parse_args()

    prob   = args.prob
    stepper= args.stepper

    fINT,fORD,fRHS,fBVP,x0,y0,x1,nstep = ode_init(prob,stepper)
    x,y,it                             = fINT(fRHS,fORD,fBVP,x0,y0,x1,nstep)

    ode_check(x,y,it,prob)

#==============================================================

main()


