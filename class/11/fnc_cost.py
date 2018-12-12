#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
import p358utilities as util
#=============================================================
# cost functions for optimization problem
# Each function has optional keyword argument, for consistency.
#=============================================================
# function cst = parabola(x)
# 
# returns function value f(x,y) of 2D parabola
# input : x, a 2-element vector
# output: f
#-------------------------------------------------------------

def parabola1d(x,**kwargs):
    if (isinstance(x,np.ndarray)):
        return x[0]**2
    else:
        return (x-1.0)**2

def parabola2d(x,**kwargs):
    return (x[0]-0.3)**2+(x[1]+0.2)**2

def sphere(x,**kwargs):
    return ((x[0]-0.1)**2+(x[1]-0.2)**2+(x[2]-0.3)**2)

#=============================================================
# function cst = mandelbrot(x)
# 
# returns function value f(x,y) of modified mandelbrot generator.
# input : x, a 2-element vector
# output: f
# defined on [-2,2]x[-2,2]
#-------------------------------------------------------------
def mandelbrot(x,**kwargs):
    return np.sqrt( np.power(x[0]*x[0]*x[0]-3*x[0]*x[1]*x[1]-1.0,2)
                    +np.power(3*x[0]*x[0]*x[1]-x[1]*x[1]*x[1],2))-x[1]+1.0

#=============================================================
# function cst = crazy(x)
# 
# returns function value f(x,y) of whatever function this is.
# input : x, a 2-element vector
# output: f
# Defined on [-1,1]x[-1,1]
#-------------------------------------------------------------
def crazy(xv,**kwargs):
    x = xv[0]
    y = xv[1]
    return (  np.power(x*np.sin(20.0*y)+y*np.sin(20.0*x),2)*np.cosh(np.sin(10.0*x)*x)
             + np.power(x*np.cos(10.0*y)-y*np.sin(10.0*x),2)*np.cosh(np.cos(20.0*y)*y))

#=============================================================
# function cst = charbonneau1(x)
#
# See Paul Charbonneau notes.
# Defined on [0,1]x[0,1]
#-------------------------------------------------------------
def charbonneau1(x,**kwargs):
    n = 9.0
    r = np.sqrt((x[0]-0.5)**2+(x[1]-0.5)**2)
    s = 0.15
    return (np.cos(n*np.pi*r))**2 * np.exp(-(r/s)**2)

#=============================================================
# function cst = charbonneau2(x)
#
# See Paul Charbonneau notes.
# Defined on [0,1]x[0,1]
#-------------------------------------------------------------
def charbonneau2(x,**kwargs):
    r1s = (x[0]-0.5)**2 + (x[1]-0.5)**2
    r2s = (x[0]-0.6)**2 + (x[1]-0.1)**2
    s1s = 0.09
    s2s = 0.0009
    a   = 0.8
    b   = 0.879008
    return a*np.exp(-r1s/s1s)+b*np.exp(-r2s/s2s)

#=============================================================
# function cst = charbonneau4(x)
#
# See Paul Charbonneau notes.
# Defined on [0,1]x[0,1]
#-------------------------------------------------------------
def charbonneau4(x,**kwargs):
    A1 = 0.9
    x1 = 0.3
    s1 = 0.1
    A2 = 0.3
    x2 = 0.8
    s2 = 0.025
    xk = np.arange(51)/51.0
    yd = A1*np.exp(-((xk-x1)/s1)**2) + A2*np.exp(-((xk-x2)/s2)**2)
    y  = x[0]*np.exp(-((xk-x[1])/x[2])**2) + x[3]*np.exp(-((xk-x[4])/x[5])**2)
    R  = np.sum((y-yd)**2)
    return R

#=============================================================
# function cst = lunarlander(tthron)
# 
# returns cost function for optimization problem, lunar landing
# input : tthron: time to switch on throttle 
# output: cst, combination of fuel used and final velocity
# requires a bunch of functions listed below
#-------------------------------------------------------------
def lunarlander(tthron,**kwargs):
    nstep  = 100
    x0     = 0.0
    x1     = 20.0
    y0     = np.array([5e2,-5.0,1e2,0.0])
    x,y,it = ode_bvp(dydx_lunarlanding,step_rk4,bvp_lunarlander,x0,y0,x1,nstep,thr=tthron)
    fused  = y[2,0]-y[2,nstep-1]
    vzend  = y[1,nstep-1]
    cst    = np.sqrt(fused**2+vzend**2) # this is the cost function including velocity
    # cst    = np.sqrt(fused**2) # this is the cost function without the velocity...
    iplot  = 0
    for key in kwargs:
        if (key == 'plot'):
            iplot = kwargs[key]
    if (iplot == 1):
        ftsz = 10
        plt.figure(num=2,figsize=(8,8),dpi=100,facecolor='white')

        plt.subplot(221)
        plt.plot(x,y[0,:],linestyle='-',color='black',linewidth=1.0)
        plt.xlabel('t [s]',fontsize=ftsz)
        plt.ylabel('z(t) [m]',fontsize=ftsz)
        util.rescaleplot(x,y[0,:],plt,0.05)
        plt.tick_params(labelsize=ftsz)

        plt.subplot(222)
        plt.plot(x,y[1,:],linestyle='-',color='black',linewidth=1.0)
        plt.xlabel('t [s]',fontsize=ftsz)
        plt.ylabel('v$_z$ [m s$^{-1}$]',fontsize=ftsz)
        util.rescaleplot(x,y[1,:],plt,0.05)
        plt.tick_params(labelsize=ftsz)

        plt.subplot(223)
        plt.plot(x,y[2,:],linestyle='-',color='black',linewidth=1.0)
        plt.xlabel('t [s]',fontsize=ftsz)
        plt.ylabel('fuel [kg]',fontsize=ftsz)
        util.rescaleplot(x,y[2,:],plt,0.05)
        plt.tick_params(labelsize=ftsz)

        plt.subplot(224)
        plt.plot(x,y[3,:],linestyle='-',color='black',linewidth=1.0)
        plt.xlabel('t [s]',fontsize=ftsz)
        plt.ylabel('throttle',fontsize=ftsz)
        util.rescaleplot(x,y[3,:],plt,0.05)
        plt.tick_params(labelsize=ftsz)

    return cst

def step_rk4(fRHS,x0,y0,dx,**kwargs):
    k1 = dx*fRHS(x0       ,y0       ,dx,**kwargs)
    k2 = dx*fRHS(x0+0.5*dx,y0+0.5*k1,dx,**kwargs)
    k3 = dx*fRHS(x0+0.5*dx,y0+0.5*k2,dx,**kwargs)
    k4 = dx*fRHS(x0+    dx,y0+    k3,dx,**kwargs)
    y  = y0+(k1+2.0*(k2+k3)+k4)/6.0
    return y,1

def ode_ivp(fRHS,fORD,fBVP,x0,y0,x1,nstep,**kwargs):
    nvar    = y0.size                      # number of ODEs
    x       = np.linspace(x0,x1,nstep+1)   # generates equal-distant support points
    y       = np.zeros((nvar,nstep+1))     # result array 
    y[:,0]  = y0                           # set initial condition
    dx      = x[1]-x[0]                    # step size
    it      = np.zeros(nstep+1)
    for k in range(1,nstep+1):
        y[:,k],it[k] = fORD(fRHS,x[k-1],y[:,k-1],dx,**kwargs)
    return x,y,it

def ode_bvp(fRHS,fORD,fBVP,x0,y0,x1g,nstep,**kwargs):
    eps     = 1e-4
    nvar    = y0.size
    x1      = x1g
    x2      = x1
    # (1) root bracketing
    x,y,it = ode_ivp(fRHS,fORD,fBVP,x0,y0,x1,nstep,**kwargs)
    f1     = fBVP(y[:,nstep])
    f2     = f1
    if (f1 > 0): # increase x1
        xfac = 1.10
    else:        # decrease x1
        xfac = 0.90
    while (f1*f2 > 0.0): # bracket the root
        f1     = f2
        x1     = x2
        x2     = xfac*x2
        x,y,it = ode_ivp(fRHS,fORD,fBVP,x0,y0,x2,nstep,**kwargs)
        f2     = fBVP(y[:,nstep])
    if (x1 > x2):     # rearrange such that x1<x2
        x1,x2 = x2,x1 # tuple swap 
        f1,f2 = f2,f1
    # (2) root finder
    xm  = 0.5*(x1+x2)
    x,y,it = ode_ivp(fRHS,fORD,fBVP,x0,y0,xm,nstep,**kwargs)
    fm  = fBVP(y[:,nstep])
    while (np.abs(x2-x1)/xm > eps):
        if (f1*fm > 0.0):
            x1 = xm
            f1 = fm
        else:
            x2 = xm
        xm     = 0.5*(x1+x2)
        x,y,it = ode_ivp(fRHS,fORD,fBVP,x0,y0,xm,nstep,**kwargs)
        fm     = fBVP(y[:,nstep])
    return x,y,it

def dydx_lunarlanding(x,y,dx,**kwargs):
    for key in kwargs:
        if (key == 'thr'):
            thrctrl = kwargs[key]
        else:
            raise Exception('[dydx_lunarlander]: need throttle value')
    dydx    =  np.zeros(4)
    thrmax  =  2.5e3
    g       =  1.62
    Vnozz   =  2.5e3
    mship   =  9e2
    z       =  y[0]
    vz      =  y[1]
    fuel    =  y[2]
    thrfrac =  y[3]

    # limit throttle value. 
    thrfrac = np.max(np.array([0.0,np.min(np.array([1.0,thrfrac]))]))
    # make sure fuel can't go negative
    if (fuel <= 0.0):
        fuel    = 0.0
        thrfrac = 0.0
    mtot    = mship+fuel

    dzdt    =  vz
    dvzdt   =  thrmax*thrfrac/mtot - g
    dfdt    = -thrmax*thrfrac/Vnozz
    if (x > thrctrl):
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

def bvp_lunarlander(y):
    return np.sign(y[0])

# end block lunarlander
#================================================
