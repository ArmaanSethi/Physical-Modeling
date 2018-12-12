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

#==============================================================
# function dydx = kepler(x,y,dx)
#
# Calculates ODE RHS for Kepler problem.
#
# input: 
#   x,y    : position and values for RHS
#            Assumes y to have the shape (x, y, v_x, v_y), 
#            with the cartesian positions (x,y) and their
#            velocities (v_x,v_y).
# global:
#   G*m1*m2   : grav constant 
# output:
#   dydx   : vector of results as in y'=f(x,y)
#--------------------------------------------------------------

def kepler(x,y,dx):
    par     = globalvar.get_odepar()
    dydx    = np.zeros(4)
    R       = np.sqrt(y[0]*y[0]+y[1]*y[1])
    R3      = np.power(R,3)
    dydx[0] = y[2]              # x' = v_x
    dydx[1] = y[3]              # y' = v_y
    dydx[2] = -par[0]*y[0]/R3
    dydx[3] = -par[0]*y[1]/R3
    return dydx

#==============================================================
# function dydx = keplerdirect(x,y,dx)
#
# Calculates ODE RHS for Kepler problem via direct summation.
#
# input: 
#   x,y    : position and values for RHS
#            Assumes y to have the shape (x, y, v_x, v_y), 
#            with the cartesian positions (x,y) and their
#            velocities (v_x,v_y).
# global:
#   G      : grav constant
# output:
#   dydx   : vector of results as in y'=f(x,y)
#--------------------------------------------------------------

def keplerdirect(x,y,dx):
    nbodies     = y.size//4 # per body, we have four variables
    par         = globalvar.get_odepar()
    npar        = par.size
    gnewton     = par[0]
    masses      = par[1:npar]
    dydx        = np.zeros(4*nbodies)
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

#==============================================================
# function dydx = keplerdirect_symp1(x,y,dx)
#
# Calculates partial derivative of Hamiltonian with respect
# to q,p for direct Kepler problem. Provides full update
# for one single symplectic Euler step.
#
# input: 
#   x,y    : position and values for RHS
#            Assumes y to have the shape (x, y, v_x, v_y), 
#            with the cartesian positions (x,y) and their
#            velocities (v_x,v_y).
# global:
#   G(m1*m2): grav constant times masses (in any units) (par[0])
# output:
#   dydx    : vector of results as in y'=f(x,y)
#--------------------------------------------------------------

def keplerdirect_symp1(x,y,dx):
    nbodies     = y.size/4 # per body, we have four variables
    par         = globalvar.get_odepar()
    npar        = par.size
    gnewton     = par[0]
    masses      = par[1:npar]
    dydx        = np.zeros(4*nbodies)
    pHpq        = np.zeros(2*nbodies)
    pHpp        = np.zeros(2*nbodies)
    indx        = 2*np.arange(nbodies)
    indy        = 2*np.arange(nbodies)+1
    indvx       = 2*np.arange(nbodies)+2*nbodies
    indvy       = 2*np.arange(nbodies)+2*nbodies+1
    px          = y[indvx]*masses # this is more cumbersome than necessary,
    py          = y[indvy]*masses # but for consistency with derivation.
    qx          = y[indx]
    qy          = y[indy]
    for i in range(nbodies):
        for j in range(0,i):
            ddx           = qx[i]-qx[j]
            ddy           = qy[i]-qy[j]
            R3            = np.power(ddx*ddx+ddy*ddy,1.5)
            pHpq[indx[i]] = pHpq[indx[i]] + masses[j]*ddx/R3
            pHpq[indy[i]] = pHpq[indy[i]] + masses[j]*ddy/R3
        for j in range(i+1,nbodies):
            ddx           = qx[i]-qx[j]
            ddy           = qy[i]-qy[j]
            R3            = np.power(ddx*ddx+ddy*ddy,1.5)
            pHpq[indx[i]] = pHpq[indx[i]] + masses[j]*ddx/R3
            pHpq[indy[i]] = pHpq[indy[i]] + masses[j]*ddy/R3
    pHpq[indx] = pHpq[indx] * gnewton * masses
    pHpq[indy] = pHpq[indy] * gnewton * masses
    pHpp[indx] = (px-dx*pHpq[indx])/masses
    pHpp[indy] = (py-dx*pHpq[indy])/masses
    dydx[indx] = pHpp[indx]
    dydx[indy] = pHpp[indy]
    dydx[indvx]= -pHpq[indx]/masses
    dydx[indvy]= -pHpq[indy]/masses
    #for i in range(nbodies):
    #    print('t=%10.2e mass=%10.2e: x,y=%13.5e %13.5e vx,vy=%13.5e %13.5e dx,dy=%13.5e %13.5e dvx,dvy= %13.5e %13.5e' 
    #          % (x,masses[i],qx[i],qy[i],px[i]/masses[i],py[i]/masses[i],dx*dydx[indx[i]],dx*dydx[indy[i]],dx*dydx[indvx[i]]/masses[i],dx*dydx[indvy[i]]/masses[i]))

    return dydx

#==============================================================
# function dydx = keplerdirect_symp2(x,y,dx)
#
# Calculates partial derivative of Hamiltonian with respect
# to q,p for direct Kepler problem. Provides full update
# for one single symplectic RK2 (Stoermer-Verlet) step.
#
# input: 
#   x,y    : position and values for RHS
#            Assumes y to have the shape (x, y, v_x, v_y), 
#            with the cartesian positions (x,y) and their
#            velocities (v_x,v_y).
# global:
#   G(m1*m2): grav constant times masses (in any units) (par[0])
# output:
#   dydx    : vector of results as in y'=f(x,y)
#--------------------------------------------------------------

def keplerdirect_symp2(x,y,dx):
    nbodies     = y.size/4 # per body, we have four variables
    par         = globalvar.get_odepar()
    npar        = par.size
    gnewton     = par[0]
    masses      = par[1:npar]
    dydx        = np.zeros(4*nbodies)
    pHpq        = np.zeros(2*nbodies)
    pHpq2       = np.zeros(2*nbodies)
    pHpp        = np.zeros(2*nbodies)
    indx        = 2*np.arange(nbodies)
    indy        = 2*np.arange(nbodies)+1
    indvx       = 2*np.arange(nbodies)+2*nbodies
    indvy       = 2*np.arange(nbodies)+2*nbodies+1
    px          = y[indvx]*masses # this is more cumbersome than necessary,
    py          = y[indvy]*masses # but for consistency with derivation.
    qx          = y[indx]
    qy          = y[indy]
    # first step: constructing p(n+1/2)
    for i in range(nbodies):
        for j in range(0,i):
            ddx           = qx[i]-qx[j]
            ddy           = qy[i]-qy[j]
            R3            = np.power(ddx*ddx+ddy*ddy,1.5)
            pHpq[indx[i]] = pHpq[indx[i]] + masses[j]*ddx/R3
            pHpq[indy[i]] = pHpq[indy[i]] + masses[j]*ddy/R3
        for j in range(i+1,nbodies):
            ddx           = qx[i]-qx[j]
            ddy           = qy[i]-qy[j]
            R3            = np.power(ddx*ddx+ddy*ddy,1.5)
            pHpq[indx[i]] = pHpq[indx[i]] + masses[j]*ddx/R3
            pHpq[indy[i]] = pHpq[indy[i]] + masses[j]*ddy/R3
    pHpq[indx] = pHpq[indx] * gnewton * masses
    pHpq[indy] = pHpq[indy] * gnewton * masses
    px2        = px-0.5*dx*pHpq[indx]
    py2        = py-0.5*dx*pHpq[indy]
    pHpp[indx] = px2/masses
    pHpp[indy] = py2/masses
    qx2        = qx+dx*pHpp[indx]
    qy2        = qy+dx*pHpp[indy]
    for i in range(nbodies):
        for j in range(0,i):
            ddx           = qx2[i]-qx2[j]
            ddy           = qy2[i]-qy2[j]
            R3            = np.power(ddx*ddx+ddy*ddy,1.5)
            pHpq2[indx[i]] = pHpq2[indx[i]] + masses[j]*ddx/R3
            pHpq2[indy[i]] = pHpq2[indy[i]] + masses[j]*ddy/R3
        for j in range(i+1,nbodies):
            ddx           = qx2[i]-qx2[j]
            ddy           = qy2[i]-qy2[j]
            R3            = np.power(ddx*ddx+ddy*ddy,1.5)
            pHpq2[indx[i]] = pHpq2[indx[i]] + masses[j]*ddx/R3
            pHpq2[indy[i]] = pHpq2[indy[i]] + masses[j]*ddy/R3
    pHpq2[indx] = pHpq2[indx] * gnewton * masses
    pHpq2[indy] = pHpq2[indy] * gnewton * masses
    dydx[indx]  = pHpp[indx]
    dydx[indy]  = pHpp[indy]
    dydx[indvx] = -0.5*(pHpq[indx]+pHpq2[indx])/masses
    dydx[indvy] = -0.5*(pHpq[indy]+pHpq2[indy])/masses

    return dydx

#==============================================================
# function dydx = h2formation(x,y,dx)
#
# Calculates RHS for molecular hydrogen formation network. 
#
# input: 
#   x,y    : time and abundances (A,B,C) for (n(H), n(H2), n(H+))
#
# global:
#   reaction rate coefficients stored in par
# output:
#   dydx    : vector of results as in y'=f(x,y)
#--------------------------------------------------------------

def h2formation(x,y,dx):
    nspec = y.size
    par   = globalvar.get_odepar()
    A     = y[0] # not necessary, but more readable
    B     = y[1]
    C     = y[2]
    #print 'A,B,C,sum = ',A,B,C,A+B+C
    # This is the stoichiometry matrix S
    S     = np.array([[-2.0, 2.0,-1.0, 1.0],
                      [ 1.0,-1.0, 0.0, 0.0],
                      [ 0.0, 0.0, 1.0,-1.0]])
    # This is the reaction rate vector. 
    # Note that the stoichiometry would give par[3]*C, not par[3]*C*C. 
    # However, we need to account for the electrons, whose density is that of C.
    v     = np.array([par[0]*A*A,par[1]*B,par[2]*A,par[3]*C*C])
    Myr   = 1e6*3.65e2*3.6e3*2.4e1
    dydx  = S.dot(v)
    return dydx


