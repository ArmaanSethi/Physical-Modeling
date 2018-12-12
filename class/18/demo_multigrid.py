#=======================================
# demo_multigrid.py
# Compares Jacobi, Gauss-Seidel, and Multigrid methods
# for solving the Laplace equation with periodic boundary
# conditions, and an initial condition u=sin(kx).
# Calling sequence:
#   python3 demo_multigrid.py jc 1 128 5 1e-4 --track 
#     Will solve Laplace eq. with Jacobi method and
#     a multi-modal sine wave on a grid with 128 support
#     points. Maximum wave number is 2**5=32. Tolerance
#     for Jacobi/Gauss-Seidel is 1e-4, and component
#     amplitudes are shown against iterations.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation 
import argparse                  # allows us to deal with arguments to main()
from argparse import RawTextHelpFormatter

#----------------------------
# residual (squared difference) between two solutions
# Used by jacobi, gaussseidel
def sresidual(u1,u2):
    return np.sqrt(np.mean((u1-u2)**2))

#-----------------------------
# removes ghost cells from array. 
def strip(u):
    return(u[1:-1])

#-----------------------------
# Residual of linear system Ax=b: Ae=r
# Used by vcycle
def residual(u,f,dx,fBNC):
    J               = f.shape[0]
    u1              = np.zeros(J+2)
    u1[1:J+1]       = u
    u1              = fBNC(u1)
    r               = f[:]*dx**2 - (u1[0:J]+u1[2:J+2]-2.0*u1[1:J+1])
    return r

#-----------------------------
# Restriction (coarsening) of grid.
# For cell-centered quantities, this
# is just taking an average.
def restrict(vfine):
    J  = vfine.size
    j   = 2*np.arange(J//2,dtype=int)
    vcoarse = 0.5*(vfine[j]+vfine[j+1])
    return vcoarse

#-----------------------------
# Prolongation (refinement) of grid
# Requires interpolation. Note the advantage
# of using cell-centered quantities.
# Also keep in mind that we need the
# boundary conditions for prologation.
def prolong(u,fBNC):
    J               = u.shape[0]
    u1              = np.zeros(J+2)
    u1[1:J+1]       = u
    u1              = fBNC(u1) 
    j               = np.arange(J,dtype=int) # indices on coarse grid
    mslopes         = np.zeros(2*J) # slopes on fine grid
    m2nd            = np.zeros(2*J)
    m2nd[2*j]       = 0.5*(u1[2:J+2]-u1[0:J]) # high resolution slopes
    m2nd[2*j+1]     = 0.5*(u1[2:J+2]-u1[0:J])
    m1st            = np.zeros(2*J)
    m1st[2*j]       = u1[j+1]-u1[j]
    m1st[2*j+1]     = u1[j+2]-u1[j+1]
    uf              = np.zeros(2*J)
    uf[2*j  ]       = u - 0.25*m2nd[2*j]
    uf[2*j+1]       = u + 0.25*m2nd[2*j+1]
    r               = np.sign((uf[2*j]-u1[1:J+1])*(uf[2*j+1]-u1[1:J+1]))
    uf[2*j  ]       = uf[2*j]  - r*(0.25*m2nd[2*j  ] - 0.25*m1st[2*j])
    uf[2*j+1]       = uf[2*j+1]+ r*(0.25*m2nd[2*j+1] - 0.25*m1st[2*j+1])
    return uf

#----------------------------
# kernel (RHS derivative operator) for iterative solvers (Jacobi, Gauss-Seidel)
def kernel(u):
    return u[2:]+u[:-2]

#-----------------------------
# periodic boundary conditions
# Returns u as copy of u0, with boundaries filled.
def bnc_periodic(u0):
    u    = np.copy(u0)
    u[0]      = u[-2]
    u[-1]     = u[1]
    return u

#-----------------------------
# no-charge (Dirichlet) boundary conditions
# Returns u as copy of u0, with boundaries filled.
def bnc_nocharge(u0):
    u         = np.copy(u0)
    u[0]      = -u[1]
    u[-1]     = -u[-2]
    return u

#-----------------------------
# Jacobi method.
# u0   : Initial guess
# f    : RHS
# dx   : distance between grid points (needed for kernel)
# maxit: maximum number of iterations
# fBNC : pointer to boundary condition function
# keywords:
#   track [True/False]: returns (J,it) array containing
#                       intermediate results
def jacobi(u0, f, dx, tol, maxit, fBNC,**kwargs):
    itrack = False
    silent = False
    for key in kwargs:
        if (key == 'track'):
            itrack = kwargs[key]
        if (key == 'silent'):
            silent = kwargs[key]
    if (itrack):
        utrack = np.zeros((len(f),maxit+1))
        utrack[:,0] = u0
    s    = u0.shape
    u1           = np.zeros(s[0]+2)
    u2           = np.zeros(s[0])
    u1[1:s[0]+1] = u0
    res  = 1.0
    it   = 0
    while ((res > tol) & (it < maxit)):
        u1      = fBNC(u1)
        u2[:]   = 0.5*(kernel(u1) - f*dx**2) # can use array operators in Jacobi method.
        res     = sresidual(strip(u1),u2)
        if (not silent):
            print('[jacobi]: it=%5i res=%13.5e min/max(u)=%13.5e %13.5e' % (it,res,np.min(u2),np.max(u2)))
        u1[1:s[0]+1] = u2[:]
        it      = it+1
        if (itrack):
            utrack[:,it] = u2
    if (itrack):
        return np.squeeze(utrack[:,:it])
    else:
        return u2

#-----------------------------
# Gauss-Seidel method. 
# u0   : Initial guess
# f    : RHS
# dx   : distance between grid points (needed for kernel)
# maxit: maximum number of iterations
# fBNC : pointer to boundary condition function
# keywords:
#   track [True/False]: returns (J,it) array containing
#                       intermediate results
def gaussseidel(u0, f, dx, tol, maxit, fBNC,**kwargs):
    itrack = False
    for key in kwargs:
        if (key == 'track'):
            itrack = kwargs[key]
    if (itrack):
        utrack = np.zeros((len(f),maxit+1))
        utrack[:,0] = u0
    s    = u0.shape
    u1           = np.zeros(s[0]+2)
    u2           = np.zeros(s[0])
    u1[1:s[0]+1] = u0
    res  = 1.0
    it   = 0
    while ((res > tol) & (it < maxit)):
        u1      = fBNC(u1)
        ut      = kernel(u1)
        for i in range(s[0]): # explicit loop because of successive updates in G-S method.
            u2[i]   = u1[i+1] # u2 does not contain boundaries
            u1[i+1] = 0.5*(u1[i]+u1[i+2]-f[i]*dx**2)
        res     = sresidual(strip(u1),u2)
        print('[gaussseidel]: it=%5i res=%13.5e min/max(u)=%13.5e %13.5e' % (it,res,np.min(u2),np.max(u2)))
        it      = it+1
        if (itrack):
            utrack[:,it] = u2
    if (itrack):
        return np.squeeze(utrack[:,:it])
    else:
        return u2

#-----------------------------
# Vcycle (core of multigrid method). 
# vh   : Initial guess
# f    : RHS
# dx   : distance between grid points (needed for kernel)
# tol  : for consistency with Jacobi/Gauss-Seidel. Not used.
# fBNC : pointer to boundary condition function
# level: recursive level for diagnostics output.
def vcycle(vh, f, dx, tol, fBNC, level):
    verbose    = None
    npre       = 10
    npost      = 10
    nx         = vh.size
    lstr = ''
    for l in range(level):
        lstr = lstr+'    '
    if (nx == 2): # coarsest possible grid: solve directly
        v1      = np.zeros(4)
        v1[1:3] = vh
        v1[0]   = -vh[0] # also different from P358 result: cannot use 0 here. 
        v1[3]   = -vh[1]
        v1[1]   = (2.0*v1[0]+v1[3]-(2.0*f[0]+f[1])*dx**2)/3.0 #f = -(dx)**2
        v1[2]   = (2.0*v1[3]+v1[0]-(2.0*f[1]+f[0])*dx**2)/3.0 #f = -(dx)**2
        vh      = v1[1:3]
    else: # all other (larger) grids: call vcycle recursively
        vh    = jacobi(vh, f, dx, tol, npre, fBNC, silent=True) # pre-conditioning
        if (verbose):
            print('%s (0) min/max(vh ) = %13.5e %13.5e min/max(f  ) = %13.5e %13.5e' % (lstr,np.min(vh),np.max(vh),np.min(f),np.max(f)))
        rh    = residual(f, vh, dx, fBNC) # get the residual
        if (verbose): 
            print('%s (1) min/max(vh ) = %13.5e %13.5e min/max(rh ) = %13.5e %13.5e' % (lstr,np.min(vh),np.max(vh),np.min(rh),np.max(rh)))
        r2h   = restrict(rh) # restrict the RHS
        e2h   = restrict(vh) # cheap way to get correct dimensions without querying array size etc
        e2h[:]= 0.0 # error guess is 0.0 
        dx2h  = dx
        for i in range(2): # recursive call of vcycle
            e2h  = vcycle(e2h, r2h, dx2h, tol, bnc_nocharge, level+1)
        if (verbose): 
            print('%s (2) min/max(e2h) = %13.5e %13.5e min/max(r2h) = %13.5e %13.5e' % (lstr,np.min(e2h),np.max(e2h),np.min(r2h),np.max(r2h)))
        eh    = prolong(e2h,fBNC) # prolong to higher-resolution grid
        vh    = vh + eh # add correction from coarser grid to finer grid
        if (verbose): 
            print('%s (3) min/max(vh ) = %13.5e %13.5e min/max(eh ) = %13.5e %13.5e' % (lstr,np.min(vh),np.max(vh),np.min(eh),np.max(eh)))
        vh    = jacobi(vh, f, dx, tol, npost, fBNC, silent=True) # smooth noise introduced by prolongation
        if (verbose): 
            print('%s (3) min/max(vh ) = %13.5e %13.5e' % (lstr,np.min(vh),np.max(vh)))
    return vh

#full multigrid 
def fullmg(f, dx, tol, fBNC, level):
    npre       = 10
    npost      = 10
    nx         = f.size
    lstr = ''
    for l in range(level):
        lstr = lstr+'    '
    if (nx == 2):
        vh = np.zeros(2)
    else:
        f2h = restrict(f)
        v2h = fullmg(f2h,2*dx,tol,bnc_nocharge,level+1)
        vh  = prolong(v2h,fBNC)

    vh = vcycle(vh, f, dx, tol, fBNC, level) 
    return vh

def fullmultigrid(u, f, dx, tol, fBNC):
    u = fullmg(f, dx, tol, fBNC, 0)
    return u

#-----------------------------
# multigrid. Calls vcycle several times.
# u    : Initial guess
# f    : RHS
# dx   : distance between grid points (needed for kernel)
# tol  : for consistency with Jacobi/Gauss-Seidel. Not used.
# fBNC : pointer to boundary condition function
# keywords:
#   track [True/False]: returns (J,it) array containing
#                       intermediate results
# Note: Could be simpler, but structure was kept the same
# as gaussseidel and jacobi, for consistency.
def multigrid(u, f, dx, tol, maxit, fBNC, **kwargs):
    itrack = False
    for key in kwargs:
        if (key == 'track'):
            itrack = kwargs[key]
    if (itrack):
        utrack = np.zeros((len(f),maxit+1))
        utrack[:,0] = u
    u1  = np.copy(u)
    u2  = np.copy(u)
    res = 1.0
    it  = 0
    while ((res > tol) & (it < maxit)):
        u2  = vcycle(u1,f,dx,tol,fBNC,0)
        res = sresidual(u1,u2)
        print('[multigrid]: it=%5i res=%13.5e min/max(u)=%13.5e %13.5e' % (it,res,np.min(u2),np.max(u2)))
        if (itrack):
            utrack[:,it+1] = u2
        u1[:] = u2[:] # need to overwrite...
        it    = it+1
    if (itrack):
        return utrack[:,:it]
    else:
        return u2

#-------------------------------
# initial conditions, grid sizes etc.
# J    : number of support points (cell-centered)
# kmax : wave number (integer)
def init(J, kmax):
    xmin = 0.0
    xmax = 1.0
    dx   = (xmax-xmin)/float(J)
    x    = (np.arange(J)+0.5)*dx+xmin
    u0   = np.zeros(J)
    for k1 in range(1,kmax):
        u0   = u0 + np.sin(2.0*(2**k1)*np.pi*x)
    f    = np.zeros(J) # RHS is just 0
    fBNC = bnc_periodic
    return x,dx,u0,f,fBNC

#---------------------------------
# Wrapper for fourier transform to get amplitudes of components
def fft(y):
    fft = np.abs(np.fft.rfft(y))
    return fft**2/len(fft)**2

#---------------------------------
# Update function used by FuncAnimation 
def update(i,x,y,l1,l2,u,ax1):
    y = np.squeeze(u[:,i+1])
    l1.set_data(x,y)
    ax1.text(0.9,1.5,"it=%5i" % (i),color='white') # ... there **must** be a better way ...
    ax1.text(0.9,1.5,"it=%5i" % (i+1))
    y2 = fft(y)
    l2.set_data(np.arange(len(y2)-1)+1, y2[1:])

#==================================
# main 
def main():
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument("solver",type=str,help="jc|gs|mg for Jacobi, Gauss-Seidel, Multigrid (Vcycle)")
    parser.add_argument("J",type=int,help="number of support points (should be power of 2 for multigrid)")
    parser.add_argument("kmax",type=int,help="max wavenumber given by 2**kmax")
    parser.add_argument("tol",type=float,help="tolerance for Jacobi and Gauss-Seidel",default=1e-4)
    parser.add_argument("-t","--track",action="store_true",help="keeps track of intermediate results.")
    parser.add_argument("-a","--anim",action="store_true",help="shows evolution of solution and its Fourier modes")
    args  = parser.parse_args()
    solver= args.solver
    J     = args.J
    kmax  = args.kmax
    tol   = args.tol

    x,dx,u0,f,fBNC = init(J,kmax)

    maxit = 10000

    if (solver == "jc"):
        u = jacobi(u0, f, dx, tol, maxit, fBNC,track=args.track)
    elif (solver == "gs"):
        u = gaussseidel(u0, f, dx, tol, maxit, fBNC,track=args.track)
    elif (solver == "mg"):
        u = multigrid(u0, f, dx, tol, maxit, fBNC,track=args.track)
    elif (solver == "fmg"):
        u = fullmultigrid(u0, f, dx, tol, fBNC)

    # for tracking, we get all the intermediate steps
    if (args.track):
        nt = (u.shape)[1]
        if (args.anim):
            y  = u[:,0]
            y2 = fft(u[:,0])
            fig, (ax1,ax2) = plt.subplots(nrows=2,figsize=(6,6))
            line1, = ax1.plot(x,y)
            line2, = ax2.plot(np.arange(len(y2)-1)+1,y2[1:])
            ax1.set_xlabel("x")
            ax1.set_ylabel("u")
            ax1.text(0.9,1.5,("it=%5i" % (0)))
            ax2.set_ylim((0,np.max(y2)))
            ax2.set_xlabel("k")
            ax2.set_ylabel("P(k)")
            y    = u[:,0]
            anim = matplotlib.animation.FuncAnimation(fig, update, frames=nt-1, repeat=True,fargs=(x,y,line1,line2,u,ax1))  
            plt.tight_layout()
            plt.show()
        else:
            itarr = np.arange(nt)
            amarr = np.zeros((kmax-1,nt))
            slope = np.zeros(kmax-1)
            for i in range(nt):
                f = fft(np.squeeze(u[:,i]))
                amarr[:,i] = f[2**(np.arange(1,kmax))]
            plt.figure(num=1,figsize=(5,5),dpi=100,facecolor='white') 
            for k1 in range(1,kmax):
                slope[k1-1] = (np.log(amarr[k1-1,5])-np.log(amarr[k1-1,0]))/(itarr[5]-itarr[0])
                plt.plot((itarr+1),np.log10(amarr[k1-1]),label=("k=%2i, c=%5.2f" % (2**k1,np.abs(slope[k1-1]))))
            plt.xlabel("iteration") 
            plt.ylabel("log amplitude")
            plt.legend()
            plt.show()
             
    else: # neither track nor anim: plot final result
        plt.figure(num=1,figsize=(6,6),dpi=100,facecolor='white')
        plt.plot(x,u)
        plt.xlabel("x")
        plt.ylabel("u")
        plt.show()

#==============================

main()



    
    
