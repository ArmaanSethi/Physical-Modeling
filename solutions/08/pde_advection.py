#============================================
# Partial differential equations: 
# Advection problem in 1D.
#============================================
import argparse                  # allows us to deal with arguments to main()
from argparse import RawTextHelpFormatter
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import p358utilities as util
#============================================
# Generate mesh
def get_mesh(minmaxx,J):
    dx    = (minmaxx[1]-minmaxx[0])/float(J)
    x     = minmaxx[0]+(np.arange(J)+0.5)*dx
    return x

#============================================
# Boundary condition functions (cell-centered)
# rho is only passed for consistency with multipol expansion.
# phi needs to include the ghost cells.
# Periodic. First active cell is 1, last active cell is J
def bnc_periodic(q):
    J      = q.size-2
    q[0  ] = q[J]
    q[J+1] = q[1]
    return q

#--------------------------------------------
# no charge boundaries
def bnc_nocharge(u):
    J      =  q.size-2
    q[0  ] = -q[1]
    q[J+1] = -q[1]
    return q

#============================================
# initial conditions
# tophat
def inc_tophat(x):
    J = x.size
    q = np.piecewise(x,[x < -0.25,x >= -0.25],[0.0,1.0])*np.piecewise(x,[x <= 0.25, x > 0.25],[1.0,0.0])
    return q

#--------------------------------------------
# gaussian
def inc_gaussian(x):
    q = np.exp(-10.0*x**2)
    return q

#============================================
# Integrators for advection problem.
# input: 
#     x      : array with cell-centered spatial support points
#     t      : array with time support points
#     alpha  : integration factor dt*c/dx
#     fBNC   : boundary condition function
#     fINC   : initial condition function
# output:
#     q      : (J,N) array containing solution 
#--------------------------------------------
# Forward-time centered-space advection update
def ftcs(x,t,alpha,fBNC,fINC):
    J          = x.size
    N          = t.size
    dt         = t[1]-t[0]
    q          = np.zeros((J+2,N))
    q[1:J+1,0] = fINC(x)
    for n in range(N-1):
        q[:,n]       = fBNC(q[:,n])
        q[1:J+1,n+1] = q[1:J+1,n] - 0.5*alpha*(q[2:J+2,n]-q[0:J,n])
        print('[ftcs]: n=%6i t=%13.5e max(q)=%13.5e' % (n,t[n+1],np.max(q[1:J+1,n+1])))
    return q[1:J+1,:]

#--------------------------------------------
# Lax scheme
def lax(x,t,alpha,fBNC,fINC):
    J          = x.size
    N          = t.size
    dt         = t[1]-t[0]
    q          = np.zeros((J+2,N))
    q[1:J+1,0] = fINC(x)
    for n in range(N-1):
        q[:,n]       = fBNC(q[:,n])
        q[1:J+1,n+1] = 0.5*(q[0:J,n]+q[2:J+2,n]) - 0.5*alpha*(q[2:J+2,n]-q[0:J,n])
        #print('[lax]: n=%6i t=%13.5e max(q)=%13.5e' % (n,t[n+1],np.max(q[1:J+1,n+1])))
    return q[1:J+1,:]

#--------------------------------------------
# Leapfrog
def leapfrog(x,t,alpha,fBNC,fINC):
    J          = x.size
    N          = t.size
    dt         = t[1]-t[0]
    q          = np.zeros((J+2,N))
    qt         = np.zeros(J+2)
    q[1:J+1,0] = fINC(x)
    q[:,0]     = fBNC(q[:,0])
    q[1:J+1,1] = q[1:J+1,0] -alpha*(q[1:J+1,0]-q[0:J,0])
    for n in range(1,N-1):
        q[:,n]       = fBNC(q[:,n])
        q[1:J+1,n+1] = q[1:J+1,n-1]-alpha*(q[2:J+2,n]-q[0:J,n])
    return q[1:J+1,:]

#--------------------------------------------
# Lax-Wendroff scheme
def laxwendroff(x,t,alpha,fBNC,fINC):
    J          = x.size
    N          = t.size
    dt         = t[1]-t[0]
    q          = np.zeros((J+2,N))
    q[1:J+1,0] = fINC(x)
    for n in range(N-1):
        q[:,n]       = fBNC(q[:,n])
        q[1:J+1,n+1] = q[1:J+1,n]-alpha*( 0.5*(q[2:J+2,n]+q[1:J+1,n])-0.5*alpha*(q[2:J+2,n]-q[1:J+1,n]) 
                                         -0.5*(q[1:J+1,n]+q[0:J  ,n])+0.5*alpha*(q[1:J+1,n]-q[0:J  ,n]))
    return q[1:J+1,:]

#--------------------------------------------
# upwind scheme
def upwind(x,t,alpha,fBNC,fINC):
    J          = x.size
    N          = t.size
    dt         = t[1]-t[0]
    q          = np.zeros((J+2,N))
    q[1:J+1,0] = fINC(x)
    for n in range(N-1):
        q[:,n]       = fBNC(q[:,n])
        q[1:J+1,n+1] = q[1:J+1,n]-alpha*(q[1:J+1,n]-q[0:J,n])
    return q[1:J+1,:]

#============================================
# Driver for the actual integrators. Sets the initial conditions
# and generates the support point arrays in space and time.
# input: J      : number of spatial support points
#        cfl    : CFL number
#        minmaxx: 2-element array containing minimum and maximum of spatial domain
#        minmaxt: 2-element array, same for time domain
#        cadv   : advection velocity
#        fINT   : integrator (one of ftcs, implicit, Lax, Lax-Wendroff)
#        fBNC   : boundary condition function
#        fINC   : initial condition function
def advection_solve(J,minmaxt,minmaxx,cfl,fSOL,fINC,fBNC):
    # time and space discretization
    cadv  = 1.0 # advection velocity
    x     = get_mesh(minmaxx,J)
    dx    = x[1]-x[0]
    dt    = cfl*dx/np.abs(cadv)
    N     = int((minmaxt[1]-minmaxt[0])/dt)
    dt    = (minmaxt[1]-minmaxt[0])/float(N) # recalculate, to make exact
    t     = minmaxt[0]+np.arange(N+1)*dt
    alpha = dt*cadv/dx
    print('[advection_solve]: alpha = %13.5e N = %7i dt = %13.5e dx = %13.5e' % (alpha,N,dt,dx))
    q        = fSOL(x,t,alpha,fBNC,fINC)
    return x,t,q

#==========================================
# initialization
def init(problem,solver):

    if (problem == 'tophat'):
        fINC    = inc_tophat
        fBNC    = bnc_periodic
        minmaxt = np.array([0.0,1.0])
        minmaxx = np.array([-0.5,0.5])
    elif (problem == 'gaussian'):
        fINC    = inc_gaussian
        fBNC    = bnc_periodic
        minmaxt = np.array([0.0,1.0])
        minmaxx = np.array([-0.5,0.5])
    else:
        print('[init]: invalid problem %s' % (problem))
        exit()

    if (solver == 'ftcs'):
        fSOL = ftcs
    elif (solver == 'lax'):
        fSOL = lax
    elif (solver == 'leapfrog'):
        fSOL = leapfrog
    elif (solver == 'laxwen'):
        fSOL = laxwendroff
    elif (solver == 'upwind'):
        fSOL = upwind
    else:
        print('[init]: invalid solver %s' % (solver))
        exit()

    return fSOL,fINC,fBNC,minmaxt,minmaxx

#============================================
def main():

    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument("J",type=int,
                        help="number of spatial support points (excluding boundaries)")
    parser.add_argument("cfl",type=float,
                        help="Courant-Friedrich-Levy (CFL) number")
    parser.add_argument("solver",type=str,
                        help="advection equation solver:\n"
                             "    ftcs    : forward-time centered-space\n"
                             "    lax     : Lax scheme\n"
                             "    leapfrog: Leapfrog scheme\n"
                             "    laxwen  : Lax-Wendroff scheme\n"
                             "    upwind  : upwind scheme")
    parser.add_argument("problem",type=str,
                        help="initial conditions:\n"
                             "    tophat  : top hat density profile\n"
                             "    gaussian: Gaussian density profile")

    args         = parser.parse_args()
    J            = args.J
    cfl          = args.cfl
    solver       = args.solver
    problem      = args.problem

    fSOL,fINC,fBNC,minmaxx,minmaxt = init(problem,solver)
    x,t,q        = advection_solve(J,minmaxx,minmaxt,cfl,fSOL,fINC,fBNC)


    ftsz    = 10
    fig     = plt.figure(num=1,figsize=(8,8),dpi=100,facecolor='white')
    ax      = fig.add_subplot(111,projection='3d')
    t2d,x2d = np.meshgrid(t,x)
    ax.plot_surface(x2d,t2d,q,cmap='rainbow')
    ax.set_zlabel('q')
    ax.set_ylabel('t')
    ax.set_xlabel('x')
    ax.tick_params(labelsize=ftsz)

    plt.savefig('figure_3d.eps',format='eps',dpi=1000)

    N       = t.size
    midq    = q[J//2,:]
    
    plt.figure(num=2,figsize=(8,5),dpi=100,facecolor='white')
    plt.subplot(121)
    plt.plot(x,q[:,0],linestyle='-',linewidth=1,color='black',label='T=0')
    plt.plot(x,q[:,N-1],linestyle='-',linewidth=1,color='red',label='T=1')
    plt.xlabel('x')
    plt.ylabel('q')
    plt.legend(fontsize=ftsz)
    plt.tick_params(labelsize=ftsz)

    plt.subplot(122)
    plt.plot(t,midq,linestyle='-',linewidth=1,color='black')
    plt.xlabel('t')
    plt.ylabel('$q_{mid}(t)$')
    plt.tick_params(labelsize=ftsz)

    plt.tight_layout()
   
    plt.savefig('figure_compare.eps',format='eps',dpi=1000)

    plt.show()

#==============================================================

main()

