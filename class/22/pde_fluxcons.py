#============================================
# Partial differential equations: 
# Advection problem in 1D.
# Demonstration of flux-conservative problems: reconstruction
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
# Generate mesh (cell-centered)
def get_mesh(minmaxx,J):
    dx    = (minmaxx[1]-minmaxx[0])/float(J)
    x     = minmaxx[0]+(np.arange(J)+0.5)*dx
    return x

#============================================
# Boundary condition functions (cell-centered)
# Periodic. First active cell is nghost, last active cell is nghost+J-1
# All array operations
def bnc_periodic(grd,q):
    j             = np.arange(grd.nghost)
    q[j         ] = q[grd.je-grd.nghost+j+1]
    q[grd.je+j+1] = q[grd.js+j]
    return q

#--------------------------------------------
# no charge boundaries
def bnc_nocharge(u):
    j             = np.arange(grd.nghost)
    q[grd.js-1-j] = -q[grd.js+j]
    q[grd.je+1+j] = -q[grd.je-j]
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
    q = np.exp(-100.0*x**2)
    return q

#============================================
# Solver for advection problem.
# Advances solution by one single time step.
# A single step consists
# input: 
#     q      : initial condition at time t
#     grd    : object of class grid
#     t      : current time 
#     dt     : timestep
#     c      : advection speed
#     fBNC   : boundary condition function
# output:
#     qnew   : solution at t+dt
#--------------------------------------------
def rsolver(q,grd,t,dt,c,fBNC):
    # Step 0: set the boundary conditions
    # ????

    # Step 1: reconstruct q on cell walls, accounting for t+0.5*dt
    # ????

    # Step 2: calculate the fluxes based on reconstructed q. The fluxes live on the cell walls.
    # ????

    # Step 3: update. Fluxes indexed by jv, since we need only the active grid cells.
    qnew       = q[grd.jv] - (dt/grd.dx)*(f[grd.jv-grd.js+1]-f[grd.jv-grd.js])
    return qnew

#--------------------------------------------
# calculates slopes including limiters for reconstruction
# input:
#     q  : quantity to be reconstructed
#     grd: object of class grid
# output:
#     sl : slopes at cell walls, indexed in same way as fluxes
# notes:
#   The array sl is defined on the cell walls BETWEEN all volume-centered
#   cells INCLUDING the ghost zones. Therefore it has length grd.Jtot-1.
#   The slope at j-1/2 is sl[j] = (q[j]-q[j-1])/dx, and it can be addressed
#   by sl[grd.jf]. Similarly, q[j] can be addressed by q[grd.jf] etc.
#--------------------------------------------
def slopes(q,grd):
    sl   = np.zeros(grd.Jtot-1)
    if (grd.recon == 'centered'): # 2nd order derivative centered on j
        # ????

    elif (grd.recon == 'upwind'): # 1st order derivative centered on j-1/2
        # ????

    elif (grd.recon == 'downwind'): # 1st order derivative centered on j+1/2
        # ????

    elif (grd.recon == 'minmod'): # minmod limiter 
        sll = np.zeros(grd.Jtot-1)
        slr = np.zeros(grd.Jtot-1)
        sll[grd.jf] = (q[grd.jf  ]-q[grd.jf-1])/grd.dx
        slr[grd.jf] = (q[grd.jf+1]-q[grd.jf  ])/grd.dx
        sl [grd.jf] = (sll[grd.jf]*slr[grd.jf] > 0.0)*(  sll[grd.jf]*(np.abs(sll[grd.jf]) <  np.abs(slr[grd.jf])) 
                                                       + slr[grd.jf]*(np.abs(sll[grd.jf]) >= np.abs(slr[grd.jf])))
    elif (grd.recon == 'superbee'): # superbee, or maxmod limiter
        sll = np.zeros(grd.Jtot-1)
        slr = np.zeros(grd.Jtot-1)
        sl1 = np.zeros(grd.Jtot-1)
        sl2 = np.zeros(grd.Jtot-1)
        sll[grd.jf] = (q[grd.jf  ]-q[grd.jf-1])/grd.dx
        slr[grd.jf] = (q[grd.jf+1]-q[grd.jf  ])/grd.dx
        sl1[grd.jf] = (sll[grd.jf]*slr[grd.jf] > 0.0)*(      slr[grd.jf]*(np.abs(    slr[grd.jf]) <  np.abs(2.0*sll[grd.jf])) 
                                                       + 2.0*sll[grd.jf]*(np.abs(    slr[grd.jf]) >= np.abs(2.0*sll[grd.jf])))
        sl2[grd.jf] = (sll[grd.jf]*slr[grd.jf] > 0.0)*(  2.0*slr[grd.jf]*(np.abs(2.0*slr[grd.jf]) <  np.abs(    sll[grd.jf]))
                                                           + sll[grd.jf]*(np.abs(2.0*slr[grd.jf]) >= np.abs(    sll[grd.jf])))
        sl [grd.jf] = (sl1[grd.jf]*sl2[grd.jf] > 0.0)*(  sl1[grd.jf]*(np.abs(sl1[grd.jf]) >  np.abs(sl2[grd.jf])) 
                                                       + sl2[grd.jf]*(np.abs(sl1[grd.jf]) <= np.abs(sl2[grd.jf])))
    return sl

#--------------------------------------------
# reconstructs quantity q on cell walls
# input:
#      q   : quantity to be reconstructed
#      c   : advection speed (needed for higher-order reconstruction)
#      grd : object of class grid
#      dt  : timestep
# output:
#      quantity q on cell walls
#--------------------------------------------
def reconstruct(q,c,grd,dt):
    sl     = slopes(q,grd) # using slope limiters
    qw     = q[grd.jf] + 0.5*sl[grd.jf]*(grd.dx-c*dt)
    return qw

#--------------------------------------------
# calculates the flux on the cell walls at t+0.5*dt
# input:
#      qw: quantity q on cell walls
#      c : advection speed
# output:
#      f : fluxes on cell walls
#--------------------------------------------
def flux(qw,c):
    f = c*qw
    return f

#============================================
# class grid containing everything necessary
# for defining the spatial grid including ghost
# zones and active volume and face cells
class grid:
    def __init__(self,J,recon,minmaxx):
        self.nghost= 2
        self.J     = J
        self.Jtot  = J+2*self.nghost
        self.recon = recon
        self.js    = self.nghost 
        self.je    = J+self.nghost-1
        self.jv    = np.arange(J)+self.nghost # volume-centered active cells
        self.jf    = np.arange(J+1)+self.js-1 # face-centered active cells (flux positions)
        self.xmin  = minmaxx[0]
        self.xmax  = minmaxx[1]
        self.xlen  = self.xmax-self.xmin
        self.dx    = self.xlen/float(self.J)
        self.x     = self.xmin+(np.arange(self.J)+0.5)*self.dx # this is the active grid

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
#        recon  : reconstruction type
# output: grid object, time array, q in space and time
#------------------------------------------
def advection_solve(J,minmaxt,minmaxx,cfl,fSOL,fINC,fBNC,recon):
    # time and space discretization
    grd             = grid(J,recon,minmaxx)
    cadv            = 1.0 # advection velocity
    dt              = cfl*grd.dx/np.abs(cadv)
    N               = int((minmaxt[1]-minmaxt[0])/dt+0.0001)
    dt              = (minmaxt[1]-minmaxt[0])/float(N) # recalculate, to make exact
    t               = minmaxt[0]+np.arange(N+1)*dt
    alpha           = dt*cadv/grd.dx
    print('[advection_solve]: alpha = %13.5e N = %7i dt = %13.5e dx = %13.5e' % (alpha,N,dt,grd.dx))
    q               = np.zeros((grd.Jtot,N+1))
    q[grd.jv,0]     = fINC(grd.x)
    for n in range(N):
        print('[advection_solve]: n = %4i t = %13.5e' % (n,t[n]))
        q[grd.jv,n+1] = fSOL(q[:,n],grd,t[n],dt,cadv,fBNC) 
    return grd,t,q[grd.jv,:]

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

    fSOL = rsolver

    return fSOL,fINC,fBNC,minmaxt,minmaxx

#============================================
def main():

    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument("J",type=int,
                        help="number of spatial support points (excluding boundaries)")
    parser.add_argument("cfl",type=float,
                        help="Courant-Friedrich-Levy (CFL) number")
    parser.add_argument("problem",type=str,
                        help="initial conditions:\n"
                             "    tophat  : top hat density profile\n"
                             "    gaussian: Gaussian density profile")
    parser.add_argument("-r","--recon",type=str,default='donor',help="reconstruction type")

    args         = parser.parse_args()
    J            = args.J
    cfl          = args.cfl
    solver       = 'riemann' # args.solver
    problem      = args.problem
    recon        = args.recon

    fSOL,fINC,fBNC,minmaxx,minmaxt = init(problem,solver)
    grd,t,q                        = advection_solve(J,minmaxx,minmaxt,cfl,fSOL,fINC,fBNC,recon)

    # 3D plot (space-time) of q
    ftsz    = 10
    fig     = plt.figure(num=1,figsize=(9,7),dpi=100,facecolor='white')
    ax      = fig.add_subplot(221,projection='3d')
    t2d,x2d = np.meshgrid(t,grd.x)
    ax.plot_surface(x2d,t2d,q,cmap='rainbow')
    ax.set_zlabel('q')
    ax.set_ylabel('t')
    ax.set_xlabel('x')
    ax.tick_params(labelsize=ftsz)

    N       = t.size
    midq    = q[J//2,:]

    # calculate the total variation
    totvar  = np.sum(np.abs(q[1:,:]-q[:-1,:]),axis=0)
    
    # initial condition and result after one complete round-trip
    plt.subplot(222)
    plt.plot(grd.x,q[:,0],linestyle='-',linewidth=1,color='black',label='T=0')
    plt.plot(grd.x,q[:,N-1],linestyle='-',linewidth=1,color='red',label='T=1')
    print(t[N-1])
    plt.xlabel('x')
    plt.ylabel('q')
    plt.legend(fontsize=ftsz)
    plt.tick_params(labelsize=ftsz)

    # total variation (sum of absolute differences)
    plt.subplot(223)
    plt.plot(t,totvar,linestyle='-',linewidth=1,color='black')
    plt.xlabel('t')
    plt.ylabel('TV')
    plt.tick_params(labelsize=ftsz)

    # value of center point against time
    plt.subplot(224)
    plt.plot(t,midq,linestyle='-',linewidth=1,color='black')
    plt.xlabel('t')
    plt.ylabel('$q_{mid}(t)$')
    plt.tick_params(labelsize=ftsz)

    plt.tight_layout()
   
    plt.savefig('figure_compare.eps',format='eps',dpi=1000)

    plt.show()

#==============================================================

main()

