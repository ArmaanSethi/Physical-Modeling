#===============================
# PDE multigrid solver
#===============================
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
    return np.meshgrid(x,x)

#============================================
# Determines the rms difference between two
# potential fields (excluding boundaries).
def get_rmsdiff(u1,u2):
    J = (u1.shape)[0]
    return np.sqrt(np.mean((u1[1:J+1,1:J+1]-u2[1:J+1,1:J+1])**2))


#============================================
# get analytic solution for homework test problem
def get_analytic(J,minmaxx,problem):
    if (problem == 'plate'):
        x,y = get_mesh(minmaxx,J)
        u = np.zeros((J,J))
        u = np.zeros((J,J)) + 0.5*(1.0-(x**2))
        for i in range(0,J//8):
            k   = float(2*i+1)
            kpi = k*np.pi
            u[:,:] = u[:,:] - (16.0/np.pi**3)*( np.sin(0.5*kpi*(1.0+x))/(k**3*np.sinh(kpi))
                                               *(np.sinh(0.5*kpi*(1.0+y))+np.sinh(0.5*kpi*(1.0-y)))  )
    elif (problem == 'briggs'):
        x,y = get_mesh(minmaxx,J)
        u   = (x**2-x**4)*(y**4-y**2) 
    return u

#--------------------------------------------
# No-charge boundaries.
# Assumes cell-centered quantities (see Zingale 2013).
def bnc_nocharge(f,u):
    J                = u.shape[0]-2
    u[0:J+2,0    ] = -u[0:J+2,1]
    u[0:J+2,J+1  ] = -u[0:J+2,J]
    u[0    ,0:J+2] = -u[1,0:J+2]
    u[J+1  ,0:J+2] = -u[J,0:J+2]
    return u

#============================================
# Given the RHS vector f and a boundary value
# function fBNC, the routine will return the
# solution of A*u=f down to an accuracy of tol.
# input:
#     f    : the RHS of A*u=f. Length J
#     fBNC : pointer to boundary value function.
#            This function should return a grid of
#            dimensions (J+2,J+2), with the first and
#            last indicies (j=0,j=J+1 in each dimension)
#            set to the boundary values.
#     tol  : tolerance (accuracy) for solution. Best
#            defined as rms difference  
#            between current and previous solution. 
#     kwargs: maxit: maximum number of iterations
#             u0   : initial guess for solution.
#-------------------------------------------
def jacobi(f,fBNC,tol,**kwargs):
    maxit = 1000000
    s = f.shape
    if (s[0] != s[1]):
        print('[jacobi]: need square matrix.')
        exit()
    J         = s[0]
    u1        = np.zeros((J+2,J+2)) # initial guess.
    u2        = np.zeros((J+2,J+2))
    for key in kwargs:
        if (key == 'maxit'):
            maxit = kwargs[key]
        if (key == 'u0'):
            u1[1:J+1,1:J+1] = kwargs[key]
            u2[1:J+1,1:J+1] = kwargs[key]
    u1        = fBNC(f,u1)
    diff      = 1e30
    it        = 0
    iind,jind = np.meshgrid(np.arange(J,dtype=int)+1,np.arange(J,dtype=int)+1)
    while((diff > tol) & (it < maxit)):
        u2[:,:]   = u1[:,:]
        u1[:,:]   = fBNC(f,u2)
        u1[iind,jind] = 0.25*(u1[iind-1,jind]+u1[iind+1,jind]+u1[iind,jind-1]+u1[iind,jind+1]) - 0.25*f[iind-1,jind-1]
        it        = it+1
        diff      = get_rmsdiff(u1,u2)
        #print('[jacobi]: it=%5i diff/tol=%13.5e' % (it,diff/tol))
    return u1[1:J+1,1:J+1]

#============================================
# Given the RHS vector f and a boundary value
# function fBNC, the routine will return the
# solution of A*u=f down to an accuracy of tol.
# input:
#     f    : the RHS of A*u=f. Length J
#     fBNC : pointer to boundary value function.
#            This function should return a grid of
#            dimensions (J+2,J+2), with the first and
#            last indicies (j=0,j=J+1 in each dimension)
#            set to the boundary values.
#     tol  : tolerance (accuracy) for solution. Best
#            defined as rms difference  
#            between current and previous solution. 
#     kwargs: maxit: maximum number of iterations
#             u0   : initial guess for solution.
# NOTE: It is highly recommended to use "black-red" indexing
#       for Gauss-Seidel.
#-------------------------------------------
def gauss_seidel(f,fBNC,tol,**kwargs):
    maxit = 1000000
    s = f.shape
    if (s[0] != s[1]):
        print('[gauss_seidel]: need square matrix.')
        exit()
    J         = s[0]
    J2        = (J*J)//2
    J22       = J2//2
    u1        = np.zeros((J+2,J+2)) # initial guess.
    u2        = np.zeros((J+2,J+2))
    for key in kwargs:
        if (key == 'maxit'):
            maxit = kwargs[key]
        if (key == 'u0'):
            u1[1:J+1,1:J+1] = kwargs[key]
            #print('using initial guess')
    u1        = fBNC(f,u1)
    diff        = 1e30
    it          = 0
    
    ir          = np.zeros(J2,dtype=int)
    jr          = np.zeros(J2,dtype=int)
    ir[0:J22]   = 2*(np.arange(J22,dtype=int)//(J//2))
    ir[J22:J2]  = 2*(np.arange(J22,dtype=int)//(J//2))+1
    jr[0:J22]   = 2*(np.arange(J22,dtype=int) % (J//2))
    jr[J22:J2]  = 2*(np.arange(J22,dtype=int) % (J//2))+1
    ib          = np.zeros(J2,dtype=int)
    jb          = np.zeros(J2,dtype=int)
    ib[0:J22]   = 2*(np.arange(J22,dtype=int)//(J//2))
    ib[J22:J2]  = 2*(np.arange(J22,dtype=int)//(J//2))+1
    jb[0:J22]   = 2*(np.arange(J22,dtype=int) % (J//2))+1
    jb[J22:J2]  = 2*(np.arange(J22,dtype=int) % (J//2))
    ir          = ir+1
    jr          = jr+1
    ib          = ib+1
    jb          = jb+1
    while((diff > tol) & (it < maxit)):
        u2[:,:]   = u1[:,:]
        u1[:,:]   = fBNC(f,u2)
        u1[ib,jb] = 0.25*(u1[ib-1,jb]+u1[ib+1,jb]+u1[ib,jb-1]+u1[ib,jb+1]) - 0.25*f[ib-1,jb-1]
        u1[ir,jr] = 0.25*(u1[ir-1,jr]+u1[ir+1,jr]+u1[ir,jr-1]+u1[ir,jr+1]) - 0.25*f[ir-1,jr-1]
        it        = it+1 
        diff      = get_rmsdiff(u1,u2)
        #print('[gauss_seidel]: it=%5i diff/tol=%13.5e' % (it,diff/tol))
    return u1[1:J+1,1:J+1]

#==========================================
# Restricts (=averages) u to half the grid size.
# Relies on cell-centered values.
def mg_restrict(u):
    J     = u.shape[0]
    i,j   = np.meshgrid(np.arange(J//2),np.arange(J//2))   
    uc    = 0.25*(u[2*i,2*j]+u[2*i,2*j+1]+u[2*i+1,2*j]+u[2*i+1,2*j+1])
    return uc

#==========================================
# Prolongates (=interpolates) u to twice the grid size.
# Relies on cell-centered values. See Zingale 2013, eq. 35
# Requires boundary information.
def mg_prolong(u,fBNC):
    J               = u.shape[0]
    u1              = np.zeros((J+2,J+2))
    u1[1:J+1,1:J+1] = u
    u1              = fBNC(u1,u1) # note that the first argument is a dummy argument
    i,j             = np.meshgrid(np.arange(J,dtype=int),np.arange(J,dtype=int)) 
    mi              = 0.5*(u1[2:J+2,1:J+1]-u1[0:J,1:J+1])
    mj              = 0.5*(u1[1:J+1,2:J+2]-u1[1:J+1,0:J])
    uf              = np.zeros((2*J,2*J))
    uf[2*i  ,2*j  ] = u - 0.25*mj - 0.25*mi
    uf[2*i  ,2*j+1] = u + 0.25*mj - 0.25*mi
    uf[2*i+1,2*j  ] = u - 0.25*mj + 0.25*mi
    uf[2*i+1,2*j+1] = u + 0.25*mj + 0.25*mi
    return uf

#==========================================
# Calculates residual r = f - A*v
# Required boundary information.
def mg_residual(u,f,fBNC):
    J               = f.shape[0]
    u1              = np.zeros((J+2,J+2))
    u1[1:J+1,1:J+1] = u
    u1              = fBNC(f,u1)
    r               = f[:,:] - (u1[0:J,1:J+1]+u1[2:J+2,1:J+1]+u1[1:J+1,0:J]+u1[1:J+1,2:J+2]-4.0*u1[1:J+1,1:J+1])
    return r 

#==========================================
# recursive multigrid V-cycle (see page 40/41 of Briggs et al.)
# input: f   : RHS 
#        fBNC: function pointer for boundary conditions
#        npre: pre-smoothing iterations (=number of Gauss-Seidel iterations before V-call)
#        npst: post-smoothing iterations
#        level: level number (just for diagnostics)
#        kwargs: allows you to pass initial guess.
def mg_vcycle(f,fBNC,npre,npst,level,**kwargs):
    verbose = 0
    for key in kwargs:
        if (key == 'verbose'):
            verbose = kwargs[key]
    mu   = 2
    tol  = 1e-8 # control just via npre,npst
    J    = f.shape[0]
    lstr = ''
    for l in range(level):
        lstr = lstr+'    '

    if (J == 1): # reached "bottom": exact solution
        vh      = np.array([[-0.125*f[0,0]]])  # only correct for cell-centered quantities
        #vh     = gauss_seidel(f,fBNC,tol,maxit=4,**kwargs)
    else:
        vh     = gauss_seidel(f,fBNC,tol,maxit=npre,**kwargs) # for repeated V-cycles, this needs kwargs to pass previous guess 
        if (verbose):
            print('[mg_vcycle]: %slevel=%2i: Av%i=f%i -> v%i: |fh%i|=%13.5e' % (lstr,level,J,J,J,J,np.sqrt(np.mean(f**2))))
        rh     = mg_residual(vh,f,fBNC)
        r2h    = mg_restrict(rh)
        if (verbose):
            r = np.sqrt(np.mean(rh**2))
            r2= np.sqrt(np.mean(r2h**2))
            print('[mg_vcycle]: %slevel=%2i: f%i=R(f%i-Av%i): |rh%i|=%13.5e |r2h%i]=%13.5e' % (lstr,level,J//2,J,J,J,r,J//2,r2))
        if (level > 1):
            vh2in = mg_restrict(vh)
        else: 
            vh2in  = np.zeros((J//2,J//2))
        v2h    = mg_vcycle(r2h,bnc_nocharge,npre,npst,level+1,u0=vh2in,verbose=verbose)
        for imu in range(mu-1):
            v2h    = mg_vcycle(r2h,bnc_nocharge,npre,npst,level+1,u0=v2h) 
        if (verbose): # check before correction
            ea = np.sqrt(np.mean(vh**2))
            ra = np.sqrt(np.mean(mg_residual(vh,f,fBNC)**2))
        vh    = vh + mg_prolong(v2h,fBNC)
        if (verbose): # check after correction
            eb = np.sqrt(np.mean(vh**2))
            rb = np.sqrt(np.mean(mg_residual(vh,f,fBNC)**2))
        vh    = gauss_seidel(f,fBNC,tol,maxit=npst,u0=vh)
        if (verbose): # check after smoothing
            e2h = np.sqrt(np.mean(v2h**2))
            eh  = np.sqrt(np.mean(vh**2))
            rc  = np.sqrt(np.mean(mg_residual(vh,f,fBNC)**2))
            if (verbose):
                print('[mg_vcycle]: %slevel=%2i: |e%i|=%13.5e ea%i=%13.5e eb%i=%13.5e ec%i=%13.5e ra%i=%13.5e rb%i=%13.5e rc%i=%13.5e' 
                     % (lstr,level,J//2,e2h,J,ea,J,eb,J,eh,J,ra,J,rb,J,rc))
        r = np.sqrt(np.mean(mg_residual(vh,f,fBNC)**2))
        if (verbose):
            print('[mg_vcycle]: %slevel=%2i: |r%i|=%13.5e max(f%i)=%13.5e' % (lstr,level,J,r,J,np.max(np.abs(f))))
    return vh 

#==========================================
# "Driver" for V-cycle.
# input: f   : RHS vector
#        fBNC: pointer to boundary function
#        tol : dummy argument for consistency with other linear solvers.
# output: Solution u as in A*u=f.
#------------------------------------------
def multigrid(f,fBNC,tol):
    npre = 10
    npst = 10
    J    = f.shape[0]
    if (J % 2 == 1):
        print('[multigrid]: J must be even: J=%4i' % (J))
        exit()
    u    = mg_vcycle(f,fBNC,npre,npst,1,verbose=1)
    for i in range(20):
        u = mg_vcycle(f,fBNC,npre,npst,1,u0=u,verbose=0)
    return u

#==========================================
# initialization
def init(problem,boundary,solver,J):

    if (problem == 'circle'):
        minmaxx = np.array([-1.0,1.0])
        xc,yc   = get_mesh(minmaxx,J)
        rc      = np.sqrt(xc**2+yc**2)
        h       = xc[0,1]-xc[0,0]
        rho = np.exp(-100.0*(rc-0.5)**2)
    elif (problem == 'plate'):
        minmaxx = np.array([-1.0,1.0])
        xc,yc   = get_mesh(minmaxx,J)
        h       = xc[0,1]-xc[0,0]
        rho = np.zeros((J,J))-1.0
    else:
        print('[init]: invalid problem %s' % (problem))
        exit()
    f    = rho*h**2

    if (boundary == 'nocharge'):
        fBNC = bnc_nocharge
    else:
        print('[init]: invalid boundary %s' % (boundary))
        exit()

    if (solver == 'GS'):
        fSOL = gauss_seidel
    elif (solver == 'JC'):
        fSOL = jacobi
    elif (solver == 'MG'):
        fSOL = multigrid
    else:
        print('[init]: invalid solver %s' % (solver))
        exit()

    return f,fBNC,fSOL,minmaxx
         
#==========================================
# (1) build Gauss-Seidel. Test. Realize that it converges slowly. nocharge boundaries. Try periodic. Why slow convergence?
# (2) build multigrid
# (3) uses cell-centered points
# (4) figure out boundaries

def main():
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument("J",type=int,
                        help="number of spatial support points in one dimension (excluding boundaries)")
    parser.add_argument("tol",type=float,
                        help="tolerance")
    parser.add_argument("solver",type=str,
                        help="Poisson solver:\n"
                             "    GS      : Gauss-Seidel\n"
                             "    JC      : Jacobi\n"
                             "    MG      : multigrid")
    parser.add_argument("problem",type=str,
                        help="source field:\n"
                             "    circle  : circular density distribution\n"
                             "    plate   : heated plate")

    args        = parser.parse_args()
    J           = args.J
    tol         = args.tol
    solver      = args.solver
    problem     = args.problem
    boundary    = 'nocharge'

    f,fBNC,fSOL,minmaxx = init(problem,boundary,solver,J)
    u                   = fSOL(f,fBNC,tol)

    if (problem == 'plate'):
        fac = 2
    else:
        fac = 1

    ftsz    = 10
    fig     = plt.figure(num=1,figsize=(10,5*fac),dpi=100,facecolor='white')
    ax      = fig.add_subplot(fac,2,1,projection='3d')
    x2d,y2d = get_mesh(minmaxx,J)
    ax.plot_surface(x2d,y2d,f,cmap='rainbow',rstride=J//16,cstride=J//32)
    ax.set_zlabel('f')
    ax.set_ylabel('x')
    ax.set_xlabel('y')
    ax.tick_params(labelsize=ftsz)
    ax      = fig.add_subplot(fac,2,2,projection='3d')
    ax.plot_surface(x2d,y2d,u,cmap='rainbow',rstride=J//16,cstride=J//32)
    ax.set_zlabel('u')
    ax.set_ylabel('x')
    ax.set_xlabel('y')
    ax.tick_params(labelsize=ftsz)

    if (problem == 'plate'):
        residual = get_analytic(J,minmaxx,problem)-u
        print('[pde_multigrid]: residual for problem %s = %13.5e max(u) = %13.5e' % (problem,np.sqrt(np.mean(residual**2)),np.max(u)))
        ax      = fig.add_subplot(223,projection='3d')
        ax.plot_surface(x2d,y2d,get_analytic(J,minmaxx,problem),cmap='rainbow',rstride=J//16,cstride=J//32)
        ax.set_zlabel('analytic')
        ax.set_ylabel('x')
        ax.set_xlabel('y')
        ax.tick_params(labelsize=ftsz)
        ax      = fig.add_subplot(224,projection='3d')
        ax.plot_surface(x2d,y2d,residual,cmap='rainbow',rstride=J//16,cstride=J//32)
        ax.set_zlabel('residual')
        ax.set_ylabel('x')
        ax.set_xlabel('y')
        ax.tick_params(labelsize=ftsz)

    plt.savefig('figure_3d_%s_%s.eps' % (solver,problem),format='eps',dpi=1000) 

    plt.show()

#==========================================

main()
