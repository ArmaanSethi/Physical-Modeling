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
# Generates mesh assuming cell-centered support points.
def get_mesh(minmaxx,J):
    dx    = (minmaxx[1]-minmaxx[0])/float(J)
    x     = minmaxx[0]+(np.arange(J)+0.5)*dx
    return np.meshgrid(x,x)

#============================================
# get analytic solution for plate problem
# input: J: number of support points
#        minmaxx: 2-element array with minimum and maximum grid extent.
#        problem: string containing problem name.
def get_analytic(J,minmaxx,problem):
    u = np.zeros((J,J)) #modified so it doesnt break for circle
    if (problem == 'plate'):
        x,y = get_mesh(minmaxx,J)
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
#............
    J = f.shape[0]
    done = False
    u = np.zeros((J+2, J+2))
    uNew = np.zeros((J+2,J+2)) #temp
    i = 0
    max_iters = 10000
    output = False
    for key in kwargs:
        if key == 'maxit':
            max_iters = kwargs[key]
        elif key == 'u0':
            u[1:J+1,1:J+1] = kwargs[key]
        if key == 'solver':
            if kwargs[key] == 'JC':
                output = True

    while not done:
        uNew[1:J+1, 1:J+1] = (0.25)*(u[0:J,1:J+1]+u[2:J+2, 1:J+1]+u[1:J+1, 0:J]+u[1:J+1,2:J+2]-f)
        uNew = fBNC(0,uNew).copy()
        rmse = np.sqrt(np.mean((uNew - u)**2))
        u = uNew.copy()
        if rmse < tol or i > max_iters: 
            done = True
        i+=1

        if output and i %100==0:
            print("Iterations:\t", i, "\t\tResidual:\t", rmse)
    if output:
        print("Iterations:\t", i, "\t\tResidual:\t", rmse)
    return u[1:-1,1:-1]

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
#..............
    J = f.shape[0]
    done = False
    u = np.zeros((J+2, J+2))
    uNew = np.zeros((J+2,J+2)) #temp
    i = 0
    max_iters = 10000
    output = False
    for key in kwargs:
        if key == 'maxit':
            max_iters = kwargs[key]
        elif key == 'u0':
            uNew[1:J+1,1:J+1] = kwargs[key]
        if key == 'solver':
            if kwargs[key] == 'GS':
                output = True

    while not done:
        for m in range(1,J):
            for k in range(1,J):
                uNew[k][m] = (0.25) * (uNew[k + 1][m] + uNew[k - 1][m] + uNew[k][m - 1] + uNew[k][m + 1] - f[k][m])
                uNew = fBNC(0,uNew)
        rmse = np.sqrt(np.mean((uNew - u)**2))
        u = uNew.copy()
        i+=1
        if rmse < tol or i > max_iters: 
            done = True

        if output and i %100==0:
            print("Iterations:\t", i, "\t\tResidual:\t", rmse)
    if output:
        print("Iterations:\t", i, "\t\tResidual:\t", rmse)

    return u[1:-1,1:-1]

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
# Calculates residual r = f - A*u
# Requires boundary information.
def mg_residual(u,f,fBNC):
#...............
    J = u.shape[0]
    u1 = np.zeros((J+2, J+2))
    u1[1:-1, 1:-1] = u
    u1 = fBNC(f,u1)
    rhs = -4*u1[1:-1, 1:-1] + u1[2:, 1:-1]+u1[0:-2,1:-1] + u1[1:-1,0:-2] +u1[1:-1,2:]
    r = f - rhs
    return r


#==========================================
# recursive multigrid V-cycle (see page 40/41 of Briggs et al.)
# input: f   : RHS 
#        fBNC: function pointer for boundary conditions
#        npre: pre-smoothing iterations (=number of Gauss-Seidel iterations before V-call)
#        npst: post-smoothing iterations
#        level: level number (just for diagnostics)
#        kwargs: allows you to pass initial guess.
def mg_vcycle(f,fBNC,npre,npst,level,tol,**kwargs): #I passed in tol
#................
    for key in kwargs:
        if key == 'maxit':
            max_iters = kwargs[key]
        elif key == 'u0':
            vh = kwargs[key]

    if(f.shape[0] == 1): #base case
        return vh
    vh    = jacobi(f, fBNC, tol, u0=vh, maxit=npre) # pre-conditioning
    # vh    = gauss_seidel(f, fBNC, tol, u0=vh, maxit=npre) # pre-conditioning
    rh    = mg_residual(vh, f, fBNC) # get the residual
    r2h   = mg_restrict(rh) # restrict the RHS
    e2h   = mg_restrict(vh) # cheap way to get correct dimensions without querying array size etc
    e2h[:]= 0.0 # error guess is 0.0 
    e2h  = mg_vcycle(r2h, fBNC, npre, npst, level+1, tol, u0=e2h)
    eh    = mg_prolong(e2h,fBNC) # prolong to higher-resolution grid
    vh    = vh + eh # add correction from coarser grid to finer grid
    vh    = jacobi(f, fBNC, tol, u0=vh, maxit=npst) # smooth noise introduced by prolongation
    # vh    = gauss_seidel(f, fBNC, tol, u0=vh, maxit=npst) # smooth noise introduced by prolongation
    return vh
#==========================================
# "Driver" for V-cycle. Improves on one single 
# V-cycle result by repeatedly calling mg_vcycle, using
# the previous result as initial guess.
# input: f   : RHS vector
#        fBNC: pointer to boundary function
#        tol : dummy argument for consistency with other linear solvers.
# output: Solution u as in A*u=f.
#------------------------------------------
def multigrid(f,fBNC,tol, **kwargs):
#................
    J = f.shape[0]
    done = False
    it = 0
    maxit = 10000
    npre = 10
    npst = 10
    u = np.zeros((J,J))

    while not done:
        # print("ITERATION:\t", it)
        uNew = mg_vcycle(f, fBNC, npre, npst, 0, tol, u0 = u)
        res = np.sqrt(np.mean((uNew - u)**2))
        u = uNew.copy()
        it+=1
        print("Iteration:\t",it,"\tResidual:\t",res)
        if((res < tol) or (it > maxit)):
            done = True

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
    u                   = fSOL(f,fBNC,tol, solver = solver)

    # plots and results go here
    #................

    ftsz    = 10
    fig     = plt.figure(num=1,figsize=(8,8),dpi=100,facecolor='white')

    ax      = fig.add_subplot(221,projection='3d')
    ax.set_title("f")
    t2d,x2d = get_mesh(minmaxx,J)
    ax.plot_surface(x2d,t2d,f,cmap='rainbow')
    ax.set_zlabel('f')
    ax.set_ylabel('y')
    ax.set_xlabel('x')
    ax.tick_params(labelsize=ftsz)

    ax2      = fig.add_subplot(222,projection='3d')
    t2d,x2d = get_mesh(minmaxx,J)
    ax2.set_title("u")
    ax2.plot_surface(x2d,t2d,u,cmap='rainbow')
    ax2.set_zlabel('u')
    ax2.set_ylabel('y')
    ax2.set_xlabel('x')
    ax2.tick_params(labelsize=ftsz)

    ax3     = fig.add_subplot(223,projection='3d')
    t2d,x2d = get_mesh(minmaxx,J)
    if problem == 'circle':
        ax3.set_title("Analytical Solution Not Applicable")
    else:
        ax3.set_title("Analytic")
    z_an = get_analytic(J, minmaxx, problem) #I modified get_analytic so it does not break with problem=circle
    ax3.plot_surface(x2d,t2d,z_an,cmap='rainbow')
    ax3.set_zlabel('analytic')
    ax3.set_ylabel('y')
    ax3.set_xlabel('x')
    ax3.tick_params(labelsize=ftsz)

    ax4     = fig.add_subplot(224,projection='3d')
    t2d,x2d = get_mesh(minmaxx,J)
    if problem == 'circle':
        ax4.set_title("Residual Not Applicable")
    else:
        ax4.set_title("Residual")
    z_an = get_analytic(J, minmaxx, problem)
    ax4.plot_surface(x2d,t2d,abs(u-z_an),cmap='rainbow')
    ax4.set_zlabel('residual')
    ax4.set_ylabel('y')
    ax4.set_xlabel('x')
    ax4.tick_params(labelsize=ftsz)

    plt.show()

#==========================================

main()
