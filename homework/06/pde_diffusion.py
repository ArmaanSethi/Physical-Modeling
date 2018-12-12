#============================================
# Partial differential equations: 
# Diffusion problem.
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
# Solver for a tridiagonal matrix.
# a,b,c are the lower, center, and upper diagonals,
# r is the RHS vector.
def tridiag(a,b,c,r):
    n    = b.size
    gam  = np.zeros(n)
    u    = np.zeros(n)
    bet  = b[0]
    u[0] = r[0]/bet 
    for j in range(1,n):
        gam[j] = c[j-1]/bet
        bet    = b[j]-a[j]*gam[j]
        if (bet == 0.0):
            print('[tridiag]: matrix not invertible.')
            exit()
        u[j]   = (r[j]-a[j]*u[j-1])/bet
    for j in range(n-2,-1,-1):
        u[j] = u[j]-gam[j+1]*u[j+1]
    return u

#============================================
# Driver for the actual integrators. Sets the initial conditions
# and generates the support point arrays in space and time.
# input: J      : number of spatial support points
#        dt0    : timestep
#        minmaxx: 2-element array containing minimum and maximum of spatial domain
#        minmaxt: 2-element array, same for time domain
#        fINT   : integrator (one of ftcs, implicit, cranknicholson)
#        fBNC   : boundary condition function
#        fINC   : initial condition function
def diffusion_solve(J,minmaxx,dt0,minmaxt,fINT,fBNC,fINC,**kwargs):
    kappa   = 1.0
    for key in kwargs:
        if (key=='kappa'):
            kappa = kwargs[key]
    # time and space discretization
    N  = int((minmaxt[1]-minmaxt[0])/dt0)+1
    dt = (minmaxt[1]-minmaxt[0])/float(N-1) # recalculate, to make exact
    dx = (minmaxx[1]-minmaxx[0])/float(J)
    x  = minmaxx[0]+(np.arange(J)+0.5)*dx
    t  = minmaxt[0]+np.arange(N)*dt
    # alpha factor
    alpha    = kappa*dt/dx**2
    print('[diffusion_solve]: alpha = %13.5e' % (alpha)) 
    print('[diffusion_solve]: N     = %7i' % (N))
    y        = fINT(x,t,alpha,fBNC,fINC)
    return x,t,y

#--------------------------------------------
# Forward-time centered-space integrator.
# Returns the full solution array (including 
# initial conditions at t=0). Array should be
# of shape (J,N), with J the spatial and N
# the temporal support points.
def ftcs(x,t,alpha,fBNC,fINC):
    J        = x.size
    N        = t.size
    y        = np.zeros((J+2,N))
    # from here ??????
    #Create A, J+2 x J+2 tridiagonal
    A = np.zeros((J+2,J+2))
    A[0,0] = 1-2*alpha
    A[J+1, J+1] = 1-2*alpha
    A[0,1] = alpha
    A[J+1,J] = alpha

    for i in range(J):
        A[i+1,i] = alpha
        A[i+1,i+1] = 1-2*alpha
        A[i+1, i+2] = alpha

    #fINC --> Initial Conditions
    y[1:J+1, 0] = fINC(x)
    #fBNC --> Boundary Conditions
    y[0,0] = fBNC(0,y[:,0])
    y[J+1,0] = fBNC(J+1, y[:,0])    
    for i in range(N-1):
        y[0,i] = fBNC(0,y[:,i])
        y[J+1, i] = fBNC(J+1, y[:,i])
        y[:,i+1] = A@y[:,i]  

    # to here ??????
    return y[1:J+1,:]

#--------------------------------------------
# Fully implicit integrator.
# Returns the full solution array (including 
# initial conditions at t=0). Array should be
# of shape (J,N), with J the spatial and N
# the temporal support points.
# Uses tridiag to solve the tridiagonal matrix.
def implicit(x,t,alpha,fBNC,fINC):
    J        = x.size
    N        = t.size
    y        = np.zeros((J+2,N))
    # from here ??????
    #Create A, J+2 x J+2 tridiagonal
    A = np.zeros((J+2, J+2))
    A[0,0] = 1+2*alpha
    A[J+1,J+1] = 1+2*alpha
    A[0,1] = -alpha
    A[J+1,J] = -alpha
    for i in range(J):
        A[i+1, i] = -alpha
        A[i+1, i+1] = 1+2*alpha
        A[i+1, i+2] = -alpha

    #fINC --> Initial Conditions
    y[1:J+1,0] = fINC(x)
    #fBNC --> Boundary Conditions
    y[0,0] = fBNC(0,y[:,0])
    y[J+1,0] = fBNC(J+1,y[:,0])
    for i in range(N-1):
        y[0,i] = fBNC(0,y[:,i])
        y[J+1,i] = fBNC(J+1,y[:,i])
        y[:,i+1] = np.linalg.solve(A,y[:,i])

    # to here ??????
    return y[1:J+1,:]

#--------------------------------------------
# Crank-Nicholson integrator.
# Returns the full solution array (including 
# initial conditions at t=0). Array should be
# of shape (J,N), with J the spatial and N
# the temporal support points.
# Uses tridiag to solve the tridiagonal matrix.
def cranknicholson(x,t,alpha,fBNC,fINC):
    J        = x.size
    N        = t.size
    y        = np.zeros((J+2,N))
    # from here ??????
    #A is n+1, B is n
    A = np.zeros((J+2,J+2))
    B = np.zeros((J+2,J+2))

    A[0,0] = 1+alpha
    A[J+1,J+1] = 1+alpha
    A[0,1] = -0.5*alpha
    A[J+1,J] = -0.5*alpha

    B[0,0] = 1-alpha
    B[J+1,J+1] = 1-alpha
    B[0,1] = 0.5*alpha
    B[J+1,J] = 0.5*alpha

    for i in range(J):
        A[i+1,i] = -0.5*alpha
        A[i+1,i+1] = 1+alpha
        A[i+1,i+2] = -0.5*alpha

        B[i+1,i] = 0.5*alpha
        B[i+1,i+1] = 1-alpha
        B[i+1,i+2] = 0.5*alpha

    #fINC --> Initial Conditions
    y[1:J+1,0] = fINC(x)
    #fBNC --> Boundary Conditions
    y[0,0] = fBNC(0,y[:,0])
    y[J+1,0] = fBNC(J+1,y[:,0])
    
    for i in range(N-1):
        y[0,i] = fBNC(0,y[:,i])
        y[J+1,i] = fBNC(J+1,y[:,i])
        y[:,i+1] = np.linalg.solve(A,B@y[:,i])

    # to here ??????
    return y[1:J+1,:]

#============================================

def init(solver,problem):
    if (solver == 'ftcs'):
        fINT = ftcs
    elif (solver == 'implicit'):
        fINT = implicit
    elif (solver == 'CN'):
        fINT = cranknicholson
    else:
        print('[init]: invalid solver %s' % (solver))
 
    if (problem == 'const'):
        fINC    = Tconst
        fBNC    = Bdirichlet
        minmaxx = np.array([-0.5,0.5])
        minmaxt = np.array([0.0,1.0])
    elif (problem == 'spike'):
        fINC    = Tspike
        fBNC    = Bdirichlet
        minmaxx = np.array([-0.5,0.5])
        minmaxt = np.array([0.0,1.0])
    elif (problem == 'random'):
        fINC    = Trandom
        fBNC    = Bperiodic
        minmaxx = np.array([-0.5,0.5])
        minmaxt = np.array([0.0,1.0])
    else:
        print('[init]: invalid problem %s' % (problem))

    return fINT,fBNC,fINC,minmaxx,minmaxt 

#============================================
# functions for setting the initial conditions (T....)
# and the boundary conditions (B.....)
def Tconst(x):
    return np.zeros(x.size)+1.0

def Tspike(x):
    return np.exp(-10.0*x**2)

def Trandom(x):
    return np.random.rand(x.size)+1.0

def Bdirichlet(iside,y):
    if (iside==0):
        return -y[1]
    else:
        return -y[y.size-2]

def Bperiodic(iside,y):
    if (iside==0):
        return y[y.size-2]
    else:
        return y[1]

#============================================
# returns a (J,N) array with the analytic solution
# to the diffusion problem (see Hancock 2006).
def diffusion_analytic(x,t):
    dx        = x[1]-x[0]
    xmin      = x[0]-0.5*dx # need the true xmin
    xmax      = x[x.size-1]+0.5*dx
    J         = x.size
    N         = t.size
    yana      = np.zeros((J,N))
    # from here ??????
    X = (x-xmin)/(xmax-xmin)
    for t_index in range(N):
        for x_index in range(J):
            summation = 0
            for n in range(50):
                summation += np.sin((2*n+1)*np.pi*(X[x_index]))/(2*n+1)*np.exp(-1*(2*n+1)**2*np.pi**2*t[t_index])
            yana[x_index,t_index] = (4/np.pi)*summation

    # to here ??????
    return yana

#============================================
def main():

    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument("J",type=int,
                        help="number of spatial support points (including boundaries)")
    parser.add_argument("dt",type=float,
                        help="timestep")
    parser.add_argument("solver",type=str,
                        help="diffusion equation solver:\n"
                             "    ftcs    : forward-time centered-space\n"
                             "    implicit: fully implicit\n"
                             "    CN      : Crank-Nicholson")
    parser.add_argument("problem",type=str,
                        help="initial conditions:\n"
                             "    const   : constant temperature\n"
                             "    spike   : peaked gaussian\n"
                             "    random  : random noise\n")

    args         = parser.parse_args()
    J            = args.J
    dt           = args.dt
    solver       = args.solver
    problem      = args.problem

    fINT,fBNC,fINC,minmaxx,minmaxt = init(solver,problem)
    x,t,y        = diffusion_solve(J,minmaxx,dt,minmaxt,fINT,fBNC,fINC)

    ftsz    = 10
    fig     = plt.figure(num=1,figsize=(8,8),dpi=100,facecolor='white')
    ax      = fig.add_subplot(111,projection='3d')
    t2d,x2d = np.meshgrid(t,x)
    ax.plot_surface(x2d,t2d,y,cmap='rainbow')
    ax.set_zlabel('T')
    ax.set_ylabel('t')
    ax.set_xlabel('x')
    ax.tick_params(labelsize=ftsz)

    plt.savefig('figure_3d.eps',format='eps',dpi=1000)

    if (problem=='const'):
        yana         = diffusion_analytic(x,t)
        rmsdiff      = np.sqrt(np.mean((y-yana)**2,axis=0))
        maxyana      = np.max(yana,axis=0)
        maxy         = np.max(y,axis=0)
        plt.figure(num=2,figsize=(8,8),dpi=100,facecolor='white')
        plt.subplot(221)
        plt.plot(t,maxyana,linestyle='-',linewidth=1,color='black',label='analytic')
        plt.plot(t,maxy,linestyle='-',linewidth=1,color='red',label='integrated')
        plt.xlabel('t')
        plt.ylabel('max(y)')
        plt.legend(fontsize=ftsz)
        plt.tick_params(labelsize=ftsz)

        plt.subplot(222)
        plt.plot(t,np.log10(np.abs(maxy-maxyana)),linestyle='-',linewidth=1,color='black')
        plt.xlabel('t')
        plt.ylabel('|max(yana)-max(y)|')
        plt.tick_params(labelsize=ftsz)

        plt.subplot(223)
        plt.plot(t,np.log10(rmsdiff),linestyle='-',linewidth=1,color='black')
        plt.xlabel('t')
        plt.ylabel('rms difference')
        plt.tick_params(labelsize=ftsz) 

        plt.subplot(224)
        plt.plot(x,yana[:,0],linestyle='-',linewidth=1,color='red',label='analytic')
        plt.plot(x,y[:,0],linestyle='-',linewidth=1,color='black',label='integrated')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend(fontsize=ftsz)
        plt.tick_params(labelsize=ftsz)

        plt.tight_layout()

        plt.savefig('figure_compare.eps',format='eps',dpi=1000)

    plt.show()
        
#========================================

main()


