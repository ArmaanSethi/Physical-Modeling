#===========================================
import argparse                  # allows us to deal with arguments to main()
from argparse import RawTextHelpFormatter
import numpy as np
import matplotlib.pyplot as plt
import p358utilities as util
#===========================================
# setting constants
hbar = 1.0
m    = 1.0
omega= 1.0

#===========================================
# These are the potential functions, and
# the total hamiltonian. 
#-------------------------------------------

def potential_free(x0):
    if (isinstance(x0,np.ndarray)):
        return np.zeros(x0.size)
    else:
        return 0.0

def potential_harmonic(x0):
    return 0.5*x0**2

def potential_pertharmonic(x0):
    l = 1.0
    return 0.5*x0**2 + l*x0**4/24.0 

def potential_instanton(x0):
    l = 1e2
    a = 1.0
    return l*(x0**2-a**2)**2

def potential_box(x0):
    return  1e6*((1.0+np.tanh((x0-0.5)/1e-3))+(1.0-np.tanh((x0+0.5)/1e-3))) 

def hamiltonian(x0,x1,eps,fPOT):
    T = 0.5*((x1-x0)/eps)**2 
    V = fPOT(x0)
    return T+V

#===========================================
# X = integrate_path(fPOT,xa,xb,ta,tb,delta,N,ncut,T)
# Returns X(N,T-ncut),E(N,T-ncut) containing all path positions between (xa,xb),
# for all iterations beyond the "burn-in" phase set by ncut, and the corresponding
# total energies. 
#  
# input : fPOT   : function pointer to potential V(x). Note that the total energy
#                  E(x_i,x_{i+1}) is calculated by the function hamiltonian just
#                  above the function integrate_path.
#         xa     : starting position
#         xb     : end position
#         ta     : starting time
#         tb     : end time
#         delta  : maximum random stepsize for path modification
#         N      : number of time steps
#         ncut   : number of "burn-in" steps to be removed from result
#         T      : maximum number of path modifications
# output:
#         X      : an (N,T-ncut) array of all path positions. A single path
#                  at iteration i is defined by X[:,i], and the position
#                  along a single path is X[j,i].
#         E      : an (N,T-ncut) array of the energies at each path position. 
#                  This will be used to calculate the (ground state) energy.
#-------------------------------------------
def integrate_path(fPOT,xa,xb,ta,tb,delta,N,ncut,T):
    # from here ????
    eps          = (tb-ta)/float(N) # "time step"
    imod         = np.zeros(N+2)
    X            = np.zeros((N+2,T))
    E            = np.zeros((N+1,T))
    Epath        = np.zeros(T)
    X[0,0]       = xa # boundaries
    X[N+1,0]     = xb
    X[1:N+1,0]   = 2.0*np.random.rand(N)-1.0
    irand        = np.random.randint(1,high=N+1,size=T) # choosing x_i to be modified
    urand        = np.random.rand(T) # for the displacements

    for n in range(1,N+2):
        E[n-1,0]  = hamiltonian(X[n-1,0],X[n,0],eps,fPOT)
    Epath[0]   = np.mean(E[:,0])

    it     = 1
    iacc   = 0
    while (it < T):
        i       = irand[it] # this is an ndim vector
        im1     = i-1
        ip1     = i+1
        xt      = X[i,it-1]+delta*(2.0*urand[it]-1.0) # eq. 4.31, trial        
        de      = ( hamiltonian(X[im1,it-1],xt       ,eps,fPOT)+hamiltonian(xt       ,X[ip1,it-1],eps,fPOT)
                   -hamiltonian(X[im1,it-1],X[i,it-1],eps,fPOT)-hamiltonian(X[i,it-1],X[ip1,it-1],eps,fPOT)) # eq. 4.35
        X[:,it] = X[:,it-1] # copy old path into next slot
        E[:,it] = E[:,it-1]
        if (de < 0.0):
            X[i,it]   = xt # definitely save new value
            E[i-1,it] = E[i-1,it] + de
            imod[i]   = imod[i]+1.0
            iacc      = iacc+1
        else:
            u = np.random.rand(1)
            if (u < np.exp(-eps*de/hbar)):
                X[i,it]   = xt        # save new value
                E[i-1,it] = E[i-1,it] + de
                imod[i]   = imod[i]+1.0
                iacc      = iacc + 1
        Epath[it] = np.mean(E[:,it])*eps
        it = it+1

    return X[:,ncut:T],E[:,ncut:T]*eps
    # to here ????
     
#===========================================
# fPOT,xa,xb,ta,tb,ncut = init(s_prob)
# Returns model parameters depending on problem name
# input : s_prob : string describing problem
# output: N      : number of timesteps
#         ta     : starting time
#         tb     : end time
#         xa     : starting position
#         xb     : end position
#         fPOT   : function pointer to potential
def init(s_prob):
    if (s_prob == 'free'):
        ncut = 1000
        ta   = 0.0
        tb   = 1e2
        xa   = 0.0
        xb   = 0.0
        fPOT = potential_free
    elif (s_prob == 'harmonic'):
        ncut = 1500
        ta   = 0.0
        tb   = 1e2
        xa   = 0.0
        xb   = 0.0
        fPOT = potential_harmonic
    elif (s_prob == 'pertharmonic'):
        ncut = 1500
        ta   = 0.0
        tb   = 1e2
        xa   = 0.0
        xb   = 0.0
        fPOT = potential_pertharmonic
    elif (s_prob == 'instanton'):
        ncut = 1500
        ta   = 0.0
        tb   = 1e2
        xa   = 0.0
        xb   = 0.0
        fPOT = potential_instanton
    elif (s_prob == 'box'):
        ncut = 1500
        ta   = 0.0
        tb   = 1e0
        xa   = 0.0
        xb   = 0.0
        fPOT = potential_box
    else:
        raise Exception("[init]: invalid s_prob = %s" % (s_prob))

    return fPOT,xa,xb,ta,tb,ncut

#=============================================
# returns analytical solutions for probabilities
def get_psi2(x,s_prob):
    psi2 = x-x
    if (s_prob == 'harmonic'):
        psi2 = (np.exp(-0.5*x**2)/np.sqrt(np.sqrt(np.pi)))**2
    elif (s_prob == 'box'):
        psi2 = (np.sqrt(2.0)*np.sin(np.pi*(x+0.5)))**2
    return psi2

#=============================================
# plots the results
def check(X,E,s_prob,fPOT):
    s          = X.shape
    N          = s[0] # number of time steps
    R          = s[1] # number of realizations

    xmin = np.min(X)
    xmax = np.max(X)

    if (   (s_prob == 'free') or (s_prob == 'harmonic') 
        or (s_prob == 'pertharmonic') or (s_prob == 'instanton') or (s_prob == 'box')):
        xmin       = np.min(xmin)
        xmax       = np.max(xmax)
        print("[check]: [xmin,xmax]=%13.5e,%13.5e" % (xmin,xmax))
        nbin       = 200
        hist,edges = np.histogram(X[1:N-1,:],nbin,range=(xmin,xmax),normed=False)
        xbin       = 0.5*(edges[0:edges.size-1]+edges[1:edges.size])
        tothist    = np.sum(hist.astype(float))
        hist       = hist.astype(float)/tothist
        print('sum histogram: %13.5e' % (np.sum(hist)))
        psi2       = hist*float(nbin)/(xmax-xmin)
        psi2ana    = get_psi2(xbin,s_prob)
        if (np.max(psi2ana) > 0.0):
            rmserr = np.sqrt(np.mean((psi2-psi2ana)**2))
        else:
            rmserr = None
        pot        = fPOT(xbin)
        print('int psi2     : %13.5e' % (np.sum(psi2)*(xbin[1]-xbin[0])))
        EX         = np.mean(X)
        sX         = np.std(X)
        EE         = np.mean(np.mean(E,axis=0))
        sE         = np.std(np.mean(E,axis=0))
        print("[check]: E[X] = %13.5e+-%13.5e  E[E]=%13.5e+-%13.5e" % (EX,sX,EE,sE))

        ftsz = 10
        # histogram of X - should be proportional to probability to find particle at given x.
        plt.figure(num=1,figsize=(8,8),dpi=100,facecolor='white')
        plt.subplot(221)
        plt.bar(xbin,hist,width=(xbin[1]-xbin[0]),facecolor='green',align='center')
        plt.xlabel('$x$',fontsize=ftsz)
        plt.ylabel('$h(x)$',fontsize=ftsz)
        plt.tick_params(labelsize=ftsz)
        # the wave function - this needs to be appropriately normalized. See above.
        plt.subplot(222)
        plt.scatter(xbin,psi2,linewidth=1,color='red')
        if (rmserr != None):
            plt.plot(xbin,psi2ana,linewidth=1,linestyle='-',color='black',label='analytic')
            plt.title('$<\Delta^2>^{1/2}$=%10.2e' % (rmserr),fontsize=ftsz)
        plt.xlabel('$x$',fontsize=ftsz)
        plt.ylabel('$|\psi(x)|^2$',fontsize=ftsz)
        plt.legend(fontsize=ftsz)
        util.rescaleplot(xbin,psi2,plt,0.05)
        plt.tick_params(labelsize=ftsz)
        # the potential (just for informational purposes)
        plt.subplot(223)
        plt.plot(xbin,pot,linewidth=1,linestyle='-',color='black',label='$V(x)$')
        plt.xlabel('$x$',fontsize=ftsz)
        plt.ylabel('$V(x)$',fontsize=ftsz)
        util.rescaleplot(xbin,pot,plt,0.05)
        plt.tick_params(labelsize=ftsz)
        plt.show()

        

#=============================================
def main():

    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument("s_prob",type=str,
                        help="problem name:\n"
                             "   free     : free, stationary electron\n"
                             "   harmonic : harmonic oscillator\n"
                             "   box      : box potential\n"
                             "   instanton: double-well potential")
    parser.add_argument("N",type=int,
                        help="number of support points along one path (typically, N=100)")
    parser.add_argument("T",type=int,
                        help="length of Markov chain (typically, T=10000)")
    parser.add_argument("delta",type=float,
                        help="stepsize to determine new trial state x'")


    args                     = parser.parse_args()
    s_prob                   = args.s_prob
    N                        = args.N
    T                        = args.T
    delta                    = args.delta

    if (N < 11):
        parser.error("N should be larger than 10 at least")
    if (T < 1500):
        parser.error("T must be larger than burn-in length of 1500")
    if (delta <= 0.0):
        parser.error("delta must be larger than zero")

    fPOT,xa,xb,ta,tb,ncut  = init(s_prob)
    X,E                    = integrate_path(fPOT,xa,xb,ta,tb,delta,N,ncut,T)

    check(X,E,s_prob,fPOT)

#=============================================

main()

