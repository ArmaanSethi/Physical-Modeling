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
    epsilon = (tb-ta)/N

    #initialize path
    path = np.zeros(N) #initial path of all 0
    # path = np.random.rand(N) #initialize randomly
    path[0] = xa 
    path[-1] = xb #initialize endpoints
    path_list = [] #list of paths
    path_list.append(path)

    #initialize energy values
    energy = np.zeros(N)
    E_list = []
    E_list.append(energy)
    for e in range (N):
        energy[e]  = hamiltonian(path[e], path[e-1], epsilon, fPOT)

    i = 0
    while i < T:
        j = np.random.randint(1,N-1)

        prev_path = path_list[-1] 
        proposed_path = np.copy(prev_path)
        proposed_path[j] += (2*np.random.rand() - 1)*delta #perturbation

        E1 = hamiltonian(prev_path[j-1], proposed_path[j], epsilon, fPOT) #term 1 of delta_E
        E2 = hamiltonian(proposed_path[j], prev_path[j+1], epsilon, fPOT) #term 2 of delta_E
        E3 = -1*hamiltonian(prev_path[j-1], prev_path[j], epsilon, fPOT) #term 3 of delta_E
        E4 = -1*hamiltonian(prev_path[j], prev_path[j+1], epsilon, fPOT) #term 4 of delta_E

        delta_E = E1 + E2 + E3 + E4
        boltz = np.exp(-epsilon*delta_E/hbar)

        if (delta_E < 0 or (np.random.rand() <= boltz)):
            path_list.append(proposed_path)
            energy[j] += delta_E
            E_list.append(energy)
        else:
            path_list.append(prev_path)
            E_list.append(energy)

        i+=1

    X = np.array(path_list).T[:,ncut:]
    E = np.array(E_list).T[:,ncut:]
    print(np.shape(X), np.shape(path_list) )
    print(np.shape(E), np.shape(E_list))
    # to here ????

    return X,E
     
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
        plt.ylabel('$|\psi(x)|^2$',fontsize=ftsz)
        plt.legend(fontsize=ftsz)
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

