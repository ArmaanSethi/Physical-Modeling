#===============================================================
# Demonstrator for Monte-Carlo rejection sampling
#===============================================================
import argparse                  # allows us to deal with arguments to main()
from argparse import RawTextHelpFormatter
import numpy as np
import matplotlib.pyplot as plt
import p358utilities as util
#===============================================================
# Probabilty density functions and probability distributions.
# Name convention follows that of R: d**** is the density function, 
# r**** is the distribution. 
# dnorm will give you the normal distribution function (a Gaussian),
# and rnorm will give you R random variables, normally distributed.
#---------------------------------------------------------------

def dlogn(x,bounds):
    return np.exp(-(np.log(x))**2)

def dexpo(x,bounds):
    return bounds[1]*np.exp(x)/(np.exp(2.0)-np.exp(-4.0))

def rexpo(R,bounds):
    return (bounds[1]-bounds[0])*np.random.rand(R)+bounds[0]

def dunif(x,bounds):
    return bounds[1]*(np.zeros(x.size)+1.0) 

def runif(R,bounds):
    return (bounds[1]-bounds[0])*np.random.rand(R)+bounds[0]

def dcauc(x,bounds):
    return bounds[1]/(np.pi*(1.0+x*x))
    
def rcauc(R,bounds):
    return np.tan(np.pi*(np.random.rand(R)-0.5))

def dnorm(x,bounds):
    return bounds[1]*np.exp(-0.5*x*x)/np.sqrt(2.0*np.pi)

def rnorm(R,bounds):
    ind1   = np.arange(R/2)*2
    ind2   = np.arange(R/2)*2+1
    u1     = np.random.rand(R/2)
    u2     = np.random.rand(R/2)
    x      = np.zeros(R)
    x[ind1]= np.sqrt(-2.0*np.log(u1))*np.cos(2.0*np.pi*u2)
    x[ind2]= np.sqrt(-2.0*np.log(u1))*np.sin(2.0*np.pi*u2)
    return x

#===============================================================
# initialization

def init(s_target,s_proposal):

    if (s_target == 'exp'):      # f(x) = exp(2*x)
        fDTAR      = dexpo
        bounds_dst = np.array([-4.0,2.0]) 
    elif (s_target == 'normal'):  # N(0,1), i.e. Gaussian with mean=0,stddev=1
        fDTAR      = dnorm
        bounds_dst = np.array([-4.0,4.0])
    elif (s_target == 'lognormal'):
        fDTAR      = dlogn
        bounds_dst = np.array([0.0,10.0])
    else: 
        raise Exception("[init]: invalid s_target=%s\n" % (s_target))

    if (s_proposal == 'uniform'):
        fDPRO   = dunif
        fRPRO   = runif
    elif (s_proposal == 'cauchy'):
        fDPRO   = dcauc
        fRPRO   = rcauc
    else: 
        raise Exception("[init]: invalid s_proposal=%s\n" % (s_proposal))

    # This is rather crude: we search for the maximum value of fDTAR on the
    # bounds_dst specified, and make sure fDPRO on this interval is larger.
    # This would not work very well for multi-dimensional problems...
    x          = (bounds_dst[1]-bounds_dst[0])*np.arange(1000)/999.0+bounds_dst[0]
    R          = x.size
    Qx         = fDPRO(x,bounds_dst) 
    Px         = fDTAR(x,bounds_dst)
    maxloc     = np.argmax(Px/Qx) # if > 1, need adaption.
    cval       = Px[maxloc]/Qx[maxloc] # in case our sampling was not sufficient
    bounds_val = np.array([0.0,cval])
    print("[init]: cval = %13.5e" % (cval)) 

    return fDTAR,fDPRO,fRPRO,bounds_dst,bounds_val

#===============================================================
# function xr = reject(fDTAR,fDPRO,fRPRO,R)
# Returns an array of random variables sampled according to a
# target distribution fDTAR. A proposal distribution fDPRO can
# be provided.
#
# input: fDTAR     : function pointer to the target distribution density function.
#                    The function must take arguments fTAR(x,bounds),
#                    and must return the value of fTAR at x.
#        fDPRO     : function pointer to the proposal distribution density function.
#        fRPRO     : function pointer to the proposal distribution function. 
#                    Note that this will return a set of random numbers sampled
#                    according to fDPRO. Check dnorm and rnorm, for example.
#        R         : number of samples to be generated
#        bounds_dst: array of shape (2,N), where N is the number of
#                    dimensions (elements) in a single x, and the two
#                    fields per dimension give the lower and upper bound
#                    in that dimension.
# output: x_r      : random variables sampled according to P(x)
#--------------------------------------------------------------
def reject(fDTAR,fDPRO,fRPRO,R,bounds_dst,bounds_val):

    # ????? more here
    xr = []
    rejects = []
    cval = bounds_val[-1]
    while len(xr) < R:
        x = fRPRO(1, bounds_dst)
        u = runif(1, np.array([0,1]))

        p = fDTAR(x, bounds_dst)
        q = fDPRO(x, bounds_dst)
        # print("len, x, u, P, Q, cval")
        # print(len(xr)+len(rejects),x, u, p, q, cval)
        # input()
        if u < (p/(cval*q)):
            xr.append(x)
        else:
            rejects.append(x)

    print("ratio:", len(xr)/(len(xr)+len(rejects)))
    xr = np.array(xr)
    
    print("Mean:", np.mean(xr))
    print("Variance:",np.var(xr))

    return xr

#===============================================================
# function check(xr,fDTAR,fDPRO,bdst,bval)
# Calculates histogram of random variables xr and compares
# distribution to target and proposal distribution function.
# input: xr   : array of random variables
#        fTAR : function pointer to target density
#        fPRO : function pointer to proposal density
#        bdst : 2-element array: boundaries for distribution function
#        bval : 2-element array: second element contains constant c as in c*Q(x)
#---------------------------------------------------------------
def check(xr,fDTAR,fDPRO,bdst,bval):

    R = xr.size
    hist,edges = np.histogram(xr,np.int(np.sqrt(float(R))),range=(bdst[0],bdst[1]),normed=False)
    x          = 0.5*(edges[0:edges.size-1]+edges[1:edges.size])
    Px         = fDTAR(x,np.array([0.0,1.0]))
    Qx         = fDPRO(x,bval)
    # The histogram is in counts. Dividing by total counts gives area of 1.
    # Which is ok for initially normalized functions. 
    tothist    = np.sum(hist.astype(float))*(x[1]-x[0])
    hist       = hist.astype(float)/tothist 
    Ex         = np.sum(xr)/float(R)
    Varx       = np.sum((xr-Ex)**2)/float(R)
    print("[check]: E[P(x)] = %13.5e Var[P(x)] = %13.5e" % (Ex,Varx))

    ftsz = 10
    plt.figure(num=1,figsize=(8,8),dpi=100,facecolor='white')
    plt.subplot(111)
    plt.bar(x,hist,width=(x[1]-x[0]),facecolor='green',align='center')
    plt.plot(x,Px,linestyle='-',color='red',linewidth=1.0,label='P(x)')
    plt.plot(x,Qx,linestyle='-',color='black',linewidth=1.0,label='c Q(x)')
    plt.xlabel('x',fontsize=ftsz)
    plt.ylabel('pdf(x)',fontsize=ftsz)
    plt.legend()
    plt.tick_params(labelsize=ftsz)

    plt.show()

#===============================================================
def main():

    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument("s_target",type=str,
                        help="target distribution:\n"
                             "   exp      : exponential\n"
                             "   normal   : normal distribution\n"
                             "   lognormal: lognormal distribution")
    parser.add_argument("s_proposal",type=str,
                        help="proposal distribution:\n"
                             "   uniform : uniform distribution\n"
                             "   cauchy  : Cauchy distribution\n"
                             "   normal  : normal distribution")
    parser.add_argument("R",type=int,
                        help="number of realizations (i.e. draws)")

    args       = parser.parse_args()
    s_target   = args.s_target
    s_proposal = args.s_proposal
    R          = args.R

    fDTAR,fDPRO,fRPRO,bdst,bval = init(s_target,s_proposal) 
    xr                          = reject(fDTAR,fDPRO,fRPRO,R,bdst,bval)

    check(xr,fDTAR,fDPRO,bdst,bval)
  
#===============================================================
main()
