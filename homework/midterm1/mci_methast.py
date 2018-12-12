#===============================================================
# Demonstrator for Monte-Carlo rejection sampling
#===============================================================
import argparse                  # allows us to deal with arguments to main()
from argparse import RawTextHelpFormatter
import numpy as np
import matplotlib.pyplot as plt
import p358utilities as util
#===============================================================
# distribution and density functions. For the MH algorithm
# (unlike for rejection and importance sampling!), only
# the probability density **functions** are needed, except for
# the *local* proposal distribution - in this case a Gaussian.
# Function names follow the R-convention, with r???? standing for
# the probability distribution, and d???? for the probability 
# density function.
#===============================================================
# x_r = rnorm(R)
# returns normal random deviates following the Box-Mueller method.
#---------------------------------------------------------------
def rnorm(R):
    # from here ????
    def gaussian(u0,u1):
        z0 = np.sqrt(-2*np.log(u0))*np.cos(2*np.pi*u1)
        z1 = np.sqrt(-2*np.log(u0))*np.sin(2*np.pi*u1)
        return z0,z1

    u0 = np.random.rand(R)
    u1 = np.random.rand(R)

    z0,z1 = gaussian(u0,u1)

    mu = 0
    sigma = 1
    # to here ????
    return z1*sigma + mu

#==============================================================
# probability density functions
# These take two arguments: an independent variable x, and 
# a 2-element array bounds, with the lower and upper bound.
# The latter is used to normalize density functions that are
# defined on finite intervals [a,b]. For example, for the 
# uniform distribution, dunif(x) = (1/(b-a)) if a<=x<=b, 0 otherwise,
# therefore, dunif(x) = 1/(bounds[1]-bounds[0]) in this implementation.
# The variable x can be a single value, or an array of values.
#--------------------------------------------------------------
def dexpo(x,bounds):
    if (np.isscalar(x)):
        if ((x >= bounds[0]) and (x <= bounds[1])):
            px = np.exp(x)/(np.exp(bounds[1])-np.exp(bounds[0]))
        else:
            px = 0.0
    else:
        px = np.zeros(len(x))
        for i in range(len(x)):
            if ((x[i] >= bounds[0]) and (x[i] <= bounds[1])):
                px[i] = np.exp(x[i])/(np.exp(bounds[1])-np.exp(bounds[0]))
            else:
                px[i] = 0.0
    return px

def dunif(x,bounds):
    if ((x >= bounds[0]) and (x <= bounds[1])):
        px = 1.0/(bounds[1]-bounds[0])
    else:
        px = 0.0
    return px

def dcauch(x,bounds):
    return 1.0/(np.pi*(1.0+x*x))

def dnorm(x,bounds):
    return np.exp(-0.5*x*x)/np.sqrt(2.0*np.pi)

def dcrazy(x,bounds):
    return np.exp(0.4*(x-0.4)**2-0.08*x**4)/7.85218

def dlognorm(x,bounds):
    return np.exp(-(np.log(x))**2)/np.sqrt(np.sqrt(np.exp(1))*np.pi)

#===============================================================

def init(s_target):

    if (s_target == 'exp'):      # f(x) = exp(2*x)
        fTAR       = dexpo
        bounds_dst = np.array([-4.0,2.0]) 
        x0         = 0.0
    elif (s_target == 'normal'):  # N(0,1), i.e. Gaussian with mean=0,stddev=1
        fTAR       = dnorm
        bounds_dst = np.array([-4.0,4.0])
        x0         = 0.0
    elif (s_target == 'crazy'):
        fTAR       = dcrazy
        bounds_dst = np.array([-4.0,4.0])
        x0         = 0.0
    elif (s_target == 'lognormal'):
        fTAR       = dlognorm
        bounds_dst = np.array([0.0,10.0])
        x0         = 0.0
    else: 
        raise Exception("[init]: invalid s_target=%s\n" % (s_target))

    return fTAR,bounds_dst,x0

#===============================================================
# function xr = methast(fTAR,bounds,R,x0,delta)
# Returns an array of random variables sampled according to a
# target distribution fTAR. A proposal distribution fPRO can
# be provided.
#
# input: fTAR      : function pointer to the target distribution.
#                    The function must take arguments fTAR(x,bounds),
#                    and must return the value of fTAR at x.
#        bounds    : lower and upper bounds for sampling x.
#                    Note that not all functions need this.
#        R         : number of samples to be generated
#        x0        : initial guess
#        delta     : step size, or scale (width of local proposal distribution Q)
# output: x_r      : random variables drawn from fTAR.
#--------------------------------------------------------------
def methast(fTAR,bounds,R,x0,delta):
    # from here ????
    t = 0
    xr = []
    xr.append(x0)
    while len(xr) < R:
        x_cand = rnorm(1)*delta + xr[t]
        a = fTAR(x_cand, bounds)/fTAR(xr[t], bounds) #acceptance ratio
        u = np.random.rand() #u from normal dist
        if u < a or a >= 1.0:
            xr.append(x_cand)
        else:  
            xr.append(xr[t])
        t+=1
    xr = np.array(xr)
    # to here ????
    return xr

#===============================================================
# function check(xr,fTAR)
# Calculates histogram of random variables xr and compares
# distribution to target and proposal distribution function.
# input: xr   : array of random variables
#        fTAR : function pointer to target distribution
#---------------------------------------------------------------
def check(xr,fTAR):

    R = xr.size
    hist,edges = np.histogram(xr,np.int(np.sqrt(float(R))),normed=False)
    x          = 0.5*(edges[0:edges.size-1]+edges[1:edges.size])
    nbin       = x.size
    Px         = fTAR(x,np.array([x[0],x[nbin-1]]))
    # The histogram is in counts. Dividing by total counts gives area of 1.
    # Which is ok for initially normalized functions. 
    tothist    = np.sum(hist.astype(float))*(x[1]-x[0])
    hist       = hist.astype(float)/tothist 

    ftsz = 10
    plt.figure(num=1,figsize=(8,6),dpi=100,facecolor='white')
    plt.subplot(121)
    plt.bar(x,hist,width=(x[1]-x[0]),facecolor='green',align='center')
    plt.plot(x,Px,linestyle='-',color='red',linewidth=1.0,label='P(x)')
    plt.xlabel('x',fontsize=ftsz)
    plt.ylabel('pdf(x)',fontsize=ftsz)
    plt.legend()
    plt.tick_params(labelsize=ftsz)

    it = np.arange(R)
    plt.subplot(122)
    plt.plot(xr,it,linestyle='-',color='black',linewidth=1.0)
    plt.xlabel('x$_r$')
    plt.ylabel('t')
    plt.tick_params(labelsize=ftsz)

    plt.show()

#===============================================================
def main():

    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument("s_target",type=str,
                        help="target distribution:\n"
                             "   exp      : exponential\n"
                             "   normal   : normal distribution\n"
                             "   crazy    : crazy distribution\n"
                             "   lognormal: lognormal distribution")
    parser.add_argument("R",type=int,
                        help="number of realizations (i.e. draws)")
    parser.add_argument("delta",type=float,
                        help="scaling for step size, i.e. width of Q(x)")

    args              = parser.parse_args()
    s_target          = args.s_target
    delta             = args.delta
    R                 = args.R
    if (delta <= 0.0):
        parser.error("delta must be positive")

    fTAR,bounds,x0    = init(s_target) 
    xr                = methast(fTAR,bounds,R,x0,delta)

    check(xr,fTAR)
  
#===============================================================
main()
