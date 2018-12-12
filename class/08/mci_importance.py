# Demonstrator for importance sampling.
#===========================================
import argparse                  # allows us to deal with arguments to main()
from argparse import RawTextHelpFormatter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import p358utilities as util
#===========================================
# The function names follow the R convention:
# dXXXX indicates the probability density function (i.e. the value of a function)
# rXXXX indicates the probability distribution (i.e. random variables distributed according to rXXXX)
#===========================================
# dnorm/rnorm: Normal distribution N(mu,sigma)
def dnorm(x,**kwargs):
    mu0 = 0.0
    sg0 = 1.0
    imu = 0
    isg = 0
    for key in kwargs:
        if (key=='mu'):
            mu  = kwargs[key]
            imu = 1
        if (key=='sg'):
            sg  = kwargs[key]
            isg = 1
    if (imu):
        mu0 = mu
    if (isg):
        sg0 = sg
    return np.exp(-0.5*((x-mu0)/sg0)**2)/(np.sqrt(2.0*np.pi)*sg0)

def rnorm(N,**kwargs): # Box-Mueller method
    mu0 = 0.0
    sg0 = 1.0
    imu = 0
    isg = 0
    for key in kwargs:
        if (key=='mu'):
            mu  = kwargs[key]
            imu = 1
        if (key=='sg'):
            sg  = kwargs[key]
            isg = 1
    if (imu):
        mu0 = mu
    if (isg):
        sg0 = sg
    if (N > 1):
         ind1   = np.arange(int(N/2))*2
         ind2   = np.arange(int(N/2))*2+1
         u1     = np.random.rand(int(N/2))
         u2     = np.random.rand(int(N/2))
         x      = np.zeros(N)
         x[ind1]= np.sqrt(-2.0*np.log(u1))*np.cos(2.0*np.pi*u2)
         x[ind2]= np.sqrt(-2.0*np.log(u1))*np.sin(2.0*np.pi*u2)
    else:
         u1     = np.random.rand(1)
         u2     = np.random.rand(1)
         x      = np.sqrt(-2.0*np.log(u1))*np.cos(2.0*np.pi*u2)
    return sg0*x+mu0

#===========================================
# dunif/runif: Uniform distribution U(a,b)
def dunif(x,**kwargs):
    a0 = 0.0
    b0 = 1.0
    ia = 0
    ib = 0
    for key in kwargs:
        if (key=='a'):
            a  = kwargs[key]
            ia = 1
        if (key=='b'):
            b  = kwargs[key]
            ib = 1
    if (ia):
       a0 = a
    if (ib):
       b0 = b
    if (len(x)>1):
        return np.zeros(x.size)+1.0/(b-a)
    else:
        return 1.0/(b-a)

def runif(N,**kwargs):
    a0 = 0.0
    b0 = 1.0
    ia = 0
    ib = 0
    for key in kwargs:
        if (key=='a'):
            a  = kwargs[key]
            ia = 1
        if (key=='b'):
            b  = kwargs[key]
            ib = 1
    if (ia):
       a0 = a
    if (ib):
       b0 = b
    return a0+(b0-a0)*np.random.rand(N)

#===========================================
# dcosh/rcosh: hyperbolic cosine distribution
def dcosh(x,**kwargs):
    return 0.5*(np.exp(x)+np.exp(-x))

def rcosh(N,**kwargs):
    u = np.random.rand(N)
    return np.log(np.sqrt(1.0+u**2)+u)

#===========================================
# Lorentzian/Cauchy
def dcauc(x,**kwargs):
    return 1.0/(np.pi*(1.0+x**2))

def rcauc(N,**kwargs):
    return np.tan(np.pi*(np.random.rand(N)-0.5))

#===========================================
# Example (integrand) functions
#-------------------------------------------
def sinxcosx(x):
    return np.sin(x*np.cos(x))

#===========================================
def expabs(x,**kwargs):
    p0 = 0.0
    ip = 0
    for key in kwargs:
        if (key=='p'):
            p  = kwargs[key]
            ip = 1
    if (ip):
       p0 = p
    return 0.5*np.exp(-np.abs(x))*x**p0

#===========================================
def gaussian(x,**kwargs):
    p0 = 0.0
    ip = 0
    for key in kwargs:
        if (key=='p'):
            p  = kwargs[key]
            ip = 1
    if (ip):
       p0 = p
    return np.exp(-0.5*x**2)*x**p0/np.sqrt(2.0*np.pi)

#===========================================
def lognormal(x,**kwargs):
    return np.exp(-np.log(x)**2)

#===========================================
# function importsamp(fINT,fRAN,fDEN,N,**kwargs)
# Calculates one realization of the mean of fINT, sampled according to fRAN.
# input: 
#       fINT  : integrand function
#       fRAN  : random variable distribution function
#       fDEN  : probability density function corresponding to fRAN
#       kwargs: a,b: integration interval. If not given, default values
#               for each distribution are assumed (e.g. [0,1] for U, infinite range for N).
def importsamp(fINT,fRAN,fDEN,N,**kwargs):
    X    = fRAN(N,**kwargs) # transform into desired deviate
    I    = fINT(X,**kwargs)/fDEN(X,**kwargs)
    est  = np.mean(I)
    std  = np.std(I)
    return est,std

#===========================================

def init(sinteg,ssample):
    if (sinteg == 'expabs'):
        fINT = expabs
        a    = -5.0
        b    = 5.0
        mu   = 0.0
        sg   = 1.0 
    elif (sinteg == 'gaussian'):
        fINT = gaussian
        a    = -10.0
        b    = 10.0
        mu   = 0.0
        sg   = 1.0
    elif (sinteg == 'lognormal'):
        fINT = lognormal
        a    = 0.0
        b    = 10.0
        mu   = 0.0
        sg   = 1.0
    else:
        raise Exception('[init]: invalid integrand function %s' % (sinteg))

    if (ssample == 'uniform'):
        fDEN = dunif
        fRAN = runif
    elif (ssample == 'normal'):
        fDEN = dnorm
        fRAN = rnorm
    elif (ssample == 'cauchy'):
        fDEN = dcauc
        fRAN = rcauc
    else:
        raise Exception('[init]: invalid sampling function %s' % (ssample))
    
    return fINT,fRAN,fDEN,a,b,mu,sg

#===========================================

def main():

    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument("sinteg",type=str,
                        help="function to be integrated:\n"
                             "   expabs   : double exponential\n"
                             "   gaussian : normal distribution\n"
                             "   lognormal: lognormal distribution")
    parser.add_argument("ipower",type=int,
                        help="p-th order moment [p=0,1,2,...]")
    parser.add_argument("ssample",type=str,
                        help="sampling function:\n"
                             "   uniform : uniform distribution\n"
                             "   cauchy  : Cauchy distribution\n"
                             "   normal  : normal distribution")
    parser.add_argument("N",type=int,
                        help="number of support points in single integral")
    parser.add_argument("R",type=int,
                        help="number of realizations (=experiments). R >0")
    args       = parser.parse_args()
    sinteg     = args.sinteg
    p          = args.ipower
    ssample    = args.ssample
    N          = args.N
    R          = args.R
    if (R <= 0):
        parser.error("R must be larger than zero")

    fINT,fRAN,fDEN,a,b,mu,sg = init(sinteg,ssample)

    if (R == 1):
        ev,sd = importsamp(fINT,fRAN,fDEN,N,a=a,b=b,mu=mu,sg=sg,p=p)
        x    = a+(b-a)*np.arange(200)/199.0
        fint = fINT(x,a=a,b=b,mu=mu,sg=sg,p=p)
        fden = fDEN(x,a=a,b=b,mu=mu,sg=sg,p=p)
        minf = np.min(np.array([np.min(fint),np.min(fden)]))
        maxf = np.max(np.array([np.max(fint),np.max(fden)]))
        print("[mci_importance]: I = %13.5e += %13.5e" % (ev,sd))
        ftsz = 10
        plt.figure(num=1,figsize=(8,8),dpi=100,facecolor='white')
        plt.subplot(111)
        plt.plot(x,fint,linewidth=1,linestyle='-',color='black',label='f(x)')
        plt.plot(x,fden,linewidth=1,linestyle='-',color='red',label='p(x)')
        plt.xlabel('x',fontsize=ftsz)
        plt.ylabel('f(x),p(x)',fontsize=ftsz)
        plt.title('$\mu=$%11.3e $\sigma=$%11.3e' % (ev,sd))
        plt.legend()
        util.rescaleplot(x,np.array([minf,maxf]),plt,0.05)
        plt.tick_params(labelsize=ftsz)

        plt.show()
    else:
        ev   = np.zeros(R)
        sd   = np.zeros(R) 
        Ev   = np.zeros(R)
        Sd   = np.zeros(R)
        for i in range(R):
            for j in range(i):
                ev[j],sd[j] = importsamp(fINT,fRAN,fDEN,N,a=a,b=b,mu=mu,sg=sg,p=p)
            Ev[i]       = np.mean(ev[0:i+1]) 
            Sd[i]       = np.std(ev[0:i+1])
        ftsz = 10
        plt.figure(num=1,figsize=(6,8),dpi=100,facecolor='white')
        plt.subplot(311)
        plt.plot(np.arange(R-2)+1,Ev[2:R],linestyle='-',linewidth=1,color='black')
        plt.xlabel('R',fontsize=ftsz)
        plt.ylabel('Ev',fontsize=ftsz)
        plt.tick_params(labelsize=ftsz)
        plt.subplot(312)
        plt.plot(np.log10(np.arange(R-2)+1),np.log10(Sd[2:R]),linestyle='-',linewidth=1,color='black')
        plt.xlabel('log R',fontsize=ftsz)
        plt.ylabel('log Var',fontsize=ftsz)
        plt.tick_params(labelsize=ftsz)
        plt.subplot(313)
        hist,edges = np.histogram(ev[0:R-1],np.int(np.sqrt(R)),normed=False)
        x          = 0.5*(edges[0:edges.size-1]+edges[1:edges.size])
        tothist    = np.sum(hist.astype(float))
        hist       = hist.astype(float)/tothist # it seems the "normed" keyword does not work in numpy/mathplotlib
        normd      = mlab.normpdf(x,Ev[R-1],Sd[R-1])*(x[1]-x[0])
        peak       = mlab.normpdf(Ev[R-1],Ev[R-1],Sd[R-1])*(x[1]-x[0]) # need this for FWHM
        print('[mci_importance]: sum(hist)   = %13.5e %13.5e' % (np.sum(hist),tothist))
        print('[mci_importance]: expectation = %13.5e +-%12.5e' % (Ev[R-1],Sd[R-1]))
        plt.bar(x,hist,width=(x[1]-x[0]),facecolor='green',align='center')
        plt.plot(np.array([1.0,1.0])*Ev[R-1],np.array([0.0,1.0]),linestyle='--',color='black',linewidth=1.0)
        plt.plot(np.array([Ev[R-1]-0.5*2.36*Sd[R-1],Ev[R-1]+0.5*2.36*Sd[R-1]]),0.5*np.array([peak,peak]),
                 linestyle='--',color='black',linewidth=1.0)
        plt.plot(x,normd,linestyle='--',color='red',linewidth=2.0)
        plt.xlabel('E[I]',fontsize=ftsz)
        plt.ylabel('frequency',fontsize=ftsz)
        plt.title('$\mu=$%11.3e $\sigma=$%11.3e' % (Ev[R-1],Sd[R-1]))
        plt.ylim(np.array([0.0,1.05*np.max(hist)]))
        plt.tick_params(labelsize=ftsz)

        plt.show()
#
#===========================================

main()
