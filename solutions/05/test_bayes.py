#============================================
# program: test_bayes.py
# purpose: testing Bayes fitting following D. Hogg's primer
#============================================
import argparse                  # allows us to deal with arguments to main()
from argparse import RawTextHelpFormatter
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import scipy.special
import p358utilities as util

#============================================
# reads table of measured data. Expects
# three columns with (x, y, sig)
# Returns arrays x,y,sig.
def readdata(name):
    f   = open(name)
    lst = []
    for line in f:
        lst.append([float(d) for d in line.split()])
    x   = np.array([d[0] for d in lst])
    y   = np.array([d[1] for d in lst])
    sig = np.array([d[2] for d in lst])
    return x,y,sig

# ????? from here

#============================================
# glinfit: general least-squares fit
# returns the parameter vector a for a generalized
# least-squares fit y = f(x,a).
# input: 
#   x   : vector of length n: "independent" variable (i.e. the variable with the smaller errors)
#   y   : vector of length n: data points y(x). 
#   sy  : measurement uncertainties for y
#   fBAS: vector of basis functions (see HW05 for explanation). 
def glinfit(x,y,sy,fBAS):
    n    = x.size
    f    = fBAS(1.0) # this just returns the correct array size.
    npar = f.size
    A    = np.zeros((n,npar))
    invC = np.zeros((n,n))
    for i in range(n):
        A[i,:] = (fBAS(x[i]))[:]
    for i in range(n):
        invC[i,i] = 1.0/sy[i]**2        
    iCAt  = np.dot(np.transpose(A),invC)
    covar = np.dot(iCAt,A)
    theta = np.dot(np.linalg.inv(covar),np.dot(iCAt,y))
    covar = np.linalg.inv(covar)
    stheta= np.sqrt(np.diagonal(covar))
    chi2  = np.dot(np.dot(np.transpose(y-np.dot(A,theta)),invC),y-np.dot(A,theta))
    if (n > npar):
        q = scipy.special.gammainc(0.5*chi2,0.5*float(n-npar))
    return theta,stheta,chi2,q

#============================================
# generative model function block
# Should return the function value evaluated at x
# with parameters theta.
def gen_linear(x,theta):
    return theta[0]+theta[1]*x

def gen_quadratic(x,theta):
    return theta[0]+theta[1]*x+theta[2]*x**2

def gen_linsin(x,theta):
    return theta[0]+x*theta[1]+np.sin(x)*theta[2]

#============================================
# basis function block
# Should return a vector of functions.
def bas_linear(x):
    if (isinstance(x,np.ndarray)):
        return np.array([[np.zeros(x.size)+1.0],[x]])
    else:
        return np.array([1.0,x])

def bas_quadratic(x):
    if (isinstance(x,np.ndarray)):
        return np.array([[np.zeros(x.size)+1.0],[x],[x**2]])
    else:
        return np.array([1.0,x,x**2])

def bas_linsin(x):
    if (isinstance(x,np.ndarray)):
        return np.array([[np.zeros(x.size)+1.0],[x],[np.sin(x)]])
    else:
        return np.array([1.0,x,np.sin(x)])

#============================================
# random function block - called by methast to pick new guess for parameters
# Returns parameter vector with new guess for values.
# This is necessary, since some parameters need to be restricted.
def ran_linear(thetaold,delta,**kwargs):
    for key in kwargs:
        if (key == 'prune'):
            iprune = kwargs[key]
    npar          = thetaold.size
    ind           = np.random.randint(0,npar) # pick one parameter to be modified
    thetanew      = np.copy(thetaold)
    u             = np.random.rand(npar)
    if (iprune):
        thetanew[:] = thetaold[:] + delta[:]*(2.0*u[:]-1.0)
    else:
        thetanew[:] = thetaold[:] + delta[:]*(2.0*u-1.0)  
    return thetanew

#============================================
# prior function block
# Returns logarithm of complete joint prior necessary for inference.
def pri_linear(theta,**kwargs):
    for key in kwargs:
        if (key == 'prune'):
            iprune = kwargs[key]
    npar          = theta.size
    
    prior = 0.0
    if (iprune):
         # (0,1) priors for b and m are flat, i.e. 0.0 because of logarithm
         # (2) prior for Pb
         if (0.0 < theta[2] < 1.0):
             prior = prior + np.log(theta[2])
         else:
             prior = prior -1e60 # very negative number to make it unlikely
         # (3) prior for Yb is flat
         # (4) prior for Vb must be larger 0. 
         prior = prior + np.log(theta[4]) # will steer it away from theta[4]=0. 
         
    return prior

#============================================
# likelihood function
# note that this is done logarithmically, to prevent underflow.
def likelihood(y,x,sy,theta,fGEN,**kwargs):
    for key in kwargs:
        if (key == 'prune'):
            iprune = kwargs[key]
    sy2 = sy**2
    if (iprune == 1):
        npar  = theta.size-3 # there are 3 additional parameters for pruning 
        Pb    = theta[npar]
        Yb    = theta[npar+1]
        Vb    = theta[npar+2]
        objq1 = (1.0-Pb)*np.exp(-0.5*(y-fGEN(x,theta))**2/sy2)/np.sqrt(2.0*np.pi*sy2)
        objq  = Pb*np.exp(-0.5*(y-Yb)**2/(Vb+sy2))/np.sqrt(2.0*np.pi*(Vb+sy2))
        obj   = np.log(objq1+objq)
    else:
        obj  = -0.5*(y-fGEN(x,theta))**2/sy2-np.log(np.sqrt(2.0*np.pi*sy2)) # Hogg eq. 11 & 12
    return np.sum(obj)

#===============================================================
# function theta = methast(T,theta,dtheta,y,x,sy,fGEN)
# Returns the posterior distribution for the parameters theta,
# given the data y and the likelihood function determined by fGEN.
#
# input:  T       : length of Markov chain
#         theta   : initial guess for parameters (array of length npar)
#         dtheta  : sampling stepsize (array of length npar)
#         y       : data points
#         x       : sampling points
#         sy      : data uncertainties
#         fGEN    : model function
# output: theta   : (npar,T-nburn) array containing the parameter distributions
#--------------------------------------------------------------
def methast(T,theta,dtheta,y,x,sy,fGEN,fRAN,fPRI,**kwargs):
    ntheta      = theta.size
    thetat      = np.zeros((ntheta,T))
    thetat[:,0] = theta # starting point
    thetap      = np.zeros(ntheta)
    lalpha      = 0.0   # logarithm of likelihood ratios
    npar        = theta.size
    for t in range(1,T):
        # this returns the guesses (x' in the old nomenclature). Note that these are all 
        thetap[:] = fRAN(thetat[:,t-1],dtheta,**kwargs)
        # note that the function likelihood returns the logarithm of the likelihood, and fPRI does the same
        # for the prior probability. The prior assures that the parameters are correctly specified, e.g.
        # it assigns zero probability (very negative number in log) for an out-of-range parameter.
        lalpha    = (  likelihood(y,x,sy,thetap       ,fGEN,**kwargs) + fPRI(thetap       ,**kwargs) 
                     - likelihood(y,x,sy,thetat[:,t-1],fGEN,**kwargs) - fPRI(thetat[:,t-1],**kwargs))
        if (lalpha >= 0.0): # definitely accept (again, lalpha is logarithm of likelihood ratios)
            thetat[:,t] = thetap[:]
        else:
            if (np.log(np.random.rand(1)) < lalpha):
                thetat[:,t] = thetap[:]
            else:
                thetat[:,t] = thetat[:,t-1]
    return thetat

#============================================
# finds the "best" parameters by taking maxima. 
# Part of marginalization in sampling approach.
def find_nbest(x,y,n):
    best = np.zeros(n)
    y0   = np.copy(y)
    ind  = -1
    for i in range(n):
        ind     = np.argmax(y0)
        best[i] = x[ind]
        y0[ind] = np.min(y0)
    return best

# ?????? to here

#============================================
def main():

    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument("Pbad",type=float,
                        help="probability for bad data points:\n"
                             "   0<=Pbad<1. First run with Pbad=0\n"
                             "   (no pruning of bad data points)")
    parser.add_argument("T",type=int,
                        help="length of Markov chain (depends on problem)")

    args       = parser.parse_args()
    Pb0        = args.Pbad
    T          = args.T

    x,y,sy       = readdata('hogg.txt')

# ???? from here

    mask         = np.zeros(x.size)+1 # if set to 1, point is being used for fit
    # for Hogg's data set, try setting first four points to 0 for "pruning by eye"
    # mask[0:4]    = 0
    x            = x[np.nonzero(mask)]
    y            = y[np.nonzero(mask)]
    sy           = sy[np.nonzero(mask)]

    imap         = [1,0] # indicates which parameters should be used for 2d histogram. 
    # set the parameters for the linear model
    fGEN         = gen_linear
    fRAN         = ran_linear
    fPRI         = pri_linear
    fBAS         = bas_linear
    if (Pb0 == 0.0): 
        theta0       = np.array([200.0,1.0]) # intercept b and slope m
        dtheta       = np.array([1.0,0.1])
    else:
        theta0       = np.array([30.0,2.0,Pb0,np.mean(y),(np.std(y))**2])
        dtheta       = np.array([10.0,1.0,0.03,0.1*np.mean(y),0.2*(np.std(y))**2]) # sample Yb at range y

    nburn        = 1000 # burn-in time
    nbest        = 10 # plot sample solutions
    npar         = theta0.size
    nbin2        = int(float(T)**0.4) # number of histogram bins
    nbin         = 100#int(np.sqrt(float(T)))

    # get the parameter distribution theta
    thetafull    = methast(T,theta0,dtheta,y,x,sy,fGEN,fRAN,fPRI,prune=(Pb0>0.0))
    theta        = thetafull[:,nburn:T]
    # get the least-squares fit
    thetalsq,sthetalsq,chi2lsq,qlsq = glinfit(x,y,sy,fBAS)

    # analysis of results, margialized over pruning parameters when appicable    
    thetasamp    = np.zeros((npar,nbest))  # sampling for display
    hist         = np.zeros((npar,nbin,2)) # number of parameters, number of bins, (x,y)
    hist2d,x1,x2 = util.histogram2d(theta[imap[0],:],theta[imap[1],:],nbin2,nbin2) # (m,b)

    # show results from Bayesian fitting
    mhist = np.zeros((len(imap),2)) # mean and std of histograms
    print("[test_bayes]: Bayesian results:")
    for i in range(len(imap)):
        hist[i,:,:]    = util.histogram(theta[i,:],nbin)
        mhist[i,0]     = np.sum(hist[i,:,0]*hist[i,:,1])/np.sum(hist[i,:,1])
        mhist[i,1]     = np.sqrt(np.sum(hist[i,:,1]*(hist[i,:,0]-mhist[i,0])**2)/np.sum(hist[i,:,1]))
        thetasamp[i,0] = find_nbest(hist[i,:,0],hist[i,:,1],1)
        for j in range(1,nbest):
            k              = np.random.randint(max(theta.shape))
            thetasamp[i,j] = theta[i,k]     
        print("[test_bayes]:   i=%1i thetaMAP[%1i] = %13.5e, <theta>=%13.5e+=%13.5e" % (i,i,thetasamp[i,0],mhist[i,0],mhist[i,1]))
    # for pruning, also calculate histogram of Pb
    
    # show results from frequentist fitting
    print("[test_bayes]: frequentist results:")
    for i in range(thetalsq.size):
        print("[test_bayes]:   i=%1i thetalsq[%1i] = %13.5e+-%13.5e" % (i,i,thetalsq[i],sthetalsq[i]))
    print("[test_bayes]:   chi2,q for least-squares fit: %13.5e %13.5e" % (chi2lsq,qlsq))

    minx2        = np.min(x2)
    maxx2        = np.max(x2)
    minx1        = np.min(x1)
    maxx1        = np.max(x1)

    xcont        = (maxx2-minx2)*np.arange(nbin2)/float(nbin2-1)+minx2
    ycont        = (maxx1-minx1)*np.arange(nbin2)/float(nbin2-1)+minx1
    xcont,ycont  = np.meshgrid(xcont,ycont)

    ftsz = 10
    plt.figure(num=1,figsize=(10,6),dpi=100,facecolor='white')
    plt.subplot(221)
    plt.imshow(hist2d,extent=(minx2,maxx2,minx1,maxx1),interpolation='bicubic',cmap=cm.gist_gray_r,aspect='auto',origin='lower')
    plt.contour(xcont,ycont,hist2d,6,colors='k')
    plt.xlabel('b',fontsize=ftsz)
    plt.ylabel('m',fontsize=ftsz)
    plt.tick_params(labelsize=ftsz)

    plt.subplot(223)
    plt.bar(hist[1,:,0],hist[1,:,1],width=(hist[1,1,0]-hist[1,0,0]),facecolor='green',align='center')
    plt.plot(hist[1,:,0],mlab.normpdf(hist[1,:,0],mhist[1,0],mhist[1,1])*(hist[1,1,0]-hist[1,0,0]),color='red',linestyle='-',linewidth=1)
    plt.title('<m>=%4.2f+-%4.2f' % (mhist[1,0],mhist[1,1]),x=0.2,y=0.8,fontsize=ftsz)
    plt.xlabel('m',fontsize=ftsz)
    plt.ylabel('#',fontsize=ftsz)
    plt.tick_params(labelsize=ftsz)

    plt.subplot(224)
    plt.bar(hist[0,:,0],hist[0,:,1],width=(hist[0,1,0]-hist[0,0,0]),facecolor='green',align='center')
    plt.plot(hist[0,:,0],mlab.normpdf(hist[0,:,0],mhist[0,0],mhist[0,1])*(hist[0,1,0]-hist[0,0,0]),color='red',linestyle='-',linewidth=1)
    plt.title('<b>=%3i+-%2i' % (mhist[0,0],mhist[0,1]),x=0.2,y=0.8,fontsize=ftsz)
    plt.xlabel('b',fontsize=ftsz)
    plt.ylabel('#',fontsize=ftsz)
    plt.tick_params(labelsize=ftsz)

    ix = np.argsort(x)

    plt.subplot(222)
    for i in range(1,nbest):
        plt.plot(x[ix],fGEN(x[ix],thetasamp[:,i]),linewidth=1,linestyle='-',color='lightgray')
    plt.plot(x[ix],fGEN(x[ix],thetasamp[:,0]),linewidth=1,linestyle='-',color='black')
    plt.plot(x[ix],fGEN(x[ix],thetalsq),linewidth=1,linestyle='-',color='red')
    plt.errorbar(x[ix],y[ix],yerr=sy[ix],fmt='o',linewidth=1)
    plt.title('m=%4.2f+-%4.2f, b=%3i+-%2i, $\chi^2$=%10.2e q=%10.2e' % (thetalsq[1],sthetalsq[1],thetalsq[0],sthetalsq[0],chi2lsq,qlsq),
             fontsize=ftsz,color='red')
    plt.xlabel('x',fontsize=ftsz)
    plt.ylabel('y',fontsize=ftsz)
    plt.tick_params(labelsize=ftsz)

    plt.savefig('figure_1.eps',format='eps',dpi=1000)

    plt.figure(num=2,figsize=(3*npar,5),dpi=100,facecolor='white')
    for i in range(npar):
        plt.subplot(1,npar,i+1)
        plt.plot(thetafull[i,:],np.arange(max(thetafull.shape)),linestyle='-',linewidth=1,color='black')
        plt.xlabel('theta%1i' % (i),fontsize=ftsz)
        plt.ylabel('t',fontsize=ftsz)
        plt.tick_params(labelsize=ftsz)

    plt.savefig('figure_2.eps',format='eps',dpi=1000)

    plt.show()

# ????? to here

#========================================

main()

