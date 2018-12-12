#======================================
# (Bayesian model comparison demonstrator - brute force)
# The program generates a "data set" from a polynomial whose 
# degree dorder can be set by the user. dorder = 2 gives a
# parabola. It then tries to fit the polynomial coefficients
# assuming polynomials of degree 1...forder. The fit results
# are assessed, and a measure for the most likely degree will
# be provided. 
#======================================
import argparse                  
from argparse import RawTextHelpFormatter
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import scipy.special
import p358utilities as util
#======================================
# generates polynomial data y at fixed x, with scatter sy. 
def make_data(n,order,theta,sigamp):
    x     = np.arange(n)/float(n-1)
    y     = np.zeros(n)+theta[0]
    for o in range(1,order+1):
        y = y*x+theta[o]
    avgy  = np.std(y)
    sy    = np.zeros(n)+sigamp*avgy
    print('[make_data]: sigma_y = %13.5e' % (sigamp*avgy))
    y     = y + np.random.randn(n)*sy
    return x,y,sy

#======================================
# logarithmic priors for polynomial coefficients (not normalized)
def pri_poly(theta,order):
    if (len(theta.shape) > 1):
        T                = max(theta.shape)
        pri              = np.zeros((order+2,T))
        ind              = np.where((0.0 <= theta[order+1,:]) & (theta[order+1,:] < 1.0))
        pri[order+1,:]   = -1e60
        pri[order+1,ind] = 0.0
        return np.sum(pri,axis=0)
    else:
        pri            = np.zeros(order+2) 
        if ((0 <= theta[order+1]) & (theta[order+1] <= 1.0)):
            pri[order+1] = 0.0
        else:
            pri[order+1] = -1e60
        return np.sum(pri)

#======================================
# model function (polynomial) for likelihood evaluation
def gen_poly(x,theta,order):
    if (isinstance(x,np.ndarray)):
        y = np.zeros(x.size)+theta[0]
    else:
        y = theta[0]
    for o in range(1,order+1):
        y = y*x+theta[o]
    return y

#======================================
# likelihood 
def likelihood(y,x,theta,fGEN,order):
    if (len(theta.shape) > 1):
        T    = max(theta.shape)
        obj  = np.zeros((x.size,T))
        for t in range(T):
            obj[:,t] = -0.5*(y-fGEN(x,theta[:,t],order))**2/theta[order+1,t]**2-np.log(np.sqrt(2.0*np.pi*theta[order+1,t]**2)) 
        return np.sum(obj,axis=0) 
    else:
        obj  = -0.5*(y-fGEN(x,theta,order))**2/theta[order+1]**2-np.log(np.sqrt(2.0*np.pi*theta[order+1]**2)) 
        return np.sum(obj)

#==============================================================
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
#         fPRI    : prior density function
# output: theta   : (npar,T-nburn) array containing the parameter distributions
#--------------------------------------------------------------
def methast(T,theta,dtheta,y,x,fGEN,fPRI,order):
    print('[methast]: order=%2i' % (order))
    ntheta      = theta.size
    thetat      = np.zeros((ntheta,T))
    thetat[:,0] = theta # starting point
    thetap      = np.zeros(ntheta)
    lalpha      = 0.0   # logarithm of likelihood ratios
    for t in range(1,T):
        thetap[:] = thetat[:,t-1]+dtheta*(2.0*np.random.rand(ntheta)-1.0)
        lalpha    = (  likelihood(y,x,thetap       ,fGEN,order) + fPRI(thetap       ,order)
                     - likelihood(y,x,thetat[:,t-1],fGEN,order) - fPRI(thetat[:,t-1],order))
        if (lalpha >= 0.0): # definitely accept (again, lalpha is logarithm of likelihood ratios)
            thetat[:,t] = thetap[:]
        else:
            if (np.log(np.random.rand(1)) < lalpha):
                thetat[:,t] = thetap[:]
            else:
                thetat[:,t] = thetat[:,t-1]
    return thetat

#============================================
# marginalizes over theta by summing over Markov Chain
# This is extremely crude to the point of being incorrect.
def marginalize_theta(theta,y,x,fGEN,fPRI,order):
    print('[marginalize_theta]: order=%2i' % (order))
    plik = likelihood(y,x,theta,fGEN,order)
    ppri = fPRI(theta,order)
    parr = np.exp(plik + ppri) # these are logarithms
    return util.decadesum(parr)/float(max(theta.shape))

#============================================
def main():

    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument("T",type=int,
                        help="length of Markov chain (depends on problem)")
    parser.add_argument("dorder",type=int,
                        help="data model order: integer > 0")
    parser.add_argument("forder",type=int,
                        help="polynomial fit order: integer > 0")
    parser.add_argument("sigamp",type=float,
                        help="uncertainty amplitude in units of rms(y)")

    args         = parser.parse_args()
    T            = args.T
    dorder       = args.dorder
    forder       = args.forder
    sigamp       = args.sigamp

    if (dorder == 0):
        raise Exception("[bayes_infer]: dorder = %i, but must be > 0\n" % (dorder))
    if (forder == 0):
        raise Exception("[bayes_infer]: forder = %i, but must be > 0\n" % (forder))

    dcoeff       = np.array([1.0,0.75,-1.5,1.0])
    ndat         = 30
    x,y,sy       = make_data(ndat,dorder,dcoeff,sigamp)

    npar         = forder+2 # a parabola is of forder=2, but has 3 coefficients. We also fit the uncertainties. 
    fGEN         = gen_poly
    fPRI         = pri_poly
    theta0       = np.zeros(npar)+1.0
    dtheta       = np.zeros((forder,npar))
    # note the separate dtheta for each parameter
    dtheta       = np.array([[0.1,0.1,0.0,0.0,0.0,0.0],      # constant and sigma
                             [0.1,0.1,0.1,0.0,0.0,0.0],      # linear and sigma
                             [0.3,0.3,0.2,0.1,0.0,0.0],      # quadratic and sigma
                             [1.0,1.5,1.0,0.1,0.1,0.0],      # cubic and sigma
                             [1.0,1.5,1.0,0.1,0.1,0.1]])     # quartic and sigma
    T0           = np.array(T*(np.arange(forder)+1.0)**2.5,dtype=int) # need to increase MC length, but can't get too crazy
    nburn        = np.array(0.1*np.array(T0,dtype=float),dtype=int) # burn-in time. Problem-dependent
    nbin         = 20
    pevidence    = np.zeros(forder)
    ix           = np.argsort(x)

    ftsz = 8
    fig1 = plt.figure(num=1,figsize=(10,2*forder),dpi=100,facecolor='white')
    fig2 = plt.figure(num=2,figsize=(10,2*forder),dpi=100,facecolor='white')

    for iford in range(0,forder): # runs from 0 to forder-1. 
        print('[bayes_infer]: iford=%1i T0=%7i nburn=%7i' % (iford+1,T0[iford],nburn[iford]))
        if (T0[iford] <= nburn[iford]):
            print('[bayes_infer]: T0=%7i must be larger than nburn=%7i' % (T0[iford],nburn[iford]))
            exit()    
        inpar              = iford+2
        theta              = (methast(T0[iford],theta0[0:inpar],dtheta[iford,0:inpar],y,x,fGEN,fPRI,iford))[:,nburn[iford]:T0[iford]]
        pevidence[iford]   = marginalize_theta(theta[0:inpar],y,x,fGEN,fPRI,iford)
        hist               = np.zeros((inpar,nbin,2))
        mhist              = np.zeros((inpar,2))
        print('[bayes_infer]: iford=%1i evidence = %13.5e' % (iford,pevidence[iford]))
        print('[bayes_infer]: plotting histograms for iford=%1i' % (iford)) 
        for p in range(inpar):
            hist[p,:,:]    = util.histogram(theta[p,:],nbin)
            mhist[p,0]     = np.sum(hist[p,:,0]*hist[p,:,1])/np.sum(hist[p,:,1])
            mhist[p,1]     = np.sqrt(np.sum(hist[p,:,1]*(hist[p,:,0]-mhist[p,0])**2)/np.sum(hist[p,:,1]))
            print('[bayes_infer]: iford=%1i   p=%1i theta=%13.5e+=%13.5e' % (iford,p,mhist[p,0],mhist[p,1]))    
   
            ax = fig1.add_subplot(forder,npar,p+(npar)*(iford)+2)
            #ax.bar(hist[p,:,0],hist[p,:,1],width=(hist[p,1,0]-hist[p,0,0]),facecolor='green',align='center')
            ax.plot(hist[p,:,0],hist[p,:,1],color='black',linewidth=1,linestyle='-')
            ax.set_xlabel('p%1i' % (p),fontsize=ftsz)
            util.rescaleplot(hist[p,:,0],hist[p,:,1],ax,0.05)
            ax.tick_params(labelsize=ftsz)

        print('[bayes_infer]: plotting data and fits for iford=%1i' % (iford))
        ax = fig1.add_subplot(forder,npar,(npar)*(iford)+1)
        ax.plot(x[ix],fGEN(x[ix],mhist[:,0],iford),linewidth=1,linestyle='-',color='red')
        ax.errorbar(x,y,yerr=sy,fmt='o',color='black',mfc='none',linewidth=1)
        ax.set_xlabel('x',fontsize=ftsz)
        ax.set_ylabel('y',fontsize=ftsz)
        util.rescaleplot(x,y,ax,0.05)
        ax.tick_params(labelsize=ftsz)

        print('[bayes_infer]: plotting Markov chains for iford=%1i' % (iford))
        for p in range(iford+2):
            ax = fig2.add_subplot(forder,forder+1,p+(forder+1)*(iford)+1)
            ax.scatter(theta[p,:],np.arange(max(theta.shape)),color='black',s=1)
            ax.set_xlabel('p%1i' % (p),fontsize=ftsz)
            ax.set_ylabel('t',fontsize=ftsz)
            util.rescaleplot(theta[p,:],np.arange(max(theta.shape)),ax,0.05)
            ax.tick_params(labelsize=ftsz)

    pevidence = pevidence/np.sum(pevidence)
    fig3 = plt.figure(num=3,figsize=(6,6),dpi=100,facecolor='white')
    ax   = fig3.add_subplot(111)
    ax.plot(np.arange(forder),pevidence,linewidth=1,linestyle='-',color='black')
    ax.set_xlabel('order',fontsize=ftsz)
    ax.set_ylabel('p(D|M)',fontsize=ftsz)
    util.rescaleplot(np.arange(forder)+1.0,pevidence,ax,0.05)
    ax.tick_params(labelsize=ftsz)
     
    plt.show()
        
#========================================

main()


