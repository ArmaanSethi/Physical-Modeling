#===================================
# Demonstrator for Law of Large Numbers (LLN)
# and Central Limit Theorem (CLT).
#===================================
import argparse                  # allows us to deal with arguments to main()
from argparse import RawTextHelpFormatter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import p358utilities as util
#===================================

def main():

    # argument parsing block
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument("itest",type=int,
                        help="itest:\n"
                             "   1        : x=U(0,1) for increasing sample length\n"
                             "   2        : x=U(0,1) for fixed sample length, but N repetitions")
    parser.add_argument("N",type=int,
                        help="N: number of draws to construct one random variable X")
    parser.add_argument("-R","--Realizations",type=int,
                        help="R: number of realizations")

    args       = parser.parse_args()
    itest      = args.itest
    if ((itest == 2) and not args.Realizations):
        parser.error("itest == 2 requires argument --Realizations R")
    N          = args.N
    if (args.Realizations):
        R      = args.Realizations

#------------------------------------

    if (itest == 1):
        # experiment 1:  
        x    = np.zeros(N)            # random variable 
        Ex   = np.zeros(N)            # expectation value of x
        Vx   = np.zeros(N)            # variance of x
        x[0] = np.random.rand(1)      # need at least 2 elements for mean,stdv
        for i in range(1,N):          #
            x[i]  = np.random.rand(1) 
            Ex[i] = np.mean(x[0:i+1])
            Vx[i] = np.mean((x[0:i+1]-Ex[i])**2)

        ftsz = 10
        plt.figure(num=1,figsize=(8,8),dpi=100,facecolor='white')
        plt.subplot(211)
        plt.plot(np.arange(N-1)+1,Ex[1:N],linestyle='-',linewidth=1,color='black')
        plt.xlabel('N',fontsize=ftsz)
        plt.ylabel('E[x]',fontsize=ftsz)
        plt.tick_params(labelsize=ftsz)
        plt.subplot(212)
        plt.plot(np.log10(np.arange(N-1)+1),np.log10(Vx[1:N]),linestyle='-',linewidth=1,color='black')
        plt.xlabel('log N',fontsize=ftsz)
        plt.ylabel('log Var[x]',fontsize=ftsz)
        plt.tick_params(labelsize=ftsz)
        plt.show()

    elif(itest == 2):
        # experiment 2:
        X    = np.zeros(R) # a random variable
        V    = np.zeros(R) # and its variance.
        EX   = np.zeros(R) # The expectation value for X, 
        VX   = np.zeros(R) # and its variance.
        for n in range(2,R): # note loop over realizations (number of experiments)
            for j in range(n): # build n random variables X with variance V.
                x    = np.random.rand(N)  
                X[j] = np.mean(x)
                V[j] = np.mean((x-X[j])**2)
            EX[n] = np.mean(X[0:n+1])            # calculate sample mean of X
            VX[n] = np.mean((X[0:n+1]-EX[n])**2) # and its Variance
        ftsz = 10
        plt.figure(num=1,figsize=(8,8),dpi=100,facecolor='white')
        plt.subplot(311)
        plt.plot(np.arange(R-2)+1,EX[2:R],linestyle='-',linewidth=1,color='black')
        plt.xlabel('R',fontsize=ftsz)
        plt.ylabel('E[X]',fontsize=ftsz)
        plt.tick_params(labelsize=ftsz)
        plt.subplot(312)
        plt.plot(np.log10(np.arange(R-2)+1),np.log10(VX[2:R]),linestyle='-',linewidth=1,color='black')
        plt.xlabel('log R',fontsize=ftsz)
        plt.ylabel('log Var[X]',fontsize=ftsz)
        plt.tick_params(labelsize=ftsz)
        plt.subplot(313)
        hist,edges = np.histogram(X[0:R-1],int(np.sqrt(R)),normed=False)
        xh         = 0.5*(edges[0:edges.size-1]+edges[1:edges.size])
        avg1,std1  = util.histmean(xh,hist,normed=True)
        tothist    = np.sum(hist.astype(float))
        hist       = hist.astype(float)/tothist # it seems the "normed" keyword does not work in numpy/mathplotlib
        normd      = mlab.normpdf(xh,avg1,std1)*(xh[1]-xh[0]) # include normalization factor accounting for area!
        peak       = mlab.normpdf(avg1,avg1,std1)*(xh[1]-xh[0]) # need this for FWHM
        print('[mci_llnclt]: sum(hist)   = %13.5e %13.5e' % (np.sum(hist),tothist))
        print('[mci_llnclt]: expectation = %13.5e +-%12.5e' % (avg1,std1))
        plt.bar(xh,hist,width=(xh[1]-xh[0]),facecolor='green',align='center')
        plt.plot(np.array([avg1,avg1]),np.array([0.0,1.0]),linestyle='--',color='black',linewidth=1.0)
        plt.plot(np.array([avg1-0.5*2.36*std1,avg1+0.5*2.36*std1]),0.5*np.array([peak,peak]),
                 linestyle='--',color='black',linewidth=1.0)
        plt.plot(xh,normd,linestyle='--',color='red',linewidth=2.0)
        plt.xlabel('E[x]',fontsize=ftsz)
        plt.ylabel('frequency',fontsize=ftsz)
        plt.title('E[X]  =%11.3e\nVar[X]=%11.3e' % (avg1,std1),x=0.2,y=0.7,fontsize=ftsz)
        plt.ylim(np.array([0.0,1.05*np.max(hist)]))
        plt.tick_params(labelsize=ftsz)
        plt.show()
    else: 
        raise Exception('[mci_llnclt]: invalid itest %i' % (itest))

#===================================
main()
