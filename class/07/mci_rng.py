#===================================
# Demonstrator for random number generators.
#===================================
import argparse                  # allows us to deal with arguments to main()
from argparse import RawTextHelpFormatter
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import p358utilities as util
#===================================
iseed    = 151
#===================================
# eps = test_correlate(fRAN,N,shift)
# calculates eps(R,n)=sum(x(i)*x(i+n)-E(x)^2)
# for a set of sample lengths R and shifts n.
# input : fRAN: function pointer to random number generator
# output: eps : (R,n) array containing rms difference
#-----------------------------------
def test_correlate(fRAN):
    R    = np.array([1000,3000,10000,30000,100000,300000,1000000,3000000])
    n    = np.array([1,3,10,30])
    eps  = np.zeros((R.size,n.size))
    for i in range(R.size):
        x  = fRAN(R[i])
        Ex = np.mean(x)
        for j in range(n.size):
            xs       = np.roll(x,n[j])
            eps[i,j] = np.abs(np.mean(x*xs)-Ex**2)
            #eps[i,j] = np.mean(x*np.roll(x,n[j]))-Ex**2

    return R,n,eps

#===================================
def test_moment(fRAN):
    R   = np.array([1000,3000,10000,30000,100000,300000,1000000,3000000])
    mom = np.zeros((R.size,4))
    for j in range(4):
        for i in range(R.size):
            x        = fRAN(R[i])
            mom[i,j] = np.abs(np.mean(x**(1+j)-1.0/(float(j+2))))
    return R,mom

#===================================
def test_histogram(fRAN):
    R         = 10000
    x         = fRAN(R)
    nbin      = np.int(np.sqrt(np.float(R)))
    hist      = np.zeros((nbin,2))
    h,e       = np.histogram(x,nbin,range=(0,1),normed=False)
    p         = 0.5*(e[0:e.size-1]+e[1:e.size])
    hist[:,0] = p
    hist[:,1] = h
    return hist

#===================================
def test_tuple2d(fRAN):
    R         = 10000
    x         = fRAN(R)
    x1        = x[2*np.arange(R//2)]   # even indices
    x2        = x[2*np.arange(R//2)+1] # odd indices
    return x1,x2

#===================================
def test_tuple3d(fRAN):
    R         = 9999
    x         = fRAN(R)
    x1        = x[3*np.arange(R//3)] 
    x2        = x[3*np.arange(R//3)+1]
    x3        = x[3*np.arange(R//3)+2]
    y1        = x1/np.sqrt(x1**2+x2**2+x3**2)
    y2        = x2/np.sqrt(x1**2+x2**2+x3**2)
    y3        = x3/np.sqrt(x1**2+x2**2+x3**2)
    return y1,y2,y3

#===================================
def lincon(R):
    a    = 106
    c    = 1283
    m    = 6075
    i    = np.zeros(R)
    i[0] = iseed 
    for j in range(1,R):
        i[j] = (a*i[j-1]+c) % m
    return i.astype(float)/float(m-1)

#===================================
def main():
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument("s_rng",type=str,
                        help="random number generator:\n"
                             "   python   : built-in python generator\n"
                             "   lincon   : linear congruential")

    args       = parser.parse_args()
    s_rng      = args.s_rng

    if (s_rng == 'python'):
        fRAN = np.random.rand
    elif(s_rng == 'lincon'):
        fRAN = lincon
    else:
        raise Exception('[mci_rng]: invalid RNG choice: %s)' % (s_rng))

    # correlation test
    Reps,neps,eps = test_correlate(fRAN)
    # moment test
    Rmom,mom      = test_moment(fRAN)
    # 2d tuple test
    x1,x2         = test_tuple2d(fRAN)
    # 3d tuple test
    y1,y2,y3      = test_tuple3d(fRAN)
    # histogram
    hist          = test_histogram(fRAN)

    # plot everything
    ftsz = 10
    fig=plt.figure(num=1,figsize=(8,8),dpi=100,facecolor='white')
    # histogram
    plt.subplot(321)
    plt.bar(hist[:,0],hist[:,1],width=(hist[1,0]-hist[0,0]),facecolor='green',align='center')
    plt.xlabel('$x_r$',fontsize=ftsz)
    plt.ylabel('#',fontsize=ftsz)
    plt.tick_params(labelsize=ftsz)
    # correlation
    plt.subplot(322)
    for j in range(neps.size):
        plt.plot(np.log10(Reps),np.log10(eps[:,j]),linewidth=1,linestyle='-',label='n=%5i' % (neps[j]))
    plt.xlabel('log R',fontsize=ftsz)
    plt.ylabel('log $\epsilon$',fontsize=ftsz)
    #plt.legend(loc=4)    
    plt.tick_params(labelsize=ftsz)
    # moment
    plt.subplot(323)
    for j in range((mom.shape)[1]):
        plt.plot(np.log10(Rmom),np.log10(mom[:,j]),linewidth=1,linestyle='-',label='k=%5i' % (j+1))
    plt.xlabel('log R',fontsize=ftsz)
    plt.ylabel('log $\mu$',fontsize=ftsz)
    plt.tick_params(labelsize=ftsz)
    # 2d tuple 
    plt.subplot(324)
    plt.scatter(x1,x2,s=1)
    plt.xlabel('$x_1$',fontsize=ftsz)
    plt.ylabel('$x_2$',fontsize=ftsz)
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.tick_params(labelsize=ftsz)
    # 3d tuple test
    ax = fig.add_subplot(325,projection='3d')
    ax.scatter(y1,y2,y3,s=1)
    ax.view_init(elev=30.0,azim=60.0)
    ax.set_xlabel('$x_1$',fontsize=ftsz)
    ax.set_ylabel('$x_2$',fontsize=ftsz)
    ax.set_zlabel('$x_3$',fontsize=ftsz)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(5)
    for tick in ax.yaxis.get_major_ticks():  
        tick.label.set_fontsize(5)
    for tick in ax.zaxis.get_major_ticks():  
        tick.label.set_fontsize(5)
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_zlim(0,1)



    plt.show()

#===================================

main()
