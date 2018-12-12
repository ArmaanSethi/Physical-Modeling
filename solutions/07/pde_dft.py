#===============================
# PDE discrete Fourier transforms
#===============================
import argparse                  # allows us to deal with arguments to main()
from argparse import RawTextHelpFormatter
import numpy as np
import matplotlib.pyplot as plt
import p358utilities as util
#===============================
# function sft (slow Fourier transform)
# input: f        : data vector f
#        direction: +1: forward transform, -1: inverse transform
# output: F       : the Fourier transform of f
#-------------------------------
def sft(f,direction):
    J    = f.size
    W    = np.exp(2.0j * np.pi * float(direction) /float(J))
    j,k  = np.meshgrid(np.arange(J),np.arange(J))
    Wmat = W**(j*k)#np.zeros((J,J),dtype=complex)
    F    = np.dot(Wmat,f)
    if (direction == 1):
        F = F / float(J)
    return F

#===============================
def init(problem,J,k):
    if (problem == 'sine'):
       x = 2.0*np.pi*np.arange(J)/float(J)
       f = np.sin(k*x)
    else:
       print('[init]: invalid problem %s' % (problem))
       exit()
    return x,f

#===============================

def main():
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument("J",type=int,
                        help="number of support points")
    parser.add_argument("problem",type=str,
                        help="data field:\n"
                             "    sine   : well, sine")
    parser.add_argument("k",type=int,help="wave number")
                    
    args        = parser.parse_args()
    J           = args.J
    problem     = args.problem
    k           = args.k


    x,f         = init(problem,J,k)

    F           = sft(f,1)
    ff          = sft(F,-1)
    F           = np.abs(F)
  
    print(F)

    ftsz        = 10
    plt.figure(num=1,figsize=(6,6),dpi=100,facecolor='white')
    plt.subplot(121)
    plt.plot(x,f,linestyle='-',linewidth=1,color='black',label='$f(x)$')
    plt.plot(x,ff,linestyle='-',linewidth=1,color='red',label='$F^{-1}[F[f]]$')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend(fontsize=ftsz)
    plt.tick_params(labelsize=ftsz)
    util.rescaleplot(x,f,plt,0.05)
    plt.subplot(122)
    plt.plot(np.arange(J),F,'o',color='black')
    plt.xlabel('k')
    plt.ylabel('F[f]')
    plt.tick_params(labelsize=ftsz)
    util.rescaleplot(np.arange(J),F,plt,0.05)
    plt.tight_layout()
    plt.show()

     
#=====================================

main()
