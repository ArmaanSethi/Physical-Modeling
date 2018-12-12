#===============================
# PDE: Convolution solver for Poisson equation
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
    Wmat = W**(j*k)
    F    = np.dot(Wmat,f)
    return F

#===============================
# function poissonfft(rho)
# input : rho    : the density field 
#         dx     : distance between support points
# output: phi    : the potential
def poissonfft(rho,dx):
  J         = len(rho)
  rhohat    = sft(rho,+1)*dx**2
  phihat    = 0.5*rhohat/(np.cos(2.0*np.pi*np.arange(J)/float(J))-1.0)
  phihat[0] = 0.0
  phi       = np.real((sft(phihat,-1)))
  return phi

#===============================
def init(problem,J):
    if (problem == 'point'):
       n    = 1.0
       x    = 2.0*(0.5+np.arange(J))/J-1.0
       rho = np.zeros(J)
       rho[J//2-1:J//2+1]=1.0
    else:
       print('[init]: invalid problem %s' % (problem))
       exit()
    dx = x[1]-x[0]
    return x,rho,dx

#===============================

def main():
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument("J",type=int,
                        help="number of support points")
    parser.add_argument("problem",type=str,
                        help="density field:\n"
                             "    point   : point charge")
    parser.add_argument("-j","--Jmax",type=int,default=None,help="maximum J")
    args        = parser.parse_args()
    J           = args.J
    problem     = args.problem
    Jmax        = args.Jmax

    

    if (Jmax):
        nx = int(np.log(Jmax//J)/np.log(2)+1)
    else:
        nx = 1
    x          = np.zeros(nx,dtype=object)
    rho        = np.zeros(nx,dtype=object)
    dx         = np.zeros(nx)
    phi        = np.zeros(nx,dtype=object)
    rhomin     = np.zeros(nx)
    rhomax     = np.zeros(nx)
    phimin     = np.zeros(nx)
    phimax     = np.zeros(nx)
    for i in range(nx):
        x[i],rho[i],dx[i] = init(problem,J*int(2**i)) 
        phi[i]            = poissonfft(rho[i],dx[i])
        rhomin[i]         = np.min(rho[i])
        rhomax[i]         = np.max(rho[i])
        phimin[i]         = np.min(phi[i])
        phimax[i]         = np.max(phi[i])
    rhomin     = np.min(rhomin)
    rhomax     = np.max(rhomax)
    phimin     = np.min(phimin)
    phimax     = np.max(phimax)

    ftsz       = 10
    plt.figure(num=1,figsize=(6,6),dpi=100,facecolor='white')
    plt.subplot(2,1,1)
    for i in range(nx):
        plt.plot(x[i],rho[i],linestyle='-',label=("J=%4i" % (J*2**i)))
    plt.xlabel('x')
    plt.ylabel(r"$\rho(x)$")
    plt.legend(fontsize=ftsz)
    plt.tick_params(labelsize=ftsz)
    util.rescaleplot(x[0],rho[0],plt,0.05,ymin=rhomin,ymax=rhomax)
    plt.subplot(2,1,2)
    for i in range(nx):
        plt.plot(x[i],phi[i],linestyle='-',label=("J=%4i" % (J*2**i)))
    plt.xlabel('x')
    plt.ylabel(r"$\Phi(x)$")
    plt.legend(fontsize=ftsz)
    plt.tick_params(labelsize=ftsz)
    util.rescaleplot(x[0],phi[0],plt,0.05,ymin=phimin,ymax=phimax)
    plt.tight_layout()
    plt.show()

     
#=====================================

main()
