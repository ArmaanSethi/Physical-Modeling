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
    N = len(f)
    W = np.empty((N,N))
    i,j = np.meshgrid(np.arange(N), np.arange(N))
    if direction == 1:#forward
        # print("FORWARD")
        W = np.exp(-2*i*j*(1j)*np.pi/N)
        F = W@f
    elif direction == -1:#inverse
        W = np.exp(2*i*j*(1j)*np.pi/N)/N
        F = W@f
        # print("INVERSE")
    else:
        print("Not Valid Direction")
    # print(F)
    
    return F

#===============================
# function poissonfft(rho)
# input : rho    : the density field 
#         dx     : distance between support points
# output: phi    : the potential
def poissonfft(rho,dx):
    J = len(rho)
    rho_k = sft(rho,1) #forward
    phi_hat = np.zeros(J)
    for k in range(1,J): 
        num = (rho_k[k]*dx**2)
        denom = (2*(np.cos(2*np.pi*k/J)-1))
        phi_hat[k] = num/denom
    phi = sft(phi_hat, -1) #inverse
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

    #???? here you'll need to set up the resolution sequence, initialize
    #     the problem for each resolution, and call the poisson solver
    x_list = list()
    phi_list = list()
    rho_list = list()
    J_list = list()
    if Jmax:#make sure Jmax exists
        for i in range(J, Jmax+1):
            if Jmax%i==0: #if divisible by Jmax
                print(Jmax,i)
                x,rho,dx = init(problem,i)
                phi = poissonfft(rho,dx)
                x_list.append(x)
                phi_list.append(phi)
                rho_list.append(rho)
                J_list.append(i)
    else:#no Jmax exists
        x,rho,dx = init(problem,J)
        phi = poissonfft(rho,dx)
        x_list = list(x)
        phi_list = list(phi)
        rho_list = list(rho)
        J_list = list(J)

    #???? ... and then you'll need to plot everything in two plots, one for
    #     rho, and one for phi.

    for i in range(len(J_list)):
        plt.subplot(121)
        plt.plot(x_list[i],rho_list[i],linestyle='-',linewidth=1,label='J={0}'.format(J_list[i]))
        plt.xlabel('x')
        plt.ylabel('ρ (rho)')
        plt.legend()
        plt.subplot(122)
        plt.plot(x_list[i],phi_list[i],linestyle='-',linewidth=1,label='J={0}'.format(J_list[i]))
        plt.xlabel('x')
        plt.ylabel('Φ (phi)')
        plt.legend()


    #????
    plt.tight_layout()
    plt.show()

     
#=====================================

main()
