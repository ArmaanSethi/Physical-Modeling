#!/usr/bin/python
import argparse                  # allows us to deal with arguments to main()
from argparse import RawTextHelpFormatter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from  decimal import * 
import random
import cost     # class providing wrapper for cost function
import fnc_cost # cost function library
#=========================================
# children = breed(parents,rate)
#
# Takes two parameter vectors whose elements
# must be normalized between 0.0 <= x < 1.0
# and breeds them (crossover), including
# random changes (mutation).
# 
# input: parents: (ndim,2) array, where ndim
#                 is the length of the vector (i.e. the
#                 vector is in R^N). 
#        rate   : mutation rate: sets the probability
#                 (between 0 and 1) that a gene in a 
#                 parent's genome is changed.   
#----------------------------------------- 
def breed(parents,rate):
    iverb= 0 # set to > 0 for diagnostic output
    p1   = parents[:,0]
    p2   = parents[:,1]
    nlen = 8 # significant digits (length of genome)
    sp1  = ''
    sp2  = ''
    # encoding: if p1 consists of three coordinates x1,x2,x3, sp1 will be x1x2x3 
    # (leading zeroes removed) 
    # Encoding consists of two steps: turn input into decimals (to remove exponents),
    # then turn these into strings.
    for x in p1: 
        sp1 = sp1 + ((("%s" % (Decimal(x))).split('.',1)[-1]).ljust(nlen,'0'))[0:nlen]
    for x in p2:
        sp2 = sp2 + ((("%s" % (Decimal(x))).split('.',1)[-1]).ljust(nlen,'0'))[0:nlen]
    if (iverb):
        print('    p1 = %8.6f %8.6f sp1  = %s' % (p1[0],p1[1],sp1))
        print('    p2 = %8.6f %8.6f sp2  = %s' % (p2[0],p2[1],sp2))

    # crossover: Cut genome of parents at random position, swap.
    ig   = np.random.randint(0,len(sp1)-1)
    so1  = sp1[0:ig] + sp2[ig:len(sp1)]
    so2  = sp2[0:ig] + sp1[ig:len(sp1)]
    if (iverb):
        print('    ig = %3i               so1  = %s' % (ig,so1))
        print('    ig = %3i               so2  = %s' % (ig,so2))
    # mutation: Step through genome and randomly change genes.
    som1 = ''
    som2 = ''
    for char in so1:
        if (random.random() < rate):
            char = random.choice('0123456789')
        som1 = som1+char
    for char in so2:
        if (random.random() < rate):
            char = random.choice('0123456789')
        som2 = som2+char
    if (iverb):
        print('                           som1 = %s' % (so1))
        print('                           som2 = %s' % (so2))
    # decoding: turn strings into decimals, then floating points. 
    if (isinstance(p1,np.ndarray)):
        ndim = p1.size
        o1   = np.zeros(ndim)
        o2   = np.zeros(ndim)
        for i in range(ndim):
            o1[i] = float(Decimal('0.%s' % (som1[i*nlen:(i+1)*nlen])))
            o2[i] = float(Decimal('0.%s' % (som2[i*nlen:(i+1)*nlen])))
    else:
        ndim = 1
        o1   = float(Decimal(som1))
        o2   = float(Decimal(som2))

    if (iverb):
        print('    o1 = %8.6f %8.6f som1 = %s' % (o1[0],o1[1],som1))
        print('    o2 = %8.6f %8.6f som2 = %s' % (o2[0],o2[1],som2))

    children      = np.zeros(parents.shape)  
    children[:,0] = o1
    children[:,1] = o2
    return children

#=======================================
# rank = stochastic_accept(weight)
#
# Returns indices of input array ordered
# probabilistically according to weight
# by picking a random position and determining
# how likely it is going to be accepted given
# its weight.
# Input : weight: array of weights (not necessarily normalized)
# Output: rank: index array of length weight.size containing
#         indices of elements most commonly accepted.
#---------------------------------------
def stochastic_accept(weight):
    n    = weight.size
    maxw = np.max(weight)
    rank = np.zeros(n,dtype=int)
    for j in range(n): # positions to be filled
        notfilled = True
        while (notfilled): 
            i = np.random.randint(0,n) # choose randomly a position
            if (np.random.rand(1) <= weight[i]/maxw): # accept if its weight allows it
                rank[j]   = i
                notfilled = False

    return rank

#=======================================
# fittest = geneticalg(cCST,npop,rate,maxit)
#
# Returns the "fittest" member of an evolutionary
# sequence using the function breed(parents)
# Input: cCST : instance of class Cost: contains
#              the cost function and its bounds.
#        npop : size of population to breed from (usually 100)
#        rate : mutation rate used by breed(parents)
#        maxit: maximum number of iterations. Use this rather
#               that tolerance.
# Output: fittest: parameter vector corresponding to 
#                  best solution (maximum or minimum, depending
#                  on settings of cCST).
# Note: The algorithm assumes a maximization problem. Minimizations
#       need to be formulated as 1/cCST.eval (see cost.py).
#---------------------------------------
def geneticalg(cCST,npop,rate,maxit,**kwargs):
    iverb = 0
    for key in kwargs:
        if (key=='iverb'):
            iverb = kwargs[key]

    # get information from cost function
    ndim = cCST.ndim()
    tol  = cCST.tol() # checks consecutive best fits for discrepancy 
    pop  = np.zeros((ndim,npop))

    # initialize starting population
    rnd  = np.random.rand(ndim,npop)
    for p in range(npop):
        pop[:,p] = cCST.denormalize(rnd[:,p])
    fit_curr = np.zeros(npop)           # note that fit == 0.0 is best here,
                                        # to allow for minimization problems
    ind_breed= np.zeros(npop,dtype=int) # index of pop members to be bred
    ind_best = -1
    parents  = np.zeros((ndim,npop))    # normalized (parent) population
    children = np.zeros((ndim,npop))    # normalized (children) population

    # To start, evaluate fitness of all population members
    # and determine breeding rank. 
    for p in range(npop):
        fit_curr[p] = cCST.eval(pop[:,p])
    ind_breed[:] = stochastic_accept(fit_curr)
    ind_best     = np.argmax(fit_curr)
    ind_wrst     = np.argmin(fit_curr)
    err          = cCST.err(pop[:,ind_best],pop[:,ind_wrst])

    if (iverb == 1):
        if (ndim > 2):
            raise Exception("[geneticalg]: --verbose not possible for ndim > 2")
        ppoints              = np.zeros(2)
        nres                 = 200
        fmap                 = np.zeros((nres,nres))
        x                    = np.arange(nres)/float(nres-1)
        y                    = np.arange(nres)/float(nres-1)
        for j in range(nres): # column
            for i in range(nres): # row, counted from top.
                fmap[nres-1-i,j] = cCST.eval(cCST.denormalize(np.array([x[j],y[i]])),function=True)
        fmax                = np.max(fmap)
        fmin                = np.min(fmap)
        fmap[(fmap > fmax)] = fmax
        fmap[(fmap < fmin)] = fmin
        bds                 = cCST.bounds()
        xmin                = bds[0,0]
        xmax                = bds[0,1]
        ymin                = bds[1,0]
        ymax                = bds[1,1]
        ftsz = 10
        plt.figure(num=1,figsize=(8,8),dpi=100,facecolor='white')
        plt.subplot(111)
        plt.imshow(fmap,extent=(xmin,xmax,ymin,ymax),interpolation='nearest',cmap=cm.gist_rainbow)
        plt.xlabel('x',fontsize=ftsz)
        plt.ylabel('y',fontsize=ftsz)
        plt.tick_params(labelsize=ftsz)
        ppoints = cCST.denormalize(rnd[:,ind_best])
        plt.scatter(ppoints[0],ppoints[1],color='white',s=2)

    # Loop until condition is met.
    it = 0
    while ((err > tol) and (it < maxit)): 
        #ind_wrst = ind_best # store for comparison in next iteration
        # (1) normalize genomes of parent population and rank them according to ind_breed
        for p in range(npop):
            parents[:,p] = cCST.normalize(pop[:,ind_breed[p]]) 
        # (2) breed the parents
        for p in range(int(npop/2)): # only half the range, because breed needs pairs of parents
            children[:,2*p:2*(p+1)] = breed(parents[:,2*p:2*(p+1)],rate) 
        # (3) denormalize genomes of children population and assign back
        for p in range(npop):
            pop[:,p] = cCST.denormalize(children[:,p]) 
        # (4) evaluate fitness of children and find ranking according to fitness
        for p in range(npop):
            fit_curr[p] = cCST.eval(pop[:,p]) 
        ind_breed[:] = stochastic_accept(fit_curr)
        ind_best     = np.argmax(fit_curr) # find fittest member
        ind_wrst     = np.argmin(fit_curr)
        err          = cCST.err(pop[:,ind_best],pop[:,ind_wrst])
        it           = it+1
        if (iverb == 1):
            ppoints = cCST.denormalize(children[:,ind_best])
            plt.scatter(ppoints[0],ppoints[1],color='white',s=2)
        
    if (iverb == 1):
        plt.show()

    if (it == maxit):
        print('[geneticalg]: warning: reached maximum number of iterations. err=%13.5e tol=%13.5e' % (err,tol))
    else:
        print('[geneticalg]: reached err <= tol after %6i iterations: err=%13.5e tol=%13.5e' % (it,err,tol))
    return pop[:,ind_best] 

#=======================================
# init
#---------------------------------------
def init(s_cost):

    if (s_cost == 'parabola1d'):
        bounds = np.array([[0.0,1.0]])
        maxmin = -1
        tol    = 1e-5
        fCST   = fnc_cost.parabola1d
    elif (s_cost == 'parabola2d'):
        bounds = np.array([[-1.0,1.0],[-1.0,1.0]])
        maxmin = -1 
        tol    = 1e-5
        fCST   = fnc_cost.parabola2d
    elif (s_cost == 'sphere'):
        bounds = np.array([[-1.0,1.0],[-1.0,1.0],[-1.0,1.0]])
        maxmin = -1
        tol    = 1e-5
        fCST   = fnc_cost.sphere
    elif (s_cost == 'charbonneau1'):
        bounds = np.array([[0.0,1.0],[0.0,1.0]])
        maxmin = 1 
        tol    = 1e-5
        fCST   = fnc_cost.charbonneau1
    elif (s_cost == 'charbonneau2'):
        bounds = np.array([[0.0,1.0],[0.0,1.0]])
        maxmin = 1
        tol    = 1e-5
        fCST   = fnc_cost.charbonneau2
    elif (s_cost == 'charbonneau4'):
        bounds = np.array([[0.0,2.0],[-1.0,1.0],[0.01,1.0],[0.0,2.0],[-1.0,1.0],[0.001,1.0]])
        maxmin = -1
        tol    = 1e-5
        fCST   = fnc_cost.charbonneau4
    else: 
        raise Exception('[init]: invalid s_prob %s')

    return cost.Cost(fCST,bounds,tol,maxmin)

#=======================================
def main():

    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument("s_cost",type=str,
                        help="function to be minimized:\n"
                             "   parabola2d \n"
                             "   sphere \n"
                             "   charbonneau1 \n"
                             "   charbonneau2 \n"
                             "   charbonneau4 ")
    parser.add_argument("npop",type=int,
                        help="size of population (usually 100)")
    parser.add_argument("rate",type=float,
                        help="mutation rate (usually 0.01)")
    parser.add_argument("maxit",type=int,
                        help="maximum number of iterations (depends on problem)")
    parser.add_argument("--verbose", help="print and show diagnostic information",action="store_true")

    args       = parser.parse_args()
    s_cost     = args.s_cost
    npop       = args.npop
    rate       = args.rate
    maxit      = args.maxit
    if args.verbose:
        iverb = 1
    else:
        iverb = 0


    cCST       = init(s_cost)

    best       = geneticalg(cCST,npop,rate,maxit,iverb=iverb)

    print(best)

#=======================================
main()
