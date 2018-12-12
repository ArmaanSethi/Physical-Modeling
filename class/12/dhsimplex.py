#!/usr/bin/python
import argparse                  # allows us to deal with arguments to main()
from argparse import RawTextHelpFormatter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cost     # class providing wrapper for cost function
import fnc_cost # cost function library

#========================================
# function xmin,it = dhsimplex(cCST, maxit)
#
# Determines the minimum of f using 
# the downhill simplex method.
#
# input : sinit: the initial simplex, unordered.
#                n*(n+1) matrix, where n is the number of 
#                parameters (dimensions), and thus n+1 is 
#                the minimum coordinate numbers for a simplex.
#         f    : the function. Must accept n-vector
#         maxit: maximum number of iterations used
#         tol  : tolerance (applied on f)
#         iverb: 0: no diagnostics
#                1: prints out progress
#                2: shows map with each simplex
# output: xmin : the minimum (center of gravity of final
#                simplex).
#         it   : number of iterations used
# Note: uses normalized parameters.
#=========================================

def dhsimplex(cCST,maxit,**kwargs):
    iverb = 0
    for key in kwargs:
        if (key=='iverb'):
            iverb = kwargs[key]

    ndim = cCST.ndim()
    tol  = cCST.tol()
    nver = ndim+1
    sinit= np.random.rand(ndim,nver)
    fcur = np.zeros(nver)    # n+1 function values (current)
    fsor = np.zeros(nver)    # sorted function values
    scur = np.copy(sinit)    # initialize with first simplex. We need a new instance of sinit.
    ssor = np.zeros((ndim,nver))  # sorted simplex positions.
    xmin = np.zeros(ndim) 
    xref = np.zeros(ndim)  # reflected point
    xexp = np.zeros(ndim)  # expanded point
    xcon = np.zeros(ndim)  # contracted point
    x0   = np.zeros(ndim) 
    x1   = np.zeros(ndim) 
    fref = 0.0   # function values for the various actions on simplex
    fexp = 0.0 
    fcon = 0.0 
    dx   = 0.0   # keeps track of largest distance between simplex points
    dxs  = 0.0 
    istol= 0 
    done = 0     # keeps track of whether we're done within one iteration
    it   = 0 
    jmax = 0 

#   these are the coefficients for...
    alpha= 1.0  # ...reflection
    gamma= 2.0  # ...expansion
    rho  = 0.5  # ...contraction
    sigma= 0.5  # ...reduction

# only for debugging purposes
    if (iverb == 1):
        if (ndim > 2):
            raise Exception("[dhsimplex]: --verbose not possible for ndim > 2")
        nres = 200

        fmap = np.zeros((nres,nres))
        x    = np.arange(nres)/float(nres-1)
        y    = np.arange(nres)/float(nres-1)
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

        psimp               = np.zeros((2,4)) # for plotting. Last element is first -- makes plotting easier.

        ftsz = 10
        plt.figure(num=1,figsize=(8,8),dpi=100,facecolor='white')

        plt.subplot(111)
        plt.imshow(fmap,extent=(xmin,xmax,ymin,ymax),interpolation='nearest',cmap=cm.gist_rainbow)
        plt.xlabel('x',fontsize=ftsz)
        plt.ylabel('y',fontsize=ftsz)
        plt.tick_params(labelsize=ftsz)

        for i in range(3):
            psimp[:,i] = cCST.denormalize(sinit[:,i])
        psimp[:,3] = cCST.denormalize(sinit[:,0])
        for i in range(3):
            plt.plot(psimp[0,i:i+2],psimp[1,i:i+2],linestyle='-',color='white')

        #plt.plot(np.array([sinit[0,0],sinit[0,1]]),np.array([sinit[1,0],sinit[1,1]]),linestyle='-',color='white')
        #plt.plot(np.array([sinit[0,1],sinit[0,2]]),np.array([sinit[1,1],sinit[1,2]]),linestyle='-',color='white')
        #plt.plot(np.array([sinit[0,2],sinit[0,0]]),np.array([sinit[1,2],sinit[1,0]]),linestyle='-',color='white')

    while ((it < maxit) and (istol == 0)):
        done = 0 # start again
#       step 1: calculate f(x) for all vertices, and sort
        for j in range(nver):
            fcur[j] = cCST.eval(cCST.denormalize(scur[:,j]))
        indx = np.argsort(fcur)
        fsor = fcur[indx]
        ssor = scur[:,indx] # re-index the positions 
#       step2: calculate the center of gravity x0 for all points except the worst x(n+1)
        sumf = np.sum(fsor[0:nver-1]) # note that this addresses all elements 0,...,ndim-2
        for i in range(ndim):
            x0[i] = np.sum(fsor[0:nver-1]*ssor[i,0:nver-1])/sumf;
        if (iverb > 0):
            print('[dhsimplex]: it=%4d - centroid: f = %13.5e' % (it+1,cCST.eval(cCST.denormalize(x0))));
#       step3: reflection
        xref = x0+alpha*(x0-ssor[:,nver-1])
        fref = cCST.eval(cCST.denormalize(xref))
        if ((fsor[0] <= fref) and (fref < fsor[ndim-1])):
            if (iverb > 0):
                print('[dhsimplex]: it=%4d - reflection: f = %13.5e' % (it+1,fref))
            ssor[:,nver-1] = xref[:]
            done         = 1
#       step4: expansion
        if (done == 0):
            if (fref < fsor[0]):
                xexp = x0+gamma*(x0-ssor[:,nver-1])
                fexp = cCST.eval(cCST.denormalize(xexp))
                if (iverb > 0):
                    print('[dhsimplex]: it=%4d - expansion:   f = %13.5e' % (it+1,fexp))
                if (fexp < fref):
                    ssor[:,nver-1] = xexp[:]
                    fsor[nver-1]   = fexp
                else:
                    ssor[:,nver-1] = xref[:]
                    fsor[nver-1]   = fref
                done      = 1
#       step5: contraction
        if (done == 0):
            if (fref < fsor[nver-2]):
                print('[dhsimplex]: invalid branch in step 5\n')
                stop
            xcon = ssor[:,nver-1]+rho*(x0-ssor[:,nver-1])
            fcon = cCST.eval(cCST.denormalize(xcon))
            if (fcon < fsor[nver-1]):
                if (iverb > 0):
                    print('[dhsimplex]: it=%4d - contraction: f = %13.5e' % (it+1,fcon))
                ssor[:,nver-1] = xcon[:]
                fsor[nver-1]   = fcon
                done           = 1
#       step6: reduction
        if (done == 0):
            if (iverb > 0):
                print('[dhsimplex]: it=%4d - reduction' % (it+1))
            for j in range(nver-1): 
                ssor[:,j+1] = ssor[:,0]+sigma*(ssor[:,j+1]-ssor[:,0])
                fsor[j+1]   = cCST.eval(cCST.denormalize(ssor[:,j+1]))
#       still need to compare current f etc (all steps).
        scur[:,:] = ssor[:,:] # need to write back the modified simplex as new input
        it = it+1
#       calculate largest distance between points: this is not recommended for real applications...
        dx = -1e10
        for i in range(ndim):
            for j in range(i+1,nver):
                dxs = np.sqrt(np.sum(np.power(ssor[:,i]-ssor[:,j],2)))
                if (dxs > dx):
                    dx   = dxs
                    jmax = j
        if (dx < tol):
            istol = 1
            xmin  = ssor[:,jmax]
        print('[dhsimplex]: it=%4d, dx = %13.5e' % (it,dx));
    
#       for visualization: plot most recent simplex
        if (iverb == 1):
            for i in range(3):
                psimp[:,i] = cCST.denormalize(ssor[:,i])
            psimp[:,3] = cCST.denormalize(ssor[:,0])
            for i in range(3):
                plt.plot(psimp[0,i:i+2],psimp[1,i:i+2],linestyle='-',color='white')

        if (it == maxit):
            print('[dhsimplex]: reached maximum number of iterations')

    if (iverb == 1):
        plt.show()

    return cCST.denormalize(xmin)

#=======================================
# init
# Note that the parameter maxmin needs to 
# be set with the optimization method in mind.
# The value of maxmin is set according to the following table:
#   cost|min|max
# opt 
#----------------
# min   | 1 | -1
# max   |-1 |  1
#
# If your function needs to be minimized and your algorithm
# is a minimizer, set maxmin = 1, etc.
#
#---------------------------------------
def init(s_cost):

    if (s_cost == 'parabola1d'):
        bounds = np.array([[0.0,1.0]])
        maxmin = 1
        tol    = 1e-6
        fCST   = fnc_cost.parabola1d
    elif (s_cost == 'parabola2d'):
        bounds = np.array([[-1.0,1.0],[-1.0,1.0]])
        maxmin = 1
        tol    = 1e-8
        fCST   = fnc_cost.parabola2d
    elif (s_cost == 'sphere'):
        bounds = np.array([[-1.0,1.0],[-1.0,1.0],[-1.0,1.0]])
        maxmin = 1
        tol    = 1e-6
        fCST   = fnc_cost.sphere
    elif (s_cost == 'charbonneau1'):
        bounds = np.array([[0.0,1.0],[0.0,1.0]])
        maxmin = -1
        tol    = 1e-6
        fCST   = fnc_cost.charbonneau1
    elif (s_cost == 'charbonneau2'):
        bounds = np.array([[0.0,1.0],[0.0,1.0]])
        maxmin = -1
        tol    = 1e-8
        fCST   = fnc_cost.charbonneau2
    elif (s_cost == 'charbonneau4'):
        bounds = np.array([[0.0,2.0],[-1.0,1.0],[0.01,1.0],[0.0,2.0],[-1.0,1.0],[0.001,1.0]])
        maxmin = 1
        tol    = 1e-8
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
                             "   charbonneau1 \n"
                             "   charbonneau2 \n"
                             "   charbonneau3")
    parser.add_argument("maxit",type=int,
                        help="maximum number of iterations (depends on problem)")
    parser.add_argument("--verbose", help="print and show diagnostic information",action="store_true")

    args       = parser.parse_args()
    s_cost     = args.s_cost
    maxit      = args.maxit
    if args.verbose:
        iverb = 1
    else: 
        iverb = 0

    cCST       = init(s_cost)

    best       = dhsimplex(cCST,maxit,iverb=iverb)

    print(best)

#=======================================
main()



