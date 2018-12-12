#=============================================================
import argparse                  # allows us to deal with arguments to main()
from argparse import RawTextHelpFormatter
import numpy as np
import matplotlib.pyplot as plt
import cost
import fnc_cost
import p358utilities as util
#=============================================================
# function bracket_min(a,b,f)
#
# Returns values of a,b bracketing minimum of f.
# Use together with golden (1D minimization)
#
# input: 
#   a, b : starting guess for minimum bracket
#   f    : function to be minimized
# output: 
#   a,b,c: as starting values for golden
#--------------------------------------------------------------

def bracket_min(cCST):
    gold  = 1.618034
    limit = 100.0
    tiny  = 1e-20

    dum   = 0.0

    bds   = cCST.bounds()
    a     = bds[0,0]
    b     = bds[0,1] 
    
    fa    = cCST.eval(a)
    fb    = cCST.eval(b)
    if (fb > fa):
        dum = a
        a   = b
        b   = dum
        dum = fb
        fb  = fa
        fa  = dum
    c     = b+gold*(b-a)
    fc    = cCST.eval(c)
    
    while (fb > fc):
        r = (b-a)*(fb-fc)
        q = (b-c)*(fb-fa)
        s = np.max(np.array([np.abs(q-r),tiny]))
        if (q-r < 0.0):
            ss = -s
        else:
            ss = s
        u    = b-((b-c)*q-(b-a)*r)/(2.0*ss)
        ulim = b+limit*(c-b)
        if ((b-u)*(u-c) > 0.0):
            fu = cCST.eval(u)
            if (fu < fc):
                a  = b
                b  = u
                fa = fb
                fb = fu
                return a,b,c
            elif (fu > fb):
                c  = u
                fc = fu
                return a,b,c
            u  = c+gold*(c-b)
            fu = cCST.eval(u)
        elif ((c-u)*(u-ulim) > 0.0):
            fu = cCST.eval(u)
            if (fu < fc):
                b  = c
                c  = u
                u  = c+gold*(c-b)
                fb = fc
                fc = fu
                fu = f(u)
        elif ((u-ulim)*(ulim-c) >= 0.0):
            u  = ulim
            fu = cCST.eval(u) 
        else:
            u  = c+gold*(c-b)
            fu = cCST.eval(u)
        a  = b
        b  = c
        c  = u
        fa = fb
        fb = fc
        fc = fu 
        print('bracket: a,b,c=%13.5e %13.5e %13.5e fa,fb,fc=%13.5e %13.5e %13.5e' % (a,b,c,fa,fb,fc))
    return a,b,c
         
#=============================================================
# function golden(a,b,f,tol)
#
# Returns location of minimum of f within tolerance tol
#
# input: 
#   a, b : starting guess for minimum bracket
#   f    : function to be minimized
#   tol  : tolerance (accuracy). ~sqrt(machine_precision),
#          see Num.Rec for discussion
# output: 
#   xmin : location of minimum of f.
# note:
#   There is no safeguard against finding a local minimum only.
#--------------------------------------------------------------
def golden(cCST,**kwargs):
    bds    = cCST.bounds()
    a      = bds[0,0]
    b      = bds[0,1]
    tol    = cCST.tol()

    C      = 0.5*(3.0-np.sqrt(5.0))
    R      = 1.0-C

    a,b,c  = bracket_min(cCST)
    
    x0     = a
    x3     = c
    if (np.abs(c-b) > np.abs(b-a)):
        x1 = b
        x2 = b+C*(c-b)
    else:
        x2 = b
        x1 = b-C*(b-a)
    f1 = cCST.eval(x1)
    f2 = cCST.eval(x2)
    while (np.abs(x3-x0) > tol*(np.abs(x1)+np.abs(x2))):
        if (f2 < f1):
            x0 = x1
            x1 = x2
            x2 = R*x1+C*x3
            f1 = f2
            f2 = cCST.eval(x2)
        else:
            x3 = x2
            x2 = x1
            x1 = R*x2+C*x0
            f2 = f1
            f1 = cCST.eval(x1)
        #print('golden: ',x0,x1,x2,x3,f1,f2,np.abs(x3-x0),tol*(np.abs(x1)+np.abs(x2))
    if (f1 < f2):
        xmin = x1
    else:
        xmin = x2
    return xmin

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
    elif (s_cost == 'lunarlander'):
        bounds = np.array([[0.0,20.0]])
        maxmin = 1
        tol    = 1e-5
        fCST   = fnc_cost.lunarlander
    else:
        raise Exception('[init]: invalid s_cost %s' % (s_cost))

    return cost.Cost(fCST,bounds,tol,maxmin)

#=============================================
# performs basic checks (plots cost function and estimated minimum)
def check(xopt,cCST):
  
    fname = cCST.evalname()
    if (fname == 'parabola1d'):
        print('[check]: minimum of parabola is at %13.5e' % (xopt))
        n         = 101
        minx      = -1.0
        maxx      = 3.0
        x         = minx+(maxx-minx)*np.arange(n)/float(n-1)
        cst       = np.zeros(n)
        for i in range(n):
            cst[i] = cCST.eval(x[i])
        minc      = np.min(cst)
        maxc      = np.max(cst)
    elif (cCST.evalname() == 'lunarlander'):
        print('[check]: optimal t_on = %13.5e s' % (xopt))
        n         = 41
        minx      = 5.0
        maxx      = 20.0
        dummy     = cCST.eval(xopt,plot=1)

    x         = minx+(maxx-minx)*np.arange(n)/float(n-1)
    cst       = np.zeros(n)
    for i in range(n):
        cst[i] = cCST.eval(x[i])
    minc      = np.min(cst)
    maxc      = np.max(cst)
    ftsz      = 10
    plt.figure(num=1,figsize=(8,8),dpi=100,facecolor='white')
    plt.subplot(111)
    plt.plot(x,cst,linestyle='-',color='black',linewidth=1.0)
    plt.plot(np.array([xopt,xopt]),np.array([minc,maxc]),linestyle='-',color='red',linewidth=1.0)
    plt.xlabel('x',fontsize=ftsz)
    plt.ylabel('cost function',fontsize=ftsz)
    util.rescaleplot(x,cst,plt,0.05)
    plt.tick_params(labelsize=ftsz)

    plt.show()

#========================================================
def main():

    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument("s_cost",type=str,
                        help="function to be minimized:\n"
                             "   parabola1d\n"
                             "   lunarlander")
    parser.add_argument("--verbose", help="print and show diagnostic information",action="store_true")

    args       = parser.parse_args()
    s_cost     = args.s_cost
    if args.verbose:
        iverb = 1
    else:
        iverb = 0

    cCST       = init(s_cost)

    best       = golden(cCST,iverb=iverb)

    check(best,cCST)

#=======================================
main()

