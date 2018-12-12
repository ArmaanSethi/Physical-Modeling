#!/usr/bin/python
import numpy as np
#=====================================
# Declaration of class Cost
#=====================================

class Cost:

    #++++++++ constructor
    def __init__(self,fCST,bounds,tol,maxmin):
        if ((bounds.shape)[1] != 2):
            raise Exception("[class Cost]: bounds needs to be of shape [[lo1,up1],[lo2,up2],...]")
        # Variables with underscores should be treated as private, i.e. not accessible 
        # from outside the object. 
        # Note that Python does not have private variables, though.
        self._fcost  = fCST   # function pointer to cost function
        self._bounds = bounds # boundaries on which to evaluate cost function
                              # must be of form np.array([[lo1,up1],[lo2,up2]...[lon,upn]])
        self._tol    = tol    # tolerance for difference between population members (see self.err)
        self._maxmin = maxmin # -1: do minimization (eval = 1.0/fCST), 1: do maximization (eval = fCST)

    #++++++++ method evalname()
    # returns function name of actual cost function
    def evalname(self):
        return self._fcost.__name__

    #++++++++ method eval(x)
    # returns the value of the cost function at position x
    def eval(self,x,**kwargs):
        for key in kwargs:
            if (key == 'function'):
                if (kwargs[key]):
                    return self._fcost(x,**kwargs)
        if (self._maxmin > 0):
            return self._fcost(x,**kwargs)
        else:
            return 1.0/self._fcost(x,**kwargs)

    #++++++++ method err(x)
    # returns the rms difference between two normalized population members
    def err(self,x1,x2):
        u1 = self.normalize(x1)
        u2 = self.normalize(x2) 
        if (isinstance(x1,np.ndarray)):
            return np.sqrt(np.sum((u1-u2)**2))
        else:
            return np.abs(u1-u2)

    # returns the number of dimensions, i.e. the length of vector x
    def ndim(self):
        s = (self._bounds).shape
        return s[0]

    # returns the range of x[dim] 
    def range(self,dim):
        return (self._bounds)[:,dim]

    # returns the full range information
    def bounds(self):
        return self._bounds

    # returns the tolerance associated with cost function
    def maxmin(self):
        return self._maxmin

    # returns the tolerance associated with cost function
    def tol(self):
        return self._tol

    #++++++++ method normalize(x)
    # returns all x scaled between 0 and 1. x is a vector of length ndim
    # x is a vector of length ndim
    def normalize(self,x):
        return (x[:]-self._bounds[:,0])/(self._bounds[:,1]-self._bounds[:,0])

    #++++++++ method denormalize(u)
    # returns all u scaled between bounds[0] and bounds[1]. u is a vector of length ndim
    def denormalize(self,u):
        return u[:]*(self._bounds[:,1]-self._bounds[:,0])+self._bounds[:,0]
