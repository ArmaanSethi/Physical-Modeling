#============================================
# program: Library for fitting functions with
# linear coefficients (linfit). If the 
# functions are non-linear, use glinfit.
#============================================
import numpy as np
import scipy.special
#===================================
# function: linfit
# purpose : fits a straight line with
#           parameters a+b*x to data set.
# input   : x   : float vector of length n: "independent" variable (assumed to have no uncertainties)
#           y   : float vector of length n: data points
#           sig : float vector of length n: measurement uncertainties for data points y.
# output  : a    : float number: fit parameter (here: offset)
#           b    : float number: fit parameter (here: slope)
#           siga : float number: uncertainty of a
#           sigb : float number: uncertainty of b
#           chi2 : floating point number. Should be around n-2, i.e. 
#                  the number of data points less the number of parameters ("degrees of freedom")
#           q    : quality of fit estimate (should be between ~0.1 and 10^(-3))
#===================================
def linfit(x,y,sig):
      # from here ????
      n     = x.size
      weight= 1.0/sig**2
      s     = np.sum(weight)
      sx    = np.sum(x*weight)
      sy    = np.sum(y*weight)
      sxx   = np.sum(x**2*weight)
      sxy   = np.sum(x*y*weight)
      delta = s*sxx-sx**2
      a     = (sxx*sy-sxy*sx)/delta
      b     = (s*sxy-sx*sy)/delta
      siga  = np.sqrt(sxx/delta)
      sigb  = np.sqrt(s/delta)
      chi2  = np.sum(((y-a-b*x)/sig)**2)
      if (n > 2):
            q = scipy.special.gammainc(0.5*chi2,0.5*float(n-2))

      return np.array([a,b]),np.array([siga,sigb]),chi2,q
      # to here ?????

#===================================
# function: glinfit
# purpose : fits a set of basis functions with linear 
#           coefficients to a data set.
#           Measurement uncertainties need to be provided.
# input   : x   : float vector of length n: "independent" variable (assumed to have no uncertainties)
#           y   : float vector of length n: data points
#           sig : float vector of length n: measurement uncertainties for data points y.
#           m   : integer: number of parameters
#           fMOD: function pointer vector of length m. (f(x))[j] must return
#                the j'th basis function, j=0...m-1. For example, if
#                we have a function basis [1,x,x^2] (parabolic fit),
#                then (f(x0))[0] = 1, (f(x0))[1] = x0, (f(x0))[2] = x0^2
# output  : a    : float vector of length n: fit parameters (here: offset and slope)
#           siga : float vector of length n: uncertainties of a ("errors")
#           chi2 : floating point number. Should be around n-m, i.e. 
#                  the number of data points less the number of parameters ("degrees of freedom")
#           q    : quality of fit estimate (should be between ~0.1 and 10^(-3))
#===================================
def glinfit(x,y,sig,m,fMOD):
    # from here ?????
    n     = x.size
    a     = np.zeros(m)
    siga  = np.zeros(m)
    b     = y/sig
    A     = np.zeros((n,m))
    for i in range(n):
        A[i,:] = (fMOD(x[i]))[:]/sig[i]
    C     = np.linalg.inv(np.dot(np.transpose(A),A))
    a     = np.dot(C,np.dot(np.transpose(A),b))
    siga  = np.diagonal(C)
    jsum  = np.zeros(n)
    for i in range(n):
        jsum[i] = np.sum((fMOD(x[i]))[:]*a[:])
    chi2  = np.sum(((y-jsum)/sig)**2)
    if (n > m):
        q = scipy.special.gammainc(0.5*chi2,0.5*float(n-m))
    return a,siga,chi2,q
    # to here ?????

