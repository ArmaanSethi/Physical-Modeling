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
    # print(x)
    # print(y)
    # print(sig)

    N = np.size(x)
    m = 2

    S     = np.sum(1 / np.square(sig))
    Sx    = np.sum(x / np.square(sig))
    Sy    = np.sum(y / np.square(sig))
    Sxx   = np.sum(np.square(x)/np.square(sig))
    Sxy   = np.sum(x*y / np.square(sig))

    omega = S*Sxx - np.square(Sx)

    a = (Sxx*Sy - Sxy*Sx)/omega
    b = (S*Sxy-Sx*Sy)/omega

    siga =  Sxx / omega
    sigb = S / omega
    chi2 = np.sum( np.square((y - (np.dot(x, b) + a)) / sig)  )

    q = scipy.special.gammainc(0.5*chi2 , 0.5*float(N-m))
    print("#####LINFIT")
    print("a: \t",a,"\n","b: \t", b, "\n", "siga:\t ", siga, "\n", "sigb:\t ", sigb, "\n", "chi2:\t ", chi2, "\n", "q:\t ", q)
    print("#####\n\n")

    return a, b, siga, sigb, chi2, q
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
    A = np.zeros((x.shape[0], 3))
    N = np.size(x)

    for i in np.arange(0, x.shape[0]): 
            A[i, :] = fMOD(x[i]) / sig[i]

    b  = y/sig
    a = np.linalg.inv(A.T @ A) @ A.T @ b

    C = np.linalg.inv(A.T@A)
    siga = np.diag(C)
    # a[0]+a[1]*x+a[2]*np.sin(x)
    chi2 = np.sum( np.square((y - (a[0]+a[1]*x+a[2]*np.sin(x) )) / sig)  )

    q = scipy.special.gammainc(0.5*chi2 , 0.5*float(N-m))
    print("#####General Linfit")
    print("a:\t ",a,"\n", "sig:\t ", sig, "\n", "chi2:\t ", chi2, "\n", "q:\t ", q)
    print("#####")
    return a, siga, chi2, q
    # to here ?????

