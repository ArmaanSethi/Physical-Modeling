#============================================
# program: fitdata.py
# purpose: calls linfit and glinfit, on a
#  set of data provided by user. Prints out
#  parameters, uncertainties, chi^2, and Q.
#============================================
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import linfit

#============================================
# reads table of measured data. Expects
# three columns with (x, y, sig)
# Returns arrays x,y,sig.
def readdata(name):
    f   = open(name)
    lst = []
    for line in f:
        lst.append([float(d) for d in line.split()])
    x   = np.array([d[0] for d in lst])
    y   = np.array([d[1] for d in lst])
    sig = np.array([d[2] for d in lst])
    return x,y,sig

#============================================
# model function
# Must return a vector of the values of the 
# basis functions at position x. The sum over the vector should
# be the value of the model function at position x.
# (Think expansion in terms of
# basis functions), i.e. ymod = np.sum((ftest(x))
def ftest(x):
    if (isinstance(x,np.ndarray)):
        n = x.size
        return np.array([[np.zeros(n)+1.0],[x],[np.sin(x)]])
    else:
        return np.array([1.0,x,np.sin(x)])

#============================================
# main() should read all 5 data sets, perform
# the fits using linfit and glinfit,  plot the 
# results, and print out the fit parameters and
# their uncertainties.
# The routines linfit and glinfit can be addressed via 
# linfit.linfit(...) and linfit.glinfit(...)
def main():

    # from here ????
    def fMOD(x):
        return np.array([1, x, np.sin(x)])

    # def Q4hw(x):
    #     return ((1**2-x**2)*(1-x**2))
    # x = np.arange(-1,1,0.01)
    # plt.figure(num=1,figsize=(8,8),dpi=100,facecolor='white')
    # plt.grid()
    # plt.title("P521 HW 4")
    # plt.ylabel("φ(x)")
    # plt.xlabel("x (in terms of a)")
    # plt.plot(x,Q4hw(x))
    # plt.show()
    #sorry about the code being this bad...it's been a rough week
    print("\n\n\n###DATA0###")
    """LINFIT DATA0"""
    x,y,sig = readdata("data0.txt")
    a, b, siga, sigb, chi2, q = linfit.linfit(x,y,sig)
    plt.figure(num=1,figsize=(8,8),dpi=100,facecolor='white')
    plt.grid()
    plt.title("Data0, linfit")
    plt.plot(x,y, label="Data")
    plt.plot(x, b*x+a, label="Linear Regression")
    plt.show()

    """GEN LINFIT DATA0"""
    x,y,sig = readdata("data0.txt")
    a, siga, chi2, q = linfit.glinfit(x,y,sig,3,fMOD)
    plt.figure(num=1,figsize=(8,8),dpi=100,facecolor='white')
    plt.title("Data0, glinfit")
    plt.grid()
    plt.plot(x,y)
    plt.plot(x, a[0]+a[1]*x+a[2]*np.sin(x))
    plt.show()


    print("\n\n\n###DATA1###")
    """LINFIT DATA1"""
    x,y,sig = readdata("data1.txt")
    a, b, siga, sigb, chi2, q = linfit.linfit(x,y,sig)
    plt.figure(num=1,figsize=(8,8),dpi=100,facecolor='white')
    plt.grid()
    plt.title("Data1, linfit")

    plt.plot(x,y)
    plt.plot(x, b*x+a)
    plt.show()

    """GEN LINFIT DATA1"""
    x,y,sig = readdata("data1.txt")
    a, siga, chi2, q = linfit.glinfit(x,y,sig,3,fMOD)
    plt.figure(num=1,figsize=(8,8),dpi=100,facecolor='white')
    plt.grid()
    plt.title("Data1, glinfit")
    plt.plot(x,y)
    plt.plot(x, a[0]+a[1]*x+a[2]*np.sin(x))
    plt.show()


    print("\n\n\n###DATA2###")
    """LINFIT DATA2"""
    x,y,sig = readdata("data2.txt")
    a, b, siga, sigb, chi2, q = linfit.linfit(x,y,sig)
    plt.figure(num=1,figsize=(8,8),dpi=100,facecolor='white')
    plt.grid()
    plt.title("Data2, linfit")
    plt.plot(x,y)
    plt.plot(x, b*x+a)
    plt.show()

    """GEN LINFIT DATA2"""
    x,y,sig = readdata("data2.txt")
    a, siga, chi2, q = linfit.glinfit(x,y,sig,3,fMOD)
    plt.figure(num=1,figsize=(8,8),dpi=100,facecolor='white')
    plt.grid()
    plt.title("Data2, glinfit")
    plt.plot(x,y)
    plt.plot(x, a[0]+a[1]*x+a[2]*np.sin(x))
    plt.show()


    print("\n\n\n###DATA3###")
    """LINFIT DATA3"""
    x,y,sig = readdata("data3.txt")
    a, b, siga, sigb, chi2, q = linfit.linfit(x,y,sig)
    plt.figure(num=1,figsize=(8,8),dpi=100,facecolor='white')
    plt.grid()
    plt.title("Data3, linfit")
    plt.plot(x,y)
    plt.plot(x, b*x+a)
    plt.show()

    """GEN LINFIT DATA3"""
    x,y,sig = readdata("data3.txt")
    a, siga, chi2, q = linfit.glinfit(x,y,sig,3,fMOD)
    plt.figure(num=1,figsize=(8,8),dpi=100,facecolor='white')
    plt.grid()
    plt.title("Data3, glinfit")
    plt.plot(x,y)
    plt.plot(x, a[0]+a[1]*x+a[2]*np.sin(x))
    plt.show()


    print("\n\n\n###DATA4###")
    """LINFIT DATA4"""
    x,y,sig = readdata("data4.txt")
    a, b, siga, sigb, chi2, q = linfit.linfit(x,y,sig)
    plt.figure(num=1,figsize=(8,8),dpi=100,facecolor='white')
    plt.grid()
    plt.title("Data4, linfit")
    plt.plot(x,y)
    plt.plot(x, b*x+a)
    plt.show()

    """GEN LINFIT DATA4"""
    x,y,sig = readdata("data4.txt")
    a, siga, chi2, q = linfit.glinfit(x,y,sig,3,fMOD)
    plt.figure(num=1,figsize=(8,8),dpi=100,facecolor='white')
    plt.grid()
    plt.title("Data4, glinfit")
    plt.plot(x,y)
    plt.plot(x, a[0]+a[1]*x+a[2]*np.sin(x))
    plt.show()
    # to here ????

#========================================

main()

