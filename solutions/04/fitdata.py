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
    base = 'data' # elements to build file names
    suff = '.txt'
    ndat = 5
    fMOD = ftest

    # set up figure
    ftsz = 8
    fig  = plt.figure(num=1,figsize=(14,8),dpi=100,facecolor='white')

    for m in range(ndat):
        print('m=%1i'%(m))
        x,y,sig = readdata("%s%1i%s" % (base,m,suff))
        a_lf ,siga_lf ,chi2_lf ,q_lf  = linfit.linfit(x,y,sig)
        a_glf,siga_glf,chi2_glf,q_glf = linfit.glinfit(x,y,sig,3,fMOD)
        y_lf    = a_lf[0]+a_lf[1]*x
        y_glf   = np.zeros(x.size)
        for j in range(a_glf.size):
            y_glf[:] = y_glf[:] + a_glf[j]*(fMOD(x[:]))[j] 
        ax = fig.add_subplot(2,ndat,m+1)
        ax.errorbar(x,y,yerr=sig,fmt='o',linewidth=1)
        plt.title('%s%i' % (base,m))
        ax.plot(x,y_lf,linewidth=1,linestyle='-')
        for j in range(a_lf.size):
            ax.annotate('a[%1i]=%10.2e+-%10.2e' % (j,a_lf[j],siga_lf[j]),xy=(0,0),xytext=(0.13+0.16*m,0.85-float(j)*0.03),
                        fontsize=ftsz,textcoords='figure fraction')
        ax.annotate('$\chi^2$=%10.2e\nq=%10.2e' % (chi2_lf,q_lf),xy=(0,0),xytext=(0.19+0.16*m,0.60),
                    fontsize=ftsz,textcoords='figure fraction')

        ax = fig.add_subplot(2,ndat,ndat+m+1)
        ax.errorbar(x,y,yerr=sig,fmt='o',linewidth=1)
        plt.title('%s%i' % (base,m))
        ax.plot(x,y_glf,linewidth=1,linestyle='-')
        for j in range(a_glf.size):
            ax.annotate('a[%1i]=%10.2e+-%10.2e' % (j,a_glf[j],siga_glf[j]),xy=(0,0),xytext=(0.13+0.16*m,0.42-float(j)*0.03),
                        fontsize=ftsz,textcoords='figure fraction')
        ax.annotate('$\chi^2$=%10.2e\nq=%10.2e' % (chi2_glf,q_glf),xy=(0,0),xytext=(0.19+0.16*m,0.15),
                    fontsize=ftsz,textcoords='figure fraction')

    plt.show()
 
    # to here ????

#========================================

main()

