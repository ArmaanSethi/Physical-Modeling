#============================================
# program: test_bayes.py
# purpose: testing Bayes fitting following D. Hogg's primer
#============================================
import argparse                  # allows us to deal with arguments to main()
from argparse import RawTextHelpFormatter
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import scipy.special
import p358utilities as util

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

# ????? from here
def six_a(x,y,sy):
    N = np.size(x)
    M = 2

    a = np.ones((1,len(x)))
    A = np.vstack((a,x)).T
    Y = np.reshape(y,(N,1))
    C = np.diag(sy*sy)

    X = np.linalg.inv(A.T@np.linalg.inv(C)@A) @ (A.T@ np.linalg.inv(C)@Y)
    chi2 = (Y-A@X).T @ np.linalg.inv(C) @ (Y-A@X)
    cov = np.linalg.inv(A.T@ np.linalg.inv(C)@A)

    print("X:\n",X)
    print("b:\n",X[0])
    print("m:\n",X[1])
    print("chi2:\n",chi2)
    print("cov:\n", cov)
    print("sigb:\n", np.sqrt(cov[0][0]))
    print("sigm:\n", np.sqrt(cov[1][1]))
    q = scipy.special.gammainc(0.5*chi2,0.5*float(N-M))
    print("q:\n",q)

    plt.figure(num=1,figsize=(8,8),dpi=100,facecolor='white')
    plt.grid()
    if(N == 20):
        plt.title("6a")
    elif(N == 16):
        plt.title("6b")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.scatter(x,y,color='black', label="real data")
    plt.errorbar(x,y,xerr=0,yerr=sy, fmt='none', color='black')
    plt.plot(x, X[1]*x+X[0], color='green', label="Linear Fit")
    plt.legend()

    
    plt.show()


def six_b(x,y,sy):
    x = x[4:]
    y = y[4:]
    sy = sy[4:]
    six_a(x,y,sy)

def rnorm(R):
    # from here ????
    def gaussian(u0,u1):
        z0 = np.sqrt(-2*np.log(u0))*np.cos(2*np.pi*u1)
        z1 = np.sqrt(-2*np.log(u0))*np.sin(2*np.pi*u1)
        return z0,z1

    u0 = np.random.rand(R)
    u1 = np.random.rand(R)

    z0,z1 = gaussian(u0,u1)

    mu = 0
    sigma = 1
    # to here ????
    return z1*sigma + mu

def quality_plot(arr, xlabel, ylabel, title):
    plt.figure(num=1,figsize=(2,8),dpi=100,facecolor='white')
    plt.grid()
    plt.plot(arr, np.arange(len(arr)), linestyle='-',color='black',linewidth=1.0)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tick_params(labelsize=10)
    plt.show()

def six_c(x,y,sy, Pb0, T):
    N = np.size(x)
    M = 2

    a = np.ones((1,len(x)))
    A = np.vstack((a,x)).T
    Y = np.reshape(y,(N,1))
    C = np.diag(sy*sy)

    X = np.linalg.inv(A.T@np.linalg.inv(C)@A) @ (A.T@ np.linalg.inv(C)@Y)
    chi2 = (Y-A@X).T @ np.linalg.inv(C) @ (Y-A@X)
    cov = np.linalg.inv(A.T@ np.linalg.inv(C)@A)
    print("X:\n",X)
    print("chi2:\n",chi2)
    print("cov:\n", cov)
    q = scipy.special.gammainc(0.5*chi2,0.5*float(N-M))
    print("q:\n",q)
    print("sigb:\n", np.sqrt(cov[0][0]))
    print("sigm:\n", np.sqrt(cov[1][1]))

    def obj_func_c(m,b):
        N = np.size(x)
        fancy_L = 1.0
        # print("i \t\t p(y|xi,syi,m,b) \t\t L")
        for i in range(N):
            p_y_given = np.power(2*np.pi*sy[i]**2, -0.5) * np.exp(-1*(y[i] - m*x[i]- b)**2 / (2*sy[i]**2))
            fancy_L *= p_y_given
            # print(i,'\t\t',p_y_given,'\t\t',fancy_L)
        return fancy_L


    L = obj_func_c(X[1], X[0])

    def met_hast(R,m0,b0,delta_m, delta_b):
        t = 0
        # xr = []
        # xr.append(x0)
        mr = []
        br = []
        mr.append(m0)
        br.append(b0)

        while len(mr) < R:
            # x_cand = rnorm(1)*delta + xr[t]
            m_cand = rnorm(1)*delta_m + mr[t]
            b_cand = rnorm(1)*delta_b + br[t]
            # print(m_cand,b_cand)
            # a = fTAR(x_cand, bounds)/fTAR(xr[t], bounds) #acceptance ratio
            a = obj_func_c(m_cand, b_cand) / obj_func_c(mr[t], br[t])
            u = np.random.rand() #u from normal dist
            if u < a or a >= 1.0:
                # xr.append(x_cand)
                mr.append(m_cand)
                br.append(b_cand)
            else:  
                # xr.append(xr[t])
                mr.append(mr[t])
                br.append(br[t])
            t+=1
        # xr = np.array(xr)
        xm = np.array(mr)
        xb = np.array(br)
    

        return (xm[200:],xb[200:])

    print("L:\n",L)
    (m_hast, b_hast) = met_hast(T, m0 = 1.0, b0 = 200, delta_m = 0.1, delta_b=1)

    """M HISTOGRAM"""
    (m_n, m_bins, patches) = plt.hist(m_hast, 'sqrt', color='green')
    plt.xlabel('m')
    plt.ylabel('#')
    plt.title('6c - Histogram of m')
    max_value = max(m_n)
    max_index = list(m_n).index(max_value)
    mode_m = (m_bins[max_index]+m_bins[max_index])/2
    print("<m>:\n",mode_m)
    plt.show()

    """B HISTOGRAM"""
    (b_n, b_bins, b_patches) = plt.hist(b_hast, 'sqrt', color='green')
    plt.xlabel('b')
    plt.ylabel('#')
    plt.title('6c - Histogram of b')
    max_value = max(b_n)
    max_index = list(b_n).index(max_value)
    mode_b = (b_bins[max_index]+b_bins[max_index])/2
    print("<b>:\n",mode_b)
    plt.show()

    """2D HISTOGRAM"""
    (h, xedges, yedges, image) = plt.hist2d(b_hast, m_hast, [b_bins, m_bins])
    plt.xlabel('b')
    plt.ylabel('m')
    plt.title('6c - 2d Histogram ')
    plt.show()

    """Line of best fits"""
    plt.figure(num=1,figsize=(8,8),dpi=100,facecolor='white')
    plt.grid()
    plt.title("6c - Line of best fits")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.scatter(x,y,color='black', label="Real Data")
    plt.errorbar(x,y,xerr=0,yerr=sy, fmt='none', color='black')
    plt.plot(x, X[1]*x+X[0], color='green', label="Linear Fit")
    plt.plot(x, mode_m*x+mode_b, color='orange', label="Histogram Linear Fit")
    plt.legend()
    plt.show()

    """Quality Assurance Plot b"""
    quality_plot(b_hast, title="6c - Quality Assurance Plot b", xlabel="b", ylabel="t")
    """Quality Assurance Plot m"""
    quality_plot(m_hast, title="6c - Quality Assurance Plot m", xlabel="m", ylabel="t")

def six_d(x,y,sy, Pb0, T):
    N = np.size(x)
    M = 2

    a = np.ones((1,len(x)))
    A = np.vstack((a,x)).T
    Y = np.reshape(y,(N,1))
    C = np.diag(sy*sy)

    X = np.linalg.inv(A.T@np.linalg.inv(C)@A) @ (A.T@ np.linalg.inv(C)@Y)
    chi2 = (Y-A@X).T @ np.linalg.inv(C) @ (Y-A@X)
    cov = np.linalg.inv(A.T@ np.linalg.inv(C)@A)
    print("X:\n",X)
    print("chi2:\n",chi2)
    print("cov:\n", cov)
    q = scipy.special.gammainc(0.5*chi2,0.5*float(N-M))
    print("q:\n",q)
    print("sigb:\n", np.sqrt(cov[0][0]))
    print("sigm:\n", np.sqrt(cov[1][1]))

    def obj_func_d(m,b, Pb, Yb, Vb):
        N = np.size(x)
        fancy_L = 1.0
        # print("i \t\t p(y|xi,syi,m,b) \t\t L")
        for i in range(N):
            p_y_given = (1-Pb)*np.power(2*np.pi*sy[i]**2, -0.5)*np.exp(-1*(y[i] - m*x[i]- b)**2 / (2*sy[i]**2)) + (Pb)*np.power(2*np.pi*(sy[i]**2+Vb), -0.5)*np.exp(-1*(y[i]-Yb)**2/(2*(Vb+sy[i]**2)))

            fancy_L *= p_y_given
            # print(i,'\t\t',p_y_given,'\t\t',fancy_L)
        return fancy_L

    def met_hast(R,m0,b0, pb0, yb0, vb0, delta_m, delta_b, delta_pb, delta_yb, delta_vb):
        t = 0
        mr = []
        br = []
        pbr = []
        ybr = []
        vbr = []

        mr.append(m0)
        br.append(b0)
        pbr.append(pb0)
        ybr.append(yb0)
        vbr.append(vb0)
        def p_theta(pb, vb):
            if(pb>1):
                return 0
            if (pb<0):
                return 0
            if(vb < 0):
                return 0
            return 1
        while len(mr) < R:

            m_cand = rnorm(1)*delta_m + mr[t]
            b_cand = rnorm(1)*delta_b + br[t]
            pb_cand = rnorm(1)*delta_pb + pbr[t]
            yb_cand = rnorm(1)*delta_yb + ybr[t]
            vb_cand = rnorm(1)*delta_vb + vbr[t]

            a = p_theta(pb_cand, vb_cand)*obj_func_d(m_cand, b_cand, pb_cand, yb_cand, vb_cand) / (p_theta(pbr[t], vbr[t])*obj_func_d(mr[t], br[t], pbr[t], ybr[t], vbr[t]))

            u = np.random.rand() #u from normal dist
            if u < a or a >= 1.0:
                mr.append(m_cand)
                br.append(b_cand)
                pbr.append(pb_cand)
                ybr.append(yb_cand)
                vbr.append(vb_cand)
            else:  
                mr.append(mr[t])
                br.append(br[t])
                pbr.append(pbr[t])
                ybr.append(ybr[t])
                vbr.append(vbr[t])

            t+=1

        xm = np.array(mr)
        xb = np.array(br)
        xpb = np.array(pbr)
        xyb = np.array(ybr)
        xvb = np.array(vbr)

        return (xm[200:],xb[200:],xpb[200:],xyb[200:],xvb[200:])

    # print(obj_func_d(2,30,0.2,0,0))

    (m_hast, b_hast, pb_hast, yb_hast, vb_hast) = met_hast(R=T, m0 = 2.0, b0 = 30, pb0=Pb0, yb0=np.mean(y), vb0=np.mean(y**2), 
        delta_m = 0.1, delta_b=1, delta_pb=.01, delta_yb=0.1, delta_vb=0.1)



    """M HISTOGRAM"""
    (m_n, m_bins, patches) = plt.hist(m_hast, 'sqrt', color='green')
    plt.xlabel('m')
    plt.ylabel('#')
    plt.title('6d - Histogram of m')
    max_value = max(m_n)
    max_index = list(m_n).index(max_value)
    mode_m = (m_bins[max_index]+m_bins[max_index])/2
    print("<m>:\n",mode_m)
    plt.show()

    """B HISTOGRAM"""
    (b_n, b_bins, b_patches) = plt.hist(b_hast, 'sqrt', color='green')
    plt.xlabel('b')
    plt.ylabel('#')
    plt.title('6d - Histogram of b')
    max_value = max(b_n)
    max_index = list(b_n).index(max_value)
    mode_b = (b_bins[max_index]+b_bins[max_index])/2
    print("<b>:\n",mode_b)
    plt.show()

    """2D HISTOGRAM"""
    (h, xedges, yedges, image) = plt.hist2d(b_hast, m_hast, [b_bins, m_bins])
    plt.xlabel('b')
    plt.ylabel('m')
    plt.title('6d - 2d Histogram ')
    plt.show()

    """Line of best fits"""
    plt.figure(num=1,figsize=(8,8),dpi=100,facecolor='white')
    plt.grid()
    plt.title("6d - Line of best fits")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.scatter(x,y,color='black', label="Real Data")
    plt.errorbar(x,y,xerr=0,yerr=sy, fmt='none', color='black')
    plt.plot(x, X[1]*x+X[0], color='green', label="Linear Fit")
    plt.plot(x, mode_m*x+mode_b, color='orange', label="Histogram Linear Fit")
    plt.legend()
    plt.show()

    """Quality Assurance Plot b"""
    quality_plot(b_hast, title="6d - Quality Assurance Plot b", xlabel="b", ylabel="t")
    """Quality Assurance Plot m"""
    quality_plot(m_hast, title="6d - Quality Assurance Plot m", xlabel="m", ylabel="t")
    """Quality Assurance Plot pb"""
    quality_plot(pb_hast, title="6d - Quality Assurance Plot pb", xlabel="pb", ylabel="t")
    """Quality Assurance Plot yb"""
    quality_plot(yb_hast, title="6d - Quality Assurance Plot yb", xlabel="yb", ylabel="t")
    """Quality Assurance Plot vb"""
    quality_plot(vb_hast, title="6d - Quality Assurance Plot vb", xlabel="vb", ylabel="t")


# ?????? to here

#============================================
def main():

    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument("Pbad",type=float,
                        help="probability for bad data points:\n"
                             "   0<=Pbad<1. First run with Pbad=0\n"
                             "   (no pruning of bad data points)")
    parser.add_argument("T",type=int,
                        help="length of Markov chain (depends on problem)")

    args       = parser.parse_args()
    Pb0        = args.Pbad
    T          = args.T

    x,y,sy       = readdata('hogg.txt')

# ???? from here
    # print("\n\n\n######### PART 6A #########")
    # six_a(x,y,sy)

    # print("\n\n\n######### PART 6B #########")
    # six_b(x,y,sy)

    # print("\n\n\n######### PART 6C #########")
    # six_c(x,y,sy,Pb0,T)

    # print("\n\n\n######### PART 6D #########")
    # six_d(x,y,sy,Pb0,T)

    # def kn(z,n):
    #     return (4*n+3-z*z)**(0.5)

    def kn(z,n):
        return np.sqrt(15)

    def kapn(z,n):
        return (z*z-(4*n+3))**(0.5)
    
    def sn(z):
        return 0.5*(z*kapn(z,3)-(np.sqrt(15)* np.log((z+kapn(z,3))/np.sqrt(15))) )

    def phi(z):
        return .5*(np.arccos(z/np.sqrt(15)) -z*kn(z,3))
    # def phi(z):
    #     return z
    def psiL(z):
        return 2*(kn(z,3))**(-0.5) * np.sin(phi(z)-np.pi/4)
    def psiR(z):
        return 2*(kapn(z,3))**(-0.5) *np.exp(-sn(z))
    print(np.sqrt(15))
    print(kapn(0,3))
    print(psiL(0))

    z_val = np.arange(0,1+np.sqrt(15),0.01)
    z_val2 = np.arange(np.sqrt(15), 3.9, 0.01)
    plt.plot(z_val, psiL(z_val))
    plt.plot(z_val2, psiR(z_val2))
    plt.ylim((-3,3))
    plt.title("Ψ(ζ) vs ζ")
    plt.xlabel("ζ")
    plt.ylabel("Ψ(ζ)")
    plt.grid()
    plt.show()

# ????? to here

#========================================

main()

