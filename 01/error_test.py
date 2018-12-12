#==============================================================
# Calculates the integration errors on y' = exp(y-2x)+2
#
#==============================================================
# required libraries
import numpy as np               # numerical routines (arrays, math functions etc)
import matplotlib.pyplot as plt  # plotting commands
import p358utilities as util     # for rescaleplot

import ode_integrators as odeint # contains the drivers for ODE integration.
import ode_step as step          # contains the ODE single step functions.

#==============================================================
# function dydx = get_dydx(x,y,dx)
#
# Calculates RHS for error test function
#
# input: 
#   x,y    : 
#
# global:
#   -
# output:
#   dydx    : vector of results as in y'=f(x,y)
#--------------------------------------------------------------

def get_dydx(x, y, dx):
    dydx = np.exp(y[0] - 2.0 * x) + 2.0
    return dydx

#==============================================================
# main
#==============================================================
def main():

#   the function should accomplish the following:
#   Test the 3 fixed stepsize integrators euler,rk2,rk4 by calculating their 
#   cumulative integration errors for increasing step sizes
#   on the function y' = exp(y(x)-2*x)+2. This is given in the function "get_dydx" above.
#   Use the integration interval x=[0,1], and the initial 
#   condition y[0] = -ln(2).
#   (1) define an array containing the number of steps you want to test. Logarithmic spacing
#       (in decades) might be useful.
#   (2) loop through the integrators and the step numbers, calculate the 
#       integral and store the error. You'll need the analytical solution at x=1,       
#       see homework assignment.
#   (3) Plot the errors against the stepsize as log-log plot, and print out the slopes.

    fINT  = odeint.ode_ivp                  # use the initial-value problem driver
    fORD  = [step.euler,step.rk2,step.rk4]  # list of stepper functions to be run
    fRHS  = get_dydx                        # the RHS (derivative) for our test ODE
    fBVP  = 0                               # unused for this problem

    #????????????????? from here

    N = np.array([10, 100, 1000, 10000, 100000])

    #########################PLOT VARIABLES################################
    fig  = plt.figure(num=1,figsize=(18,5),dpi=100,facecolor='white')
    fig.subplots_adjust(hspace=2)

    ax   = fig.add_subplot(131)
    color = ['green','blue','red']
    labels = []
    plot_number = 1
    #########################################################################
    for stepper in fORD:
        errors = []
        for nstep in N:
            x0 = 0
            y0 = -np.log(2)
            x1 = 1

            x,y,it = fINT(fRHS,stepper,fBVP,x0,y0,x1,nstep)

            err = 2-y[0][-1]
            errors.append(abs(err))
            labels.append(stepper.__name__)

            # just for formatting
            if(nstep < 100000):
                print(stepper.__name__,'\t\t', nstep,'\t\t',"Error: " , err )
            else:
                print(stepper.__name__,'\t\t', nstep,'\t',"Error: " , err )

        # print(stepper.__name__, "Error List", errors)
        #PLOT
        if(plot_number != 1):
            plt.subplot(1,3,plot_number)
        plt.loglog(N,errors,linestyle='-' ,color=color[plot_number-1],linewidth=1.0)
        plt.xlabel("Iterations")
        plt.ylabel("Cumulative Error")
        plt.title(stepper.__name__)
        plot_number+=1

        #Slope
        print("Slope for ", stepper.__name__, ": ", ( np.log10(errors[-1])-np.log10(errors[0]) ) / ( np.log10(N[-1]) - np.log10(N[0]) ) )
        if(stepper.__name__ == "rk4"):
            print("Slope for first 3 points of", stepper.__name__, ": ", ( np.log10(errors[-3])-np.log10(errors[0]) ) / ( np.log10(N[-3]) - np.log10(N[0]) ) )

            

    plt.show()

    #????????????????? to here


#==============================================================

main()


