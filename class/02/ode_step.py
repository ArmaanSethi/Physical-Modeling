import numpy as np
#==============================================================
# function y = euler(fRHS,x0,y0,dx)
#
# Advances solution of ODE by one Euler step y = y0+f*dx, where f = y'
#
# input: 
#   fRHS   : function handle. Needs to return a vector of size(y0);
#   x0     : starting x
#   y0     : starting y(x0)
#   dx     : step size
# output:
#   y,1    : vector of results. For consistency with adaptive step size
#            integrators, we return an additional variable.
#--------------------------------------------------------------

def euler(fRHS,x0,y0,dx):
    y  = y0 + dx*fRHS(x0,y0,dx)
    return y,1


