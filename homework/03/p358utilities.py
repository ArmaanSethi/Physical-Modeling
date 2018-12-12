#!/usr/bin/python
import matplotlib.pyplot as plt
import numpy as np

# rescales plot window boundaries
def rescaleplot(x,y,p,fac):
    xmin    = np.min(x)
    xmax    = np.max(x)
    ymin    = np.min(y)
    ymax    = np.max(y)
    dx      = xmax-xmin
    dy      = ymax-ymin
    if (dy == 0.0):
        if (ymax == 0):
            dy = 1.0
        else:
            dy = 0.1*ymax
    minx    = xmin-fac*dx
    maxx    = xmax+fac*dx
    miny    = ymin-fac*dy
    maxy    = ymax+fac*dy
    xlim    = np.array([minx,maxx])
    ylim    = np.array([miny,maxy])
    p.xlim(xlim)
    p.ylim(ylim)

def histmean(x,h,**kwargs):
    inorm = False
    for key in kwargs:
        if (key=='normed'):
            inorm = kwargs[key]
    if (inorm):
        h1 = np.copy(h.astype(float))/np.sum(h.astype(float))
    else:
        h1 = np.copy(h.astype(float))
    avg = np.sum(x*h1)/np.sum(h1)
    std = np.sqrt(np.sum(np.power(x-avg,2)*h1)/np.sum(h1))
    
    return avg,std

