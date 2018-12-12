import matplotlib
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
    if (isinstance(p,matplotlib.axes.Axes)):
        p.set_xlim(xlim)
        p.set_ylim(ylim)
    else:
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

def histogram(x,n):
    hist,edges = np.histogram(x,n,normed=False)
    x          = 0.5*(edges[0:edges.size-1]+edges[1:edges.size])
    nbin       = x.size
    tothist    = np.sum(hist.astype(float))
    hist       = hist.astype(float)/tothist
    histx      = np.zeros((n,2))
    histx[:,0] = x
    histx[:,1] = hist
    return histx

def histogram2d(x,y,nx,ny):
    hist,xedges,yedges= np.histogram2d(x,y,[nx,ny],normed=False)
    xe                = 0.5*(xedges[0:xedges.size-1]+xedges[1:xedges.size])
    ye                = 0.5*(yedges[0:yedges.size-1]+yedges[1:yedges.size])
    nbinx             = xe.size
    nbiny             = ye.size
    tothist           = np.sum(hist.astype(float))
    hist              = hist.astype(float)/tothist
    return hist,xe,ye

def histogramdd(x,nx):
    hist,edges        = np.histogramdd(x,nx,normed=False)
    dim               = len(nx)
    xedges            = np.zeros((nx[0],dim))
    for d in range(dim):
        xedges[:,d]   = 0.5*((edges[d])[0:(edges[d]).size-1]+(edges[d])[1:(edges[d]).size])
    tothist           = np.sum(hist.astype(float))
    hist              = hist.astype(float)/tothist
    return hist,xedges

def decadesum(x):
    lx    = np.log10(x)
    minlx = int(np.floor(np.min(lx)))
    maxlx = int(np.ceil(np.max(lx)))
    lsum  = 0.0
    for i in range(minlx,maxlx):
        lsum = lsum + np.sum(x[np.where((float(i) <= lx) & (lx < float(i+1)))])    
    return lsum
    
    
    
    

