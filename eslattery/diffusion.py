"""
Created on Tue Jul 14 15:01:47 2015

@author: THIS IS NEW nesh, jonathan, Max
"""

import numpy as np
#import math
from numba import njit, float64, types#,guvectorize

prll_1d = False # 1d faster without parallelization
prll_2d = True  # 2d faster with parallelization




# @njit("f8[:](f8,f8[:],f8[:],i4)")
# def f0d(t, y, float_params, niter):
#     """ odeint function for the zero-dimensional ice model (only source terms)"""
#     Nbar, Nstar, sigmastepmax, sigma0, deprate = float_params  # unpack parameters
    
#     ## THIS LINE IS THE ERROR..
#     Fliq0, Ntot0 = y   # unpack current values of y

#     delta = (Fliq0 - (Nbar - Nstar))/(2*Nstar)
#     sigD = (sigmastepmax - delta * sigma0)/(1+delta*sigma0)
#     depsurf = deprate * sigD

#     ## just replaced getdNliq_dNtot() with actual formula from paper in terms of Ntot
#     dFliq0_dt = depsurf * Nstar * 2*np.pi*np.cos(2*np.pi*Ntot0)
#     dNtot_dt = depsurf
    
#     derivs = np.array([dFliq0_dt, dNtot_dt])
#     return derivs


## not working??
@njit("f8[:](f8,f8[:],f8[:],i4)")
def f0d(t, y, float_params, niter):
    """ odeint function for the zero-dimensional ice model (only source terms)"""
    Nbar, Nstar, sigmastepmax, sigma0, deprate = float_params  # unpack parameters
    
    Ntot0 = y[1]

    ## calc Fliq from Ntot
    Fliq0 = 1 + Nstar/Nbar * np.sin(2*np.pi*(Ntot0 - Nbar)) 

    delta = (Fliq0 - (Nbar - Nstar))/(2*Nstar)
    sigD = (sigmastepmax - delta * sigma0)/(1+delta*sigma0)
    depsurf = deprate * sigD

    ## replaced getdNliq_dNtot() with formula in terms of Ntot
    dNtot_dt = depsurf
    
    derivs = np.array([dNtot_dt])
    return derivs

## always worked, unchanged from diffusionstuff7
@njit("f8[:](f8[:],f8)",parallel=prll_1d)
def diffuse_1d(Fliq0, DoverdeltaX2):
    l = len(Fliq0)
    dy = np.zeros((l,))
    for i in range(1,-1):#(1,l-1):
        dy[i] = DoverdeltaX2*(Fliq0[i+1]-2*Fliq0[i]+Fliq0[i-1])
    ##dy[1:-1] = DoverdeltaX2 * (Fliq0[:-2] - 2 * Fliq0[1:-1] + Fliq0[2:])
    #boundary conditions
    dy[0] = DoverdeltaX2*(Fliq0[1]-2*Fliq0[0]+Fliq0[l-1])
    dy[l-1] = DoverdeltaX2*(Fliq0[0]-2*Fliq0[l-1]+Fliq0[l-2])
    return dy


@njit("f8[:](f8,f8[:],f8[:],f8[:])",parallel=prll_1d)#slower with paralellization right now
def f1d(t, Ntot0,  float_params, sigmastep): #sigmastep is an array
    """ odeint function for the one-dimensional ice model, calculates Fliq0 from Ntot
    
    Current version has implemented changes:
        Replaced calls to diffusionstuff7.getdNliq_dNtot_array() with calculations of Fliq0 from Ntot0
        Only takes Ntot0 values as an argument (rather than Fliq0 and Ntot0) 
        Only returns dNtot_dt values (rather than dFliq_dt amd dNtot_dt)"""
    
    # unpack parameters
    Nbar, Nstar, sigma0, deprate, DoverdeltaX2 = float_params 

    ## Ntot is passed in, Fqll calculated from Ntot
    Fliq0 = 1 + Nstar/Nbar * np.sin(2*np.pi*(Ntot0 - Nbar))

    ## WHY??? still unsure about this one....
    delta = (Fliq0 - (Nbar - Nstar))/(2*Nstar)
    sigD = (sigmastep - delta * sigma0)/(1+delta*sigma0)
    depsurf = deprate * sigD

    dNtot_dt = depsurf

    # Diffusion
    dy = diffuse_1d(Fliq0,DoverdeltaX2)
    dNtot_dt += dy 

    ## Package for output, only values of dNtot
    derivs = np.empty(len(Ntot0))
    derivs[:] = dNtot_dt
    return derivs




@njit(float64[:](float64,float64[:],float64,types.int64[:]), parallel=prll_2d)
def diffuse_2d(t,y,D,shape):
    """ Applies numerical solution to liquid to find diffusive effects. Fliq0 is flattened 2d array of shape shape.
    
    Parameters
    ----------
    Fliq0 : 2D Numpy Array 
        The thickness of the liquid over a 2D area

    t : float
        The time step-- unused- placeholder for odeint

    D : float64
        Diffusion coefficient -- divided by deltaX^2??? #TODO needs to be divided by 
                                            #TODO: cont.      x^2 or y^2 inside this function in order to have non-square discretization

    shape : tuple
        The shape of the 2D array Fliq0

    Returns
    -------
    dy : Flattened 2D Numpy Array
        The change to the thickness of the liquid at each point in the 2d area over
         the time step
    """
    m,n = shape
    Fliq0 = np.reshape(np.ascontiguousarray(y),(m,n))
    dy = np.zeros((m,n))

    # Calculate derivatives in the interior
    dy[1:-1, 1:-1] = (
        D * (Fliq0[:-2, 1:-1] - 2 * Fliq0[1:-1, 1:-1] + Fliq0[2:, 1:-1])
        + D * (Fliq0[1:-1, :-2] - 2 * Fliq0[1:-1, 1:-1] + Fliq0[1:-1, 2:])
    )

    # Handle periodic boundary conditions
    #Edges
    dy[0, 1:-1] = (
        D * (Fliq0[-1, 1:-1] - 2 * Fliq0[0, 1:-1] + Fliq0[1, 1:-1])
        + D * (Fliq0[0, :-2] - 2 * Fliq0[0, 1:-1] + Fliq0[0, 2:])
    )
    dy[-1, 1:-1] = (
        D * (Fliq0[-2, 1:-1] - 2 * Fliq0[-1, 1:-1] + Fliq0[0, 1:-1])
        + D * (Fliq0[-1, :-2] - 2 * Fliq0[-1, 1:-1] + Fliq0[-1, 2:])
    )
    dy[1:-1, 0] = (
    D * (Fliq0[:-2, 0] - 2 * Fliq0[1:-1, 0] + Fliq0[2:, 0])
    + D * (Fliq0[1:-1, -1] - 2 * Fliq0[1:-1, 0] + Fliq0[1:-1, 1])
    )
    dy[1:-1, -1] = (
        D * (Fliq0[:-2, -1] - 2 * Fliq0[1:-1, -1] + Fliq0[2:, -1])
        + D * (Fliq0[1:-1, -2] - 2 * Fliq0[1:-1, -1] + Fliq0[1:-1, 0])
    )
    #Corners
    dy[0, 0] = (
    D * (Fliq0[-1, 0] - 2 * Fliq0[0, 0] + Fliq0[1, 0])
    + D * (Fliq0[0, -1] - 2 * Fliq0[0, 0] + Fliq0[0, 1])
    )
    dy[-1, 0] = (
        D * (Fliq0[-2, 0] - 2 * Fliq0[-1, 0] + Fliq0[0, 0])
        + D * (Fliq0[-1, -1] - 2 * Fliq0[-1, 0] + Fliq0[-1, 1])
    )
    dy[0, -1] = (
        D * (Fliq0[-1, -1] - 2 * Fliq0[0, -1] + Fliq0[1, -1])
        + D * (Fliq0[0, -2] - 2 * Fliq0[0, -1] + Fliq0[0, 0])
    )
    dy[-1, -1] = (
        D * (Fliq0[-2, -1] - 2 * Fliq0[-1, -1] + Fliq0[0, -1])
        + D * (Fliq0[-1, -2] - 2 * Fliq0[-1, -1] + Fliq0[-1, 0])
    )

    return dy.flatten()

## getting errors (when used in solve_ivp)
@njit("f8[:](f8,f8[:],f8[:],i8[:],f8[:,:])",parallel=prll_2d) #NOTE: t and y swapped for solve_ivp compatability
def f2d(t, Ntot0, float_params, int_params, sigmastep):
    """ 2D version of f1d """

    # diffusion = True
    
    # unpack parameters
    Nbar, Nstar, sigma0, deprate, DoverdeltaX2 = float_params 
    nx, ny = int_params


    # unpack current values of y
    Fliq0 = 1 + Nstar/Nbar * np.sin(2*np.pi*(Ntot0 - Nbar))
    # Deposition
    delta = (Fliq0 - (Nbar - Nstar))/(2*Nstar)
    sigD = (sigmastep - delta * sigma0)/(1+delta*sigma0)
    depsurf = deprate * sigD

    ## 
    dNtot_dt = depsurf

    # if diffusion:
    # Diffusion
    dy =  np.reshape(np.ascontiguousarray(diffuse_2d(t, np.reshape(np.ascontiguousarray(Fliq0),nx*ny), DoverdeltaX2, np.array((nx,ny)))), (nx,ny))
    # Combined
    dNtot_dt += dy

    # Package for output
    derivs = dNtot_dt.flatten()    
    return derivs


@njit(float64[:](float64[:],float64,float64,float64))#,types.unicode_type))
def getsigmastep(x,xmax,center_reduction,sigmastepmax):#,method='parabolic'): 
    sigmapfac = 1-center_reduction/100 #float64
    xmid = max(x)/2 #float64
    #try:
        #if method == 'sinusoid':
        #    fsig = (np.cos(x/xmax*np.pi*2)+1)/2*(1-sigmapfac)+sigmapfac
        #elif method == 'parabolic':
    fsig = (x-xmid)**2/xmid**2*(1-sigmapfac)+sigmapfac
    #except:
    #    print('bad method')
    return fsig*sigmastepmax


#@njit(float64[:,:](float64[:],float64[:],float64,float64))
def getsigmastep_2d(xs,ys,center_reduction,sigmastepmax) -> np.ndarray: 
    c_r=center_reduction/100 #float64, convert percentage into decimal form
    # Getting the middle values of "x" and "y"
    xmax = np.max(xs)
    xmin = np.min(xs)
    xmid = (xmax-xmin)/2 +xmin 
    
    ymax = np.max(ys)
    ymin = np.min(ys)
    ymid = (ymax-ymin)/2 +ymin

    # Decide on an asymmetry factor
    asym = (xmax-xmin)/(ymax-ymin)
    # Calculate 2d parabolic coefficients
    C0 = sigmastepmax - c_r
    Cx = c_r/(xmax-xmid)**2
    Cy = c_r/(ymax-ymid)**2/asym

    if sigmastepmax < 0: # ablation case
        C0 = sigmastepmax + c_r
        Cx *= -1
        Cy *= -1      

    # Make a grid and evaluate supersaturation on it
    #xgrid,ygrid = np.meshgrid(xs-xmid,ys-ymid)
    ygrid,xgrid = np.meshgrid(ys-ymid,xs-xmid)
    #print(xgrid,ygrid)

    #Cy = 0.0 #TODO: temporary, see effect on sigmastep

    return C0 + xgrid**2*Cx + ygrid**2*Cy
    #return np.reshape(C0 + xgrid**2*Cx + ygrid**2*Cy,(xs.size,ys.size))