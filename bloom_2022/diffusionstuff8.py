# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 15:01:47 2015

@author: nesh, jonathan, Max, jake
"""

import numpy as np
#import math
from numba import njit, float64, types#,guvectorize

prll_1d = False # 1d faster without parallelization
prll_2d = True  # 2d faster with parallelization

@njit("f8[:](f8[:],f8,f8[:],i4)")
def f0d(y, t, float_params, niter):
    """ odeint function for the zero-dimensional ice model """
    Nbar, Nstar, sigmastepmax, sigma0, deprate = float_params  # unpack parameters
    
    Fliq0, Ntot0 = y   # unpack current values of y

    delta = (Fliq0 - (Nbar - Nstar))/(2*Nstar)
    sigD = (sigmastepmax - delta * sigma0)/(1+delta*sigma0)
    depsurf = deprate * sigD

    #dFliq0_dt = getNliqprime(Ntot0,Nstar,Nbar,niter)*depsurf
    dFliq0_dt = Nstar*np.cos(2*np.pi*(Ntot0))*2*np.pi*depsurf
    dNtot_dt = depsurf
    
    derivs = np.array([dFliq0_dt, dNtot_dt])
    return derivs

@njit("f8[:](f8[:],f8)",parallel=prll_1d)
def diffuse_1d(Fliq0,DoverdeltaX2):
    l = len(Fliq0)
    dy = np.zeros((l,))
    for i in range(0,l):
        dy[i] = DoverdeltaX2*(Fliq0[i+1]-2*Fliq0[i]+Fliq0[i-1])
        # Boundary Conditions (periodic at ends)
        dy[0] = DoverdeltaX2*(Fliq0[1]-2*Fliq0[0]+Fliq0[l-1]) 
        dy[l-1] = DoverdeltaX2*(Fliq0[0]-2*Fliq0[l-1]+Fliq0[l-2])
    return dy

@njit("f8[:](f8[:],f8,f8)")#,parallel=prll_1d) #NOTE: to test with paralellization
def get_qll_1d(Ntot,Nbar,Nstar):
    return Nbar + Nstar * np.sin(2*np.pi*(Ntot - Nbar))

@njit("f8[:](f8,f8[:],f8[:],f8[:])", parallel=prll_1d)#slower with paralellization right now
def f1d(t, Ntot, float_params, sigmastep):
    """ odeint function for the one-dimensional ice model """
     # unpack parameters
    Nbar, Nstar, sigma0, deprate, DoverdeltaX2 = float_params

    # compute quasi-liquid layer from Ntot
    NQLL = get_qll_1d(Ntot,Nbar,Nstar)# Nbar + Nstar * np.sin(2*np.pi*(Ntot - Nbar))
    
    delta = (NQLL - (Nbar - Nstar))/(2*Nstar)
    sigD = (sigmastep - delta * sigma0)/(1+delta*sigma0)
    depsurf = deprate * sigD
    dNtot_dt = depsurf

    # Diffusion
    dy = diffuse_1d(NQLL,DoverdeltaX2)
     
    # Combined
    dNtot_dt += dy
    return dNtot_dt

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
    Fliq0 = y
    m,n = shape
    Fliq0 = np.reshape(np.ascontiguousarray(Fliq0),(m,n)) #reshaping required for odeint/solve_ivp
    #Fliq0 = np.reshape(Fliq0,(m,n)) #reshaping required for odeint/solve_ivp
    dy = np.zeros((m,n)) 
   
    for i in range(0,m): #go from left column to right
        for j in range(0,n): #go from top row to bottom
            ip1=i+1 #i plus one
            jp1=j+1
            # Boundary Conditions (periodic at ends)
            if i == m-1: #take care of right column condition wrapping to left edge
                ip1 = 0
            if j == n-1: #take care of bottom edge wrapping to top edge
                jp1 = 0

            ux = (Fliq0[ip1,j] - 2*Fliq0[i,j] + Fliq0[i-1,j])
            uy = (Fliq0[i,jp1] - 2*Fliq0[i,j] + Fliq0[i,j-1])

            dy[i,j] = D*(ux+uy)
    #dy = diffuse_vector_helper(Fliq0,dy,D)
            
    return np.reshape(dy,(m*n))

@njit("f8[:,:](f8[:,:],f8,f8)")#,parallel=prll_2d) #NOTE: to test with paralellization
def get_qll_2d(Ntot,Nbar,Nstar):
    return Nbar + Nstar * np.sin(2*np.pi*(Ntot - Nbar))

@njit("f8[:](f8,f8[:],f8[:],i8[:],f8[:,:])", parallel=prll_2d) #NOTE: t and y swapped for solve_ivp compatability
def f2d(t, y, float_params, int_params, sigmastep):
    """ 2D version of f1d """

    # diffusion = True

    # unpack parameters
    Nbar, Nstar, sigma0, deprate, DoverdeltaX2 = float_params 
    nx, ny = int_params

    
    # unpack current values of y
    Ntot = np.reshape(np.ascontiguousarray(y),(nx,ny))

    #Calculate QLL from ntot
    NQLL = get_qll_2d(Ntot,Nbar,Nstar)#Nbar + Nstar * np.sin(2*np.pi*(Ntot - Nbar))

    # Deposition
    delta = (NQLL - (Nbar - Nstar))/(2*Nstar)
    sigD = (sigmastep - delta * sigma0)/(1+delta*sigma0)
    depsurf = deprate * sigD

    dNtot_dt = depsurf

    # Diffusion
    dy =  np.reshape(np.ascontiguousarray( diffuse_2d(t, np.reshape(np.ascontiguousarray(NQLL),nx*ny), DoverdeltaX2, np.array((nx,ny))) ), (nx,ny))
    # Combined
    #dFliq0_dt += dy
    dNtot_dt += dy

    ## Package for output
    #derivs = np.reshape(dNtot_dt,nx*ny)
    #return derivs
    return np.reshape(dNtot_dt,nx*ny)
    #return dNtot_dt

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