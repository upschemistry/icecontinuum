# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 15:01:47 2015

@author: nesh, jonathan, Max
"""

import numpy as np
from numba import njit, float64, types

prll_bool = False

@njit("f8(f8,f8,f8,f8)") 
def fqll_next(fqll_last,Ntot,Nstar,Nbar):
    #Ntot is a list of the amount of each type of ice
    fstar = Nstar/Nbar
    return 1 + fstar*np.sin(2*np.pi*(Ntot-Nbar*fqll_last))

@njit("f8[:](f8[:],f8[:],f8,f8)", parallel=prll_bool)
def fqll_next_array(fqll_last,Ntot,Nstar,Nbar):
    #Ntot is a list of the amount of each type of ice
    fstar = Nstar/Nbar
    return 1 + fstar*np.sin(2*np.pi*(Ntot-Nbar*fqll_last))

@njit("f8[:,:](f8[:,:],f8[:,:],f8,f8)", parallel=prll_bool)
def fqll_next_2d_array(fqll_last,Ntot,Nstar,Nbar):
    #Ntot is a list of the amount of each type of ice
    fstar = Nstar/Nbar
    return 1 + fstar*np.sin(2*np.pi*(Ntot-Nbar*fqll_last))


@njit("f8(f8,f8,f8,i4)") #Ntot is float in this case, Nstar and Nbar are floats, niter is an int literal
def getNliq(Ntot,Nstar,Nbar,niter):
    fqll_last = 1.0
    for i in range(niter):
        fqll_last = fqll_next(fqll_last,Ntot,Nstar,Nbar)
    return fqll_last*Nbar

@njit("f8[:](f8[:],f8,f8,i4)") #Ntot is ndarray of numbers (ints, become floats), Nstar and Nbar are floats, niter is an int literal
def getNliq_array(Ntot,Nstar,Nbar,niter):
    fqll_last = np.array([1.0]*np.shape(Ntot)[0])
    for i in range(niter):
        fqll_last = fqll_next_array(fqll_last,Ntot,Nstar,Nbar)
    return fqll_last*Nbar

@njit("f8[:,:](f8[:,:],f8,f8,i4)") #Ntot is ndarray of numbers (ints, become floats), Nstar and Nbar are floats, niter is an int literal
def getNliq_2d_array(Ntot,Nstar,Nbar,niter):
    """ Ntot is the ice- this returns the liquid layer prequilibrated to 1 bilayer equivlaent"""
    m,n = np.shape(Ntot)
    fqll_last = np.ones((m,n))
    for i in range(niter):
        fqll_last = fqll_next_2d_array(fqll_last,Ntot,Nstar,Nbar)
    return fqll_last*Nbar

@njit("f8[:](f8[:],f8[:],f8,f8)")
def fqllprime_next(fqll_last,Ntot,Nstar,Nbar):
    fstar = Nstar/Nbar
    return 1 + fstar*np.sin(2*np.pi*(Ntot-Nbar*fqll_last))

@njit("f8(f8,f8,f8,f8,f8)") #quirk: fqll_last is a float but must also have array implemenetation for 1-d model
def getdfqll_dNtot_next(dfqll_dNtot_last,fqll_last,Ntot,Nstar,Nbar):
    fstar = Nstar/Nbar
    return fstar*np.cos(2*np.pi*(Ntot-fqll_last))*2*np.pi*(1-Nbar*dfqll_dNtot_last)

@njit("f8[:](f8[:],f8[:],f8[:],f8,f8)",parallel=prll_bool) #quirk: fqll_last is a float but must be array for above implemenetation
def getdfqll_dNtot_next_array(dfqll_dNtot_last,fqll_last,Ntot,Nstar,Nbar):
    fstar = Nstar/Nbar
    return fstar*np.cos(2*np.pi*(Ntot-fqll_last))*2*np.pi*(1-Nbar*dfqll_dNtot_last)

@njit("f8[:,:](f8[:,:],f8[:,:],f8[:,:],f8,f8)",parallel=prll_bool) #quirk: fqll_last is a float but must be array for above implemenetation
def getdfqll_dNtot_next_2d_array(dfqll_dNtot_last,fqll_last,Ntot,Nstar,Nbar):
    fstar = Nstar/Nbar
    return fstar*np.cos(2*np.pi*(Ntot-fqll_last))*2*np.pi*(1-Nbar*dfqll_dNtot_last)

@njit("f8(f8,f8,f8,i4)")
def getdNliq_dNtot(Ntot,Nstar,Nbar,niter):
    dfqll_dNtot_last = 0.0
    fqll_last = 1.0
    for i in range(niter):
        dfqll_dNtot_last = getdfqll_dNtot_next(dfqll_dNtot_last,fqll_last,Ntot,Nstar,Nbar)
        fqll_last = fqll_next(fqll_last,Ntot,Nstar,Nbar)
    return dfqll_dNtot_last*Nbar 

@njit("f8[:](f8[:],f8,f8,i4)",parallel=prll_bool)
def getdNliq_dNtot_array(Ntot,Nstar,Nbar,niter):
    dfqll_dNtot_last = np.zeros(np.shape(Ntot)[0])
    fqll_last = np.ones(np.shape(Ntot)[0])
    for i in range(niter):
        dfqll_dNtot_last = getdfqll_dNtot_next_array(dfqll_dNtot_last,fqll_last,Ntot,Nstar,Nbar)
        fqll_last = fqll_next_array(fqll_last,Ntot,Nstar,Nbar)
    return dfqll_dNtot_last*Nbar 

@njit("f8[:,:](f8[:,:],f8,f8,i4)",parallel=prll_bool)
def getdNliq_dNtot_2d_array(Ntot,Nstar,Nbar,niter):
    m,n = np.shape(Ntot)
    s =(m,n)
    dfqll_dNtot_last = np.zeros(s) #np.array([0.0])
    fqll_last = np.ones(s) #np.array([1.0])

    for i in range(niter):
        dfqll_dNtot_last = getdfqll_dNtot_next_2d_array(dfqll_dNtot_last,fqll_last,Ntot,Nstar,Nbar)
        fqll_last = fqll_next_2d_array(fqll_last,Ntot,Nstar,Nbar)
    return dfqll_dNtot_last*Nbar 

@njit("f8[:](f8[:],f8,f8[:],i4)")
def f0d(y, t, float_params, niter):
    """ odeint function for the zero-dimensional ice model """
    Nbar, Nstar, sigmastepmax, sigma0, deprate = float_params  # unpack parameters
    
    Fliq0, Ntot0 = y   # unpack current values of y
    #Fliq0, Ntot0 = np.reshape(y,2)    

    delta = (Fliq0 - (Nbar - Nstar))/(2*Nstar)
    sigD = (sigmastepmax - delta * sigma0)/(1+delta*sigma0)
    depsurf = deprate * sigD

    #dFliq0_dt = getNliqprime(Ntot0,Nstar,Nbar,niter)*depsurf
    dFliq0_dt = getdNliq_dNtot(Ntot0,Nstar,Nbar,int(niter))*depsurf
    dNtot_dt = depsurf
    
    derivs = np.array([dFliq0_dt, dNtot_dt])
    return derivs

@njit("f8[:](f8[:],f8)",parallel=prll_bool)
def diffuse_1d(Fliq0,DoverdeltaX2):
    l = len(Fliq0)
    dy = np.zeros((l,))#np.shape(Fliq0))
    for i in range(0,l):#(1,l-1):
        dy[i] = DoverdeltaX2*(Fliq0[i+1]-2*Fliq0[i]+Fliq0[i-1])
        # Boundary Conditions (periodic at ends)
        dy[0] = DoverdeltaX2*(Fliq0[1]-2*Fliq0[0]+Fliq0[l-1]) 
        dy[l-1] = DoverdeltaX2*(Fliq0[0]-2*Fliq0[l-1]+Fliq0[l-2])
    return dy

@njit("f8[:](f8[:],f8,f8[:],i4[:],f8[:])",parallel=prll_bool)#slower with paralellization right now
def f1d(y, t, float_params, int_params, sigmastep): #sigmastep is an array
    """ odeint function for the one-dimensional ice model """
     # unpack parameters
    Nbar, Nstar, sigma0, deprate, DoverdeltaX2 = float_params 
    niter, nx = int_params

    # unpack current values of y
    Fliq0, Ntot0 = np.reshape(np.ascontiguousarray(y),(types.int32(2),types.int32(nx)))#TODO c
    
    # Deposition
    delta = (Fliq0 - (Nbar - Nstar))/(2*Nstar)
    sigD = (sigmastep - delta * sigma0)/(1+delta*sigma0)
    depsurf = deprate * sigD
    dFliq0_dt = getdNliq_dNtot_array(Ntot0,Nstar,Nbar,niter)*depsurf
    dNtot_dt = depsurf

    # Diffusion
    dy = diffuse_1d(Fliq0,DoverdeltaX2)
     
    # Combined
    dFliq0_dt += dy
    dNtot_dt += dy

    # Package for output
    #derivs = np.reshape(np.array([[*dFliq0_dt], [*dNtot_dt]]),2*nx) #need to unpack lists back into arrays of proper shape (2,nx) before reshaping
    derivs = np.reshape(np.stack((dFliq0_dt,dNtot_dt),axis=0),2*nx)
    return derivs

@njit("f8[:](f8[:],f8,f8[:],i4[:],f8[:])",parallel=prll_bool)#slower with paralellization right now
def f1d_diff_only(y, t, float_params, int_params, sigmastep): #sigmastep is an array
    """ odeint function for the one-dimensional ice model-- only does diffusion of liquid layer"""
     # unpack parameters
    Nbar, Nstar, sigma0, deprate, DoverdeltaX2 = float_params 
    niter, nx = int_params

    # unpack current values of y
    Fliq0, Ntot0 = np.reshape(np.ascontiguousarray(y),(types.int32(2),types.int32(nx))) #old
    #Fliq0 = np.reshape(y,nx)
    
    # Deposition
    delta = (Fliq0 - (Nbar - Nstar))/(2*Nstar)
    sigD = (sigmastep - delta * sigma0)/(1+delta*sigma0)
    depsurf = deprate * sigD
    dFliq0_dt = getdNliq_dNtot_array(Ntot0,Nstar,Nbar,niter)*depsurf
    dNtot_dt = depsurf

    # Diffusion
    dy = diffuse_1d(Fliq0,DoverdeltaX2)

    # Combined
    dFliq0_dt += dy
    dNtot_dt += dy

    # Package for output
    print('shape of dFliq0_dt: ',dFliq0_dt.shape)
    derivs = np.reshape(dFliq0_dt,nx)

    return derivs

#@njit("f8[:](f8[:],f8,f8,i4[:])",parallel=prll_bool)
@njit(float64[:](float64,float64[:],float64,types.UniTuple(types.int64,2)),parallel=prll_bool)
def diffuse_2d(t,y,D,shape):
    """ Applies numerical solution to find diffusive effects at each time step. Fliq0 is flattened 2d array of shape shape.
    
    Parameters
    ----------
    Fliq0 : 2D Numpy Array 
        The thickness of the liquid over a 2D area

    t : float
        The time step-- unused- placeholder for odeint

    D : float64
        Diffusion coefficient -- divided by deltaX^2??? #TODO needs to be divided by 
            x^2 or y^2 inside this function in order to have non-square discretization

    shape : tuple
        The shape of the 2D array Fliq0

    Returns
    -------
    dy : Flattened 2D Numpy Array
        The change to the thickness of the liquid at each point in the 2d area over
         the time step
    """
    Fliq0 = y
    Fliq0 = np.reshape(np.ascontiguousarray(Fliq0),shape)
    m,n = shape
    dy = np.zeros((m,n)) 
    """
    #attempt 1
    for i in range(0,n): #go across the liquid, from top to bottom
        dy[i,:] = diffuse_1d(Fliq0[i,:],D)#calculate left to right at each row 
        for j in range(0,m): # for each row, calculate the effects on rows below & above  it
            dy[:,j] = diffuse_1d(Fliq0[:,j],D)
    #NOTE: 1d func manages periodic boundary conditions
    """
    #attempt 2
    for i in range(0,m): #go from left column to right
        for j in range(0,n): #go from top row to bottom

            ip1=i+1
            jp1=j+1
            # Boundary Conditions (periodic at ends) #TODO: test this
           
            if i == m-1: #take care of right column condition wrapping to left edge
                ip1 = 0

            if j == n-1: #take care of bottom edge wrapping to top edge
                jp1 = 0

            ux = (Fliq0[ip1,j] - 2*Fliq0[i,j] + Fliq0[i-1,j])
            uy = (Fliq0[i,jp1] - 2*Fliq0[i,j] + Fliq0[i,j-1])

            dy[i,j] = D*(ux+uy)
            
    return np.reshape(dy,(m*n))

@njit("f8[:](f8[:],f8,f8[:],i4[:],f8[:])",parallel=prll_bool)
def f2d(y, t, float_params, int_params, sigmastep):#NOTE, TODO: sigmastep needs to become 2D -- rotate parabola around vertical axis
    """ 2D version of f1d """
    # unpack parameters
    Nbar, Nstar, sigma0, deprate, DoverdeltaX2 = float_params 
    niter, nx, ny = int_params

    # unpack current values of y
    Fliq0, Ntot0 = np.reshape(np.ascontiguousarray(y),(types.int32(2),types.int32(nx),types.int32(ny)))
    
    # Deposition
    delta = (Fliq0 - (Nbar - Nstar))/(2*Nstar)
    sigD = (sigmastep - delta * sigma0)/(1+delta*sigma0)
    depsurf = deprate * sigD
    dFliq0_dt = getdNliq_dNtot_2d_array(Ntot0,Nstar,Nbar,niter)*depsurf
    dNtot_dt = depsurf

    # Diffusion
    dy = diffuse_2d(t,np.reshape(np.ascontiguousarray(Fliq0),nx*ny),DoverdeltaX2,(types.int32(nx),types.int32(ny)))
     
    # Combined
    dFliq0_dt += dy
    dNtot_dt += dy

    # Package for output
    derivs = np.reshape(np.stack((dFliq0_dt,dNtot_dt),axis=0),2*nx*ny)#(types.int32(2),nx,ny) )#,order='C')
    return derivs

@njit(float64[:](float64[:],float64,float64,float64,types.unicode_type))
def getsigmastep(x,xmax,center_reduction,sigmastepmax,method='parabolic'): 
    sigmapfac = 1-center_reduction/100 #float64
    xmid = max(x)/2 #float64
    try:
        if method == 'sinusoid':
            fsig = (np.cos(x/xmax*np.pi*2)+1)/2*(1-sigmapfac)+sigmapfac
        elif method == 'parabolic':
            fsig = (x-xmid)**2/xmid**2*(1-sigmapfac)+sigmapfac
    except:
        print('bad method')

    return fsig*sigmastepmax

@njit(float64[:](float64[:],float64,float64,float64,types.unicode_type))
def getsigmastep_2d(x,xmax,center_reduction,sigmastepmax,method='parabolic'): #TODO: implement 
    sigmapfac = 1-center_reduction/100 #float64
    xmid = max(x)/2 #float64
    
    fsig = (x-xmid)**2/xmid**2*(1-sigmapfac)+sigmapfac
    
    return fsig*sigmastepmax