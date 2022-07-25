# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 15:01:47 2015

@author: nesh, jonathan, Max
"""
import numpy as np
from numba import njit, float64, types

@njit("f8[:](f8[:],f8[:],f8,f8)") #important
def fqll_next_array(fqll_last,Ntot,Nstar,Nbar):
    #Ntot is a list of the amount of each type of ice
    fstar = Nstar/Nbar
    return 1 + fstar*np.sin(2*np.pi*(Ntot-Nbar*fqll_last))

@njit("f8(f8,f8,f8,f8)") #important
def fqll_next(fqll_last,Ntot,Nstar,Nbar):
    #Ntot is a list of the amount of each type of ice
    fstar = Nstar/Nbar
    return 1 + fstar*np.sin(2*np.pi*(Ntot-Nbar*fqll_last))

@njit("f8[:,:](f8[:,:],f8[:,:],f8,f8)") #important
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
    fqll_last = np.array([1.0])
    for i in range(niter):
        fqll_last = fqll_next_array(fqll_last,Ntot,Nstar,Nbar)
    return fqll_last*Nbar

@njit("f8[:,:](f8[:,:],f8,f8,i4)") #Ntot is ndarray of numbers (ints, become floats), Nstar and Nbar are floats, niter is an int literal
def getNliq_2d_array(Ntot,Nstar,Nbar,niter):
    """ Ntot is the ice- this returns the liquid layer prequilibrated to 1 bilayer equivlaent"""
    fqll_last = np.ones(np.shape(Ntot))
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

@njit("f8[:](f8[:],f8[:],f8[:],f8,f8)") #quirk: fqll_last is a float but must be array for above implemenetation
def getdfqll_dNtot_next_array(dfqll_dNtot_last,fqll_last,Ntot,Nstar,Nbar):
    fstar = Nstar/Nbar
    return fstar*np.cos(2*np.pi*(Ntot-fqll_last))*2*np.pi*(1-Nbar*dfqll_dNtot_last)

@njit("f8[:,:](f8[:,:],f8[:,:],f8[:,:],f8,f8)") #quirk: fqll_last is a float but must be array for above implemenetation
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

@njit("f8[:,:](f8[:,:],f8,f8,i4)")
def getdNliq_dNtot_2d_array(Ntot,Nstar,Nbar,niter):
    s = np.shape(Ntot)
    dfqll_dNtot_last = np.zeros(s) #np.array([0.0])
    fqll_last = np.ones(s) #np.array([1.0])

    for i in range(niter):
        dfqll_dNtot_last = getdfqll_dNtot_next_2d_array(dfqll_dNtot_last,fqll_last,Ntot,Nstar,Nbar)
        fqll_last = fqll_next_2d_array(fqll_last,Ntot,Nstar,Nbar)
    return dfqll_dNtot_last*Nbar 

@njit("f8[:](f8[:],f8,f8,i4)")
def getdNliq_dNtot_array(Ntot,Nstar,Nbar,niter):
    dfqll_dNtot_last = np.array([0.0])
    fqll_last = np.array([1.0])
    for i in range(niter):
        dfqll_dNtot_last = getdfqll_dNtot_next_array(dfqll_dNtot_last,fqll_last,Ntot,Nstar,Nbar)
        fqll_last = fqll_next_array(fqll_last,Ntot,Nstar,Nbar)
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

@njit("f8[:](f8[:],f8)")
def diffuse_1d(Fliq0,DoverdeltaX2):
    l = len(Fliq0)
    dy = np.zeros((l,))#np.shape(Fliq0))
    for i in range(0,l):#(1,l-1):
        dy[i] = DoverdeltaX2*(Fliq0[i+1]-2*Fliq0[i]+Fliq0[i-1])
        # Boundary Conditions (periodic at ends)
        dy[0] = DoverdeltaX2*(Fliq0[1]-2*Fliq0[0]+Fliq0[l-1]) 
        dy[l-1] = DoverdeltaX2*(Fliq0[0]-2*Fliq0[l-1]+Fliq0[l-2])
    return dy

@njit("f8[:](f8[:],f8,f8[:],i4[:],f8[:])")#, parallel = True)
def f1d(y, t, float_params, int_params, sigmastep): #sigmastep is an array
    """ odeint function for the one-dimensional ice model """
     # unpack parameters
    Nbar, Nstar, sigma0, deprate, DoverdeltaX2 = float_params 
    niter, nx = int_params

    # unpack current values of y
    Fliq0, Ntot0 = np.reshape(np.ascontiguousarray(y),(types.int32(2),types.int32(nx)))
    
    # Deposition
    delta = (Fliq0 - (Nbar - Nstar))/(2*Nstar)
    sigD = (sigmastep - delta * sigma0)/(1+delta*sigma0)
    depsurf = deprate * sigD
    dFliq0_dt = getdNliq_dNtot_array(Ntot0,Nstar,Nbar,niter)*depsurf
    dNtot_dt = depsurf

    # Diffusion
    l = len(Fliq0)
    dy = np.zeros((l,))#np.shape(Fliq0))
    for i in range(0,l-1):#(1,l-1):
        dy[i] = DoverdeltaX2*(Fliq0[i+1]-2*Fliq0[i]+Fliq0[i-1])
        # Boundary Conditions (periodic at ends)
        dy[0] = DoverdeltaX2*(Fliq0[1]-2*Fliq0[0]+Fliq0[l-1]) 
        dy[l-1] = DoverdeltaX2*(Fliq0[0]-2*Fliq0[l-1]+Fliq0[l-2])
     
    # Combined
    dFliq0_dt += dy
    dNtot_dt += dy

    # Package for output
    derivs = np.reshape(np.array([[*dFliq0_dt], [*dNtot_dt]]),2*nx) #need to unpack lists back into arrays of proper shape (2,nx) before reshaping
    return derivs

@njit("f8[:,:](f8[:,:],f8)")
def diffuse_2d(Fliq0,D):
    """ Applies numerical solution to find diffusive effects at each time step.
    
    Parameters
    ----------
    Fliq0 : 2D Numpy Array 
        The thickness of the liquid over a 2D area

    D : float64
        Diffusion coefficient

    Returns
    -------
    dy : 2D Numpy Array
        The change to the thickness of the liquid at each point in the 2d area over
         the time step
    """
    s = np.shape(Fliq0)
    dy = np.zeros(s) 
    for i in range(0,s[1]): #go across the liquid, from top to bottom
        dy[i,:] = diffuse_1d(Fliq0[i,:],D)#calculate left to right at each row 
        for j in range(0,s[0]): # for each row, calculate the effects on rows below it
            dy[:,j] = diffuse_1d(Fliq0[:,j],D)
    #NOTE: 1d func manages periodic boundary conditions
    return dy

@njit("f8[:](f8[:],f8,f8[:],i4[:],f8[:])")
def f2d(y, t, float_params, int_params, sigmastep):#NOTE: sigmastep needs to become 2D -- rotate parabola around vertical axis
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
    dy = diffuse_2d(Fliq0,DoverdeltaX2)
     
    # Combined
    dFliq0_dt += dy
    dNtot_dt += dy

    # Package for output
    #derivs = np.reshape([dFliq0_dt, dNtot_dt],(2,nx,nx) ) #np.reshape(np.array([[*dFliq0_dt], [*dNtot_dt]]),2*nx) #need
    ## NOTE: this is not it   -- ##### #derivs = np.reshape(np.ascontiguousarray((np.concatenate((dFliq0_dt,dNtot_dt),axis=0))),(types.int32(2),types.int32(nx))) #NOTE: two arrays(of size nx) containing arrays (also size nx)

   # derivs = np.reshape(np.ascontiguousarray(np.array([dFliq0_dt,dNtot_dt])),2*nx*ny)
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