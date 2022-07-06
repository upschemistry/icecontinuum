# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 15:01:47 2015

@author: nesh, jonathan, jake
"""
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
#import copy
from numba import njit,prange

def fftnorm(u_full):
    """Computes normalized FFT (such that FFT and IFFT are symmetrically normalized)
    
    Parameters
    ----------
    u_full : 1D Numpy Array (N,)
        The vector whose discrete FFT is to be computed

    Returns
    -------
    normalizedFFT : 1D Numpy Array (N,)
        The transformed version of that vector
    """

    normalizedFFT = np.fft.rfft(u_full,norm = "forward")
    return normalizedFFT

def ifftnorm(u_full):
    """Computes normalized IFFT (such that FFT and IFFT are symmetrically normalized)
    
    Parameters
    ----------
    u_full : 1D Numpy Array (N,)
        The vector whose discrete IFFT is to be computed

    Returns
    -------
    normalizedIFFT : 1D Numpy Array (N,)
        The transformed version of that vector
    """
    

    normalizedIFFT = np.fft.irfft(u_full, norm = "forward")
    return normalizedIFFT

def convolution(nT,nu_kin,depRate,Nstar):
    """Computes Fourier transform of the nonlinear term in the QLL PDE
    
    2 pi N^* sigmaM vKin cos(Ntot)
    
    Computed in real space and then converted back
    to Fourier space.
    
    Parameters
    ----------
    nT : 1D Numpy Array (N,)
        Total water layers
        
    nu_kin : TBD
        TBD
        
    sigmastep : TBD
        TBD
        
    Nstar : TBD
        TBD

    Returns
    -------
    convo : 1D Numpy Array (N,)
        Fourier transform of the nonlinear term
    """
    
    # compute double sum in real space, then apply scalar multiplier
    convo = 2 * np.pi * Nstar * nu_kin * fftnorm(depRate * np.cos(ifftnorm(nT)))
    return convo

#@njit("f8[:](f8[:],i4,f8[:],f8[:],f8)")
def nTotRHS(nQLL,nu_kin,depRate_FFT,k,D):
    """Computes RHS of the ODE for the positive modes of Ntot
    
    dnk/dt = -k^2 D nkQLL + 2 pi FFT(sigma_m) nu_kin
    
    
    Parameters
    ----------
    nQLL : 1D Numpy Array (N,)
        Positive modes of state vector for quasi-liquid layers
        
    nu_kin : TBD
        TBD
        
    sigmastep_FFT : TBD
        TBD
        
    k : 1D Numpy Array (N,)
        Vector of wavenumbers
        
    D : float
        Diffusion coefficient

    Returns
    -------
    dnTot : 1D Numpy Array (N,)
        Rate of change of positive modes of nTot
    """

    dnTot = -k**2 * D * nQLL + depRate_FFT
    
    return dnTot

def nQLLRHS(nTot,nQLL,nu_kin,depRate,k,D,Nstar,N):
    """Computes RHS of the ODE for the positive modes of Ntot
    
    dn0/dt = 2 * pi * sigma_m * nu_kin
    dnk/dt = -k^2 D nkQLL
    
    
    Parameters
    ----------
    nTot : 1D Numpy Array (N,)
        Positive modes of state vector for total layers
    
    nQLL : 1D Numpy Array (N,)
        Positive modes of state vector for quasi-liquid layers
        
    nu_kin : TBD
        TBD
        
    sigmastep_FFT : TBD
        TBD
        
    k : 1D Numpy Array (N,)
        Vector of wavenumbers
        
    D : float
        Diffusion coefficient
        
    Nstar : float
        TBD

    Returns
    -------
    dnQLL : 1D Numpy Array (N,)
        Rate of change of positive modes of nTot
    """
    
    convo = convolution(nTot,nu_kin,depRate, Nstar)

    dnQLL = -k**2 * D * nQLL + convo
    
    return dnQLL


def RHS(t,n,params):
    """
    Computes the RHS for a full KdV or ROM simulation. For use in solver.
    
    Parameters
    ----------
    t : float
        Current time
        
    n : Numpy array (2N,)
        Current state vector of positive modes (total first, then QLL)
              
    params : Dictionary
             Dictionary of relevant parameters (see below)
        N : float, number of positive modes in simulation
        nu_kin : 
        sigmastep : 
        sigmastep_FFT : 
        k : 
        D : 

        
    Returns
    -------
    RHS : 1D Numpy array (2N,)
          Derivative of each positive mode in state vector
    """
    
    # extract parameters from dictionary
    N = params['N']
    nu_kin = params['nu_kin']
    depRate = params['depRate']
    depRate_FFT = params['depRate_FFT']
    k = params['k']
    D = params['D']
    Nstar = params['Nstar']
    
    nTot = n[0:N]
    nQLL = n[N:]
    
    
    dnT = nTotRHS(nQLL,nu_kin,depRate_FFT,k,D)
    dnQ = nQLLRHS(nTot,nQLL,nu_kin,depRate,k,D,Nstar,N)
    
    RHS = np.concatenate((dnT,dnQ))

    return RHS

def runSim(params):
    """
    Runs a simulation of the ice continuum in Fourier space
    
    Parameters
    ----------
    params : Dictionary
             Dictionary of relevant parameters (see below)
        N : float, number of positive modes in simulation
        nu_kin : 
        sigmastep : 
        sigmastep_FFT : 
        k : 
        D : 

        
    Returns
    -------
    uSim : ODE solver output
           Output solution from sp.integrate.solve_ivp (includes state vector at all timesteps, time vector, etc.)
    """
    
    # unpack parameters from dictionary
    N = params['N']
    ICNT = params['ICNT']
    ICNQLL = params['ICNQLL']
    endtime = params['endtime']
    timesteps = params['timesteps']
    
    nTotIC = fftnorm(ICNT)[0:N]
    nQLLIC = fftnorm(ICNQLL)[0:N]
    
    n = np.concatenate((nTotIC,nQLLIC))
    
    # define RHS in form appropriate for solve_ivp
    def myRHS(t,y):
        out = RHS(t,y,params)
        return out
    
    # solve the IVP
    uSim = sp.integrate.solve_ivp(fun = myRHS, t_span = [0,endtime], y0 = n, t_eval = timesteps)
    return uSim

def makeReal(fourierSol): 
    
    N = int(fourierSol.shape[0]/2)
    timesteps = fourierSol.shape[1]
    
    NTot = np.zeros((timesteps,2*N-2))
    NQLL = np.zeros((timesteps,2*N-2))
    
    for i in range(timesteps):
        NTot[i,:] = ifftnorm(fourierSol[0:N,i])
        NQLL[i,:] = ifftnorm(fourierSol[N:,i])
        
    return [NTot, NQLL]








from numba import njit, float64,types

#NOTE: as of june 20 2022, the only functions that are used in continuum_model6 
#       are those with explicit type signatures.

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

@njit("f8[:](f8[:],f8,f8,i8)") #Ntot is ndarray of numbers (ints, become floats), Nstar and Nbar are floats, niter is an int literal
def getNliq_array(Ntot,Nstar,Nbar,niter):
    fqll_last = np.array([1.0])
    for i in range(niter):
        fqll_last = fqll_next_array(fqll_last,Ntot,Nstar,Nbar)
    return fqll_last*Nbar

@njit("f8(f8,f8,f8,i8)") #Ntot is float in this case, Nstar and Nbar are floats, niter is an int literal
def getNliq(Ntot,Nstar,Nbar,niter):
    fqll_last = 1.0
    for i in range(niter):
        fqll_last = fqll_next(fqll_last,Ntot,Nstar,Nbar)
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

@njit("f8[:](f8[:],f8,f8,i8)")
def getdNliq_dNtot_array(Ntot,Nstar,Nbar,niter):
    dfqll_dNtot_last = np.array([0.0])
    fqll_last = np.array([1.0])
    for i in range(niter):
        dfqll_dNtot_last = getdfqll_dNtot_next_array(dfqll_dNtot_last,fqll_last,Ntot,Nstar,Nbar)
        fqll_last = fqll_next_array(fqll_last,Ntot,Nstar,Nbar)
    return dfqll_dNtot_last*Nbar 

@njit("f8(f8,f8,f8,i8)")
def getdNliq_dNtot(Ntot,Nstar,Nbar,niter):
    dfqll_dNtot_last = 0.0
    fqll_last = 1.0
    for i in range(niter):
        dfqll_dNtot_last = getdfqll_dNtot_next(dfqll_dNtot_last,fqll_last,Ntot,Nstar,Nbar)
        fqll_last = fqll_next(fqll_last,Ntot,Nstar,Nbar)
    return dfqll_dNtot_last*Nbar 

@njit("f8[:](f8[:],f8,f8[:],i8)") #float_params is a list not an array
def f0d(y, t, float_params, niter):
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

#array(float64, 1d, C), float64, array(float64, 1d, C), array(int32, 1d, C), array(float64, 1d, C)
#@njit("f8[:](f8[:],f8,f8[:],i4[:],f8[:])")#, parallel = True) # in the current use case it is faster without paralellization (requires use of prange instead of range)
def f1d(y, t, float_params, int_params, sigmastep): #sigmastep is an array
     # unpack parameters
    Nbar, Nstar, sigma0, deprate, DoverdeltaX2 = float_params 
    niter, nx = int_params

    # unpack current values of y
    Fliq0, Ntot0 = np.reshape(np.ascontiguousarray(y),(types.int32(2),types.int32(nx)))
    
    # Deposition - JAKE NOTE - I THINK THIS IS WHERE I'M GOING WRONG
    delta = (Fliq0 - (Nbar - Nstar))/(2*Nstar)
    sigD = (sigmastep - delta * sigma0)/(1+delta*sigma0)
    depsurf = deprate * sigD
    dFliq0_dt = getdNliq_dNtot_array(Ntot0,Nstar,Nbar,niter)*depsurf
    dNtot_dt = depsurf

    # Diffusion
    l = len(Fliq0)
    dy = np.zeros((l,))#np.shape(Fliq0))
    for i in range(0,l):#(1,l-1):
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

@njit(float64[:](float64[:],float64,float64,float64,types.unicode_type))
def getsigmastep(x,xmax,center_reduction,sigmastepmax,method='parabolic'):
    sigmapfac = 1-center_reduction/100 #float64
    xmid = max(x)/2 #float64
    if method == 'sinusoid':
        fsig = (np.cos(x/xmax*np.pi*2)+1)/2*(1-sigmapfac)+sigmapfac
    elif method == 'parabolic':
        fsig = (x-xmid)**2/xmid**2*(1-sigmapfac)+sigmapfac
    else:
        print('bad method')
    return fsig*sigmastepmax