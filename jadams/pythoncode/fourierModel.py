# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 15:01:47 2015

@author: nesh, jonathan
"""
import numpy as np
import scipy as sp
#import copy
#from numba import njit,prange

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
    

    normalizedIFFT = np.fft.irfft(u_full)
    return normalizedIFFT

def convolution(nT,nu_kin,sigmastep,Nstar):
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
    convo = fftnorm(np.cos(ifftnorm(nT)))
    convo = 2 * np.pi * Nstar * nu_kin * sigmastep * convo
    return convo

#@njit("f8[:](f8[:],i4,f8[:],f8[:],f8)")
def nTotRHS(nQLL,nu_kin,sigmastep_FFT,k,D):
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

    #print(type(nQLL),type(nu_kin),type(sigmastep_FFT),type(k),type(D)) #print types of parameters
    #print(nQLL.shape,"INT",sigmastep_FFT.shape,k.shape,"Flo0at")       #print shapes of parameters

    dnTot = -k**2 * D * nQLL + 2*np.pi*nu_kin*sigmastep_FFT
    
    return dnTot

def nQLLRHS(nTot,nQLL,nu_kin,sigmastep,k,D,Nstar,N):
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
    
    
    F = fftnorm(sigmastep*ifftnorm(nTot))

    dnQLL = -k**2 * D * nQLL + 2 * np.pi * Nstar * nu_kin * F
    
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
    sigmastep = params['sigmastep']
    sigmastep_FFT = params['sigmastep_FFT']
    k = params['k']
    D = params['D']
    Nstar = params['Nstar']
    
    nTot = n[0:N]
    nQLL = n[N:]
    
    
    dnT = nTotRHS(nQLL,nu_kin,sigmastep_FFT,k,D)
    dnQ = nQLLRHS(nTot,nQLL,nu_kin,sigmastep,k,D,Nstar,N)
    
    RHS = np.concatenate((dnT,dnQ))

    return RHS

def runSim(params):
    """
    Runs an actual ROM or non-ROM simulation of KdV
    
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
    
    NTot = np.zeros((timesteps,2*N))
    NQLL = np.zeros((timesteps,2*N))
    
    for i in range(timesteps):
        nTot = fourierSol[0:N,i]
        nQLL = fourierSol[N:,i]

        nTotFull = np.zeros(2*N) 
        nQLLFull = np.zeros(2*N)
        
        nTotFull[0:N] = nTot
        nTotFull[N+1:] = np.conj(np.flip(nTot[1:]))
        nQLLFull[0:N] = nQLL
        nQLLFull[N+1:] = np.conj(np.flip(nQLL[1:]))
        
        NTot[i,:] = ifftnorm(nTotFull)
        NQLL[i,:] = ifftnorm(nQLLFull)
        
    return [NTot, NQLL]