
"""
Created on Mon Jun 19 2023

@author: ella
"""


import numpy as np
import scipy as sp
import matplotlib.pyplot as plt



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


def convolution(nT,nu_kin,sigmaM,Nstar):
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
    convo = 2 * np.pi * Nstar * nu_kin * fftnorm(sigmaM * np.cos(2*np.pi * ifftnorm(nT)))
    return convo

def nTotRHS(nQLL,nu_kin,sigmaM,k,D):
    """Computes RHS of the ODE for the positive modes of Ntot
    
    ##TODO: why factor of 2pi??
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

    dnTot = -k**2 * D * nQLL + nu_kin * fftnorm(sigmaM)
    return dnTot

def nQLLRHS(nTot,nQLL,nu_kin,depRate,k,D,Nstar,N):
    """Computes RHS of the ODE for the positive modes of Ntot
    
    ##TODO: i dont see where the 2pi in dn0 comes from??
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