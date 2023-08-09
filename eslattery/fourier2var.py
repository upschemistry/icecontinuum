
"""
Created on Mon Jun 19 2023

@author: ella
"""

import diffusionstuff10 as ds
import numpy as np
from scipy.integrate import solve_ivp



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

def convolution(nTOTk,sigMk,params):
    """Computes Fourier transform of the nonlinear term in the QLL PDE
    
    2 pi N^* sigmaM vKin cos(Ntot)
    
    Computed in real space and then converted back
    to Fourier space.
    
    Parameters
    ----------
    nTOTk : 1D Numpy Array (N,)
        Positive modes of state vector for total layers
    sigMk : 1D Numpy array (N,)
        Sigma M, in k space
    params : Dictionary
             Dictionary of relevant parameters (see below)
        nu_kin_mlyperus : float, speed of water vapor hitting qll layer in monolayers per microsecond
        Nstar : float, best fit amplitude for sinusoidal NQLL(Ntot)

    Returns
    -------
    convo : 1D Numpy Array (N,)
        Fourier transform of the nonlinear term
    """
    
    # unpack params
    nu_kin_mlyperus = params['nu_kin_mlyperus']
    Nstar = params['Nstar']

    # compute double sum in real space, then apply scalar multiplier
    convo = -2 * np.pi * Nstar * nu_kin_mlyperus * fftnorm(ifftnorm(sigMk) * np.cos(2*np.pi * ifftnorm(nTOTk)))
    return convo

def nTotRHS(nQLLk,sigMk,params):
    """Computes RHS of the ODE for the positive modes of Ntot
    
    dnk/dt = -k^2 D nkQLL + 2 pi FFT(sigma_m) nu_kin
    
    Parameters
    ----------
    nQLLk : 1D Numpy Array (N,)
        Positive modes of state vector for quasi-liquid layers
    sigMk : 1D Numpy array (N,)
        Sigma M, in k space
    params : Dictionary
             Dictionary of relevant parameters (see below)
        nu_kin_mlyperus : float, speed of water vapor hitting qll layer in monolayers per microsecond 
        k : 1D Numpy array (N,), array of available wavenumbers
        D : float, diffusion coefficient

    Returns
    -------
    dnTot : 1D Numpy Array (N,)
        Rate of change of positive modes of nTot
    """

    # unpack params
    nu_kin_mlyperus = params['nu_kin_mlyperus']
    k = params['k']
    D = params['D']

    # define dntot/dt rhs
    dnTot = -k**2 * D * nQLLk + nu_kin_mlyperus * sigMk
    return dnTot

def nQLLRHS(nTOTk,nQLLk,sigMk,params):
    """Computes RHS of the ODE for the positive modes of Ntot
    
    dn0/dt = 2 * pi * sigma_m * nu_kin
    dnk/dt = -k^2 D nkQLL
    
    Parameters
    ----------
    nTot : 1D Numpy Array (N,)
        Positive modes of state vector for total layers
    nQLL : 1D Numpy Array (N,)
        Positive modes of state vector for quasi-liquid layers
    sigMk : 1D Numpy array (N,)
        Sigma M, in k space
    params : Dictionary
             Dictionary of relevant parameters (see below)
        k : 1D Numpy array (N,), array of available wavenumbers 
        D : float, diffusion coefficient

    Returns
    -------
    dnQLL : 1D Numpy Array (N,)
        Rate of change of positive modes of nTot
    """
    
    # unpack needed params
    k = params['k']
    D = params['D']

    # define dnqll/dt rhs
    convo = convolution(nTOTk,sigMk,params)
    dnQLL = -k**2 * D * nQLLk + convo
    return dnQLL


def RHS(t,n,params):
    """
    Computes the RHS for a full KdV or ROM simulation. For use in solver.
    
    Parameters
    ----------
    t : float
        Current time
    n : Numpy array (2N,)
        Current state vector of positive modes (QLL first, then total)
    params : Dictionary
             Dictionary of relevant parameters (see below)
        N : float, number of positive modes in simulation
        Nstar : float, best fit amplitude for sinusoidal NQLL(Ntot)
        Nbar : float, best fit intercept for sinusoidal NQLL(Ntot)        
        
    Returns
    -------
    RHS : 1D Numpy array (2N,)
          Derivative of each positive mode in state vector
    """
    
    # extract parameters from dictionary
    N = params['N']
    Nstar = params['Nstar']
    Nbar = params['Nbar']
    sigma0 = params['sigma0']
    sigmaI = params['sigmaI']
    
    # unpack initial conditions from concatenated array
    nQLL = n[:N]
    nTot = n[N:]
    
    # calc sigma m and fft
    sigmaM = ds.getsigmaM(ifftnorm(nQLL),[Nbar,Nstar,sigmaI,sigma0])
    sigMk = fftnorm(sigmaM)

    # define rhs
    dnT = nTotRHS(nQLL,sigMk,params)
    dnQ = nQLLRHS(nTot,nQLL,sigMk,params)

    # print(len(dnT))
    # print(len(dnQ))
    RHS = np.concatenate((dnQ, dnT))
    return RHS

def runSim(params):
    """
    Runs a simulation of the ice continuum in Fourier space
    
    Parameters
    ----------
    params : Dictionary
             Dictionary of relevant parameters (see below)
        N : float, number of positive modes in simulation
        Nstar : float, best fit amplitude for sinusoidal NQLL(Ntot)
        Nbar : float, best fit intercept for sinusoidal NQLL(Ntot)        
        
    Returns
    -------
    uSim : ODE solver output
           Output solution from sp.integrate.solve_ivp (includes state vector at all timesteps, time vector, etc.)
    """
    
    # unpack parameters from dictionary
    N = params['N']
    ICNT = params['ICNT']
    ICNQLL = params['ICNQLL']
    tinterval = params['tinterval']
    
    # ensure initial conditions are the correct size    
    nTotIC = fftnorm(ICNT)[:N]
    nQLLIC = fftnorm(ICNQLL)[:N]
    n = np.concatenate((nQLLIC,nTotIC))
    
    # define RHS in form appropriate for solve_ivp
    def myRHS(t,y):
        out = RHS(t,y,params)
        return out

    solv = solve_ivp(fun=myRHS, t_span=[tinterval[0],tinterval[-1]], y0=n, t_eval = tinterval, rtol=1e-12, method='RK45')

    # Call the ODE solver
    ykeep_ft = solv.y
    tkeep_ft = solv.t
    return [ykeep_ft,tkeep_ft]


def makeReal(fourierSol): 
    """Inverse transforms the solution to real xt space
    
    Parameters
    ----------
    fourierSol : 2D Numpy Array (N,M)
        A 2D array containing y solutions for NQLL and NTot, respectively, at each timestep in k space
        
    Returns
    -------
    [NQLL, NTot] : 2D Numpy Array (N,M)
        A 2D array containing the y solutions for NQLL and NTot, respectively, in real xt space
    """
    N = int(fourierSol.shape[0]/2)
    timesteps = fourierSol.shape[1]
    
    NTot = np.zeros((timesteps,2*N-2))
    NQLL = np.zeros((timesteps,2*N-2))
    
    for i in range(timesteps):
        NQLL[i,:] = ifftnorm(fourierSol[0:N,i])
        NTot[i,:] = ifftnorm(fourierSol[N:,i])
        
    return [NQLL, NTot]