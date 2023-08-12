
"""
Created on Mon Jul 20 2023

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


def nQLLRHS(nQLLk,sigMk,params):
    """Computes RHS of Nqll in k space

    Parameters
    ----------
    nQLLk : 1D Numpy array (N,)
            Thickness of quasiliquid water layer, in k space
    sigMk : 1D Numpy array (N,)
            Sigma M, in k space
    params : Dictionary
             Dictionary of relevant parameters (see below)
        nu_kin_mlyperus : float, speed of water vapor hitting qll layer in monolayers per microsecond
        k : 1D Numpy array (N,), array of available wavenumbers
        D : float, diffusion coefficient
        Nstar : float, best fit amplitude for sinusoidal NQLL(Ntot)
        Nbar : float, best fit intercept for sinusoidal NQLL(Ntot)        
    
    Returns
    -------
    dnQLL : 1D Numpy array (N,)
        Array containing derivative of nQLL wrt t, in k space
    """

    # unpack necessary params
    D = params['D']
    nu_kin_mlyperus = params['nu_kin_mlyperus']
    Nstar = params['Nstar']
    Nbar = params['Nbar']
    k = params['k']

    # get real stuff for comparisons... not a fan of forcing conditions but unable to figure something else out thus far
    Nqll = ifftnorm(nQLLk)
    convReal = ifftnorm(sigMk) * (np.sqrt(Nstar**2 - (Nbar - Nqll)**2))##(np.cos(np.arcsin((Nbar - ifftnorm(nQLLk))/Nstar)))) * Nstar
    
    n = int(len(Nqll)/2)
    for i in range (1,n-1):
        if Nqll[i-1] - Nqll[i] > 0.0000000001:
            convReal[i] = -convReal[i]
    for i in range (n+1,2*n-1):
        if Nqll[i] - Nqll[i+1] < -0.0000000001:
            convReal[i] = -convReal[i]
    # Correct center points to follow surrounding points
    for i in range (n-1, n+1):
        if convReal[i-1] <= 0 and convReal[i+3] <= 0:
            convReal[i] = -convReal[i]
    # Correct endpoints to follow leading points
    # Index 0
    if Nqll[2] - Nqll[1] > 0 and Nqll[0] > Nqll[1]: #increasing slope, initial point is bigger than next
        convReal[0] = -convReal[0]
    if Nqll[2] - Nqll[1] < 0: #decreasing slope
        convReal[0] = -convReal[0]
    # Index -1
    if Nqll[-3] - Nqll[-2] > 0 and Nqll[-1] > Nqll[-2]: #decreasing slope, final point is bigger than last
        convReal[-1] = -convReal[-1]
    if Nqll[-3] - Nqll[-2] < 0: #increasing slope
        convReal[-1] = -convReal[-1]    
        
    # Add to transformed diffusion
    dnQLL = - k**2 * D*nQLLk + 2*np.pi*nu_kin_mlyperus * fftnorm(convReal)
    return dnQLL
    

def RHS(t,n,params):
    """Computes the RHS for the 1 variable system. For use in solver.
    
    Parameters
    ----------
    t : float
        Current time, for use in solver
    n : Numpy array (N,)
        Current state vector of positive modes (only total)   
    params : Dictionary
             Dictionary of relevant parameters (see below)
        Nstar : float, best fit amplitude for sinusoidal NQLL(Ntot)
        Nbar : float, best fit intercept for sinusoidal NQLL(Ntot)

    Returns
    -------
    RHS : 1D Numpy array (N,)
          Derivative of each positive mode in state vector
    """
    
    # find NQLL in real space
    nQLL = n
    Nqll = ifftnorm(nQLL)
    
    # calc sigma M and transform
    sigmaM = ds.getsigmaM(Nqll,params)
    sigMk = fftnorm(sigmaM)
    
    # calc dnQLL/dt in k space and return as array
    dnQ = nQLLRHS(nQLL,sigMk,params)
    RHS = np.array(dnQ)
    return RHS


def runSim(params):
    """Runs a simulation of the ice continuum in Fourier space
    
    Parameters
    ----------
    params : Dictionary
             Dictionary of relevant parameters (see below)
        N : float, number of positive modes in simulation
        nu_kin_mlyperus : float, speed of water vapor hitting qll layer in monolayers per microsecond
        sigma0 : 
        sigmaI : 
        k : 1D Numpy array (N,), array of available wavenumbers
        D : float, diffusion coefficient
        Nstar : float, best fit amplitude for sinusoidal NQLL(Ntot)
        Nbar : float, best fit intercept for sinusoidal NQLL(Ntot)
        tinterval : 1D Numpy array (N,), timesteps for solver
        ICNT : 1D Numpy array (N,), initial total thickness of qll and ice layers
        
    Returns
    -------
    uSim : ODE solver output
           Output solution from sp.integrate.solve_ivp with method RK45 (includes state vector at all timesteps, time vector, etc.)
    """
    
    # unpack parameters from dictionary
    N = params['N']
    ICNT = params['ICNT']
    tinterval = params['tinterval']
    Nstar = params['Nstar']
    Nbar = params['Nbar']
    
    nQLLIC = fftnorm(ds.getNQLL(ICNT,Nstar,Nbar))[:N]
    n = np.array(nQLLIC)
    
    # define RHS in form appropriate for solve_ivp
    def myRHS(t,y):
        out = RHS(t,y,params)
        return out

    # Call the ODE solver
    solv = solve_ivp(fun=myRHS, t_span=[tinterval[0],tinterval[-1]], y0=n, t_eval = tinterval, rtol=1e-12, method='RK45')
    ykeep_ft = solv.y
    tkeep_ft = solv.t
    return [ykeep_ft,tkeep_ft]


def makeReal(fourierSol,params): 
    """Transforms the solution to the PDE back into real x space

    Parameters
    ----------
    fourierSol : 2D Numpy array (2,N)
        Array containing y solutions to RHS at position 0 and timesteps at position 1
    params : Dictionary
             Dictionary of relevant parameters (see below)
        N : float, number of positive modes in simulation
        nu_kin_mlyperus : float, speed of water vapor hitting qll layer in monolayers per microsecond (unused explicitly here)
        sigma0 : (unused explicitly here)
        sigmaI : (unused explicitly here)
        k : 1D Numpy array (N,), array of available wavenumbers (unused explicitly here)
        D : float, diffusion coefficient (unused explicitly here)
        Nstar : float, best fit amplitude for sinusoidal NQLL(Ntot) (unused explicity here)
        Nbar : float, best fit intercept for sinusoidal NQLL(Ntot) (unused explicitly here)
        tinterval : 1D Numpy array (N,), timesteps for solver (unused explicitly here)
        ICNT : 1D Numpy array (N,), initial total thickness of qll and ice layers (unused explicitly here)
    
    Returns
    -------
    Nqll : 2D Numpy array (somesize)
        Array containing thickness of the quasiliquid layer, in real xt space
    """

    N = params['N']
    timesteps = fourierSol.shape[1]

    Nqll = np.zeros((timesteps,2*N-2))
    for i in range(timesteps):
        Nqll[i,:] = ifftnorm(fourierSol[:,i])        
    return Nqll