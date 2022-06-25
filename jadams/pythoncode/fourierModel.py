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
    
    N = u_full.shape[0]
    normalizedFFT = np.fft.fft(u_full)*1/N
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
    
    N = u_full.shape[0]
    normalizedIFFT = np.real(np.fft.ifft(u_full)*N)
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
    
    # construct full Fourier vector from only the positive modes
    nTotFull = np.zeros(2*N) + 1j*np.zeros(2*N)
    nTotFull[0:N] = nTot
    nTotFull[N+1:] = np.conj(np.flip(nTot[1:]))
    
    Ffull = fftnorm(sigmastep*ifftnorm(nTotFull)[0:N])
    
    F = Ffull[0:N]

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
    
    nTot = n[0:N]
    nQLL = n[N:]
    
    
    dnT = nTotRHS(nQLL,nu_kin,sigmastep_FFT,k,D)
    dnQ = nQLLRHS(nTot,nQLL,nu_kin,sigmastep,k,D)
    
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

""" 
#below is taken from diffusionstuff6: updated to numba version by Max for speed
@njit
def diffuse(y_old, diff):
    y = copy.copy(y_old) #doesn't diffuse properly if we say y = y_old
    l = len(y_old)
    for i in prange(1,l-1):
        y[i] = y_old[i] + ((diff[i+1]-diff[i-1])/2)*((y_old[i+1]-y_old[i-1])/2) + diff[i]*(y_old[i+1]-2*y_old[i]+y_old[i-1])
    # Boundary Conditions (reflection at ends)
    y[0] = y_old[0] + ((diff[1]-diff[0])/2)*((y_old[1]-y_old[0])/2)+diff[0]*(y_old[1]-2*y_old[0]+y_old[0]) #assuming second derivative for x[0] is essentially the same as  it is for x[1]
    y[l-1] = y_old[l-1] + ((diff[l-2]-diff[l-1])/2)*((y_old[l-2]-y_old[l-1])/2)+diff[l-1]*(y_old[l-2]-2*y_old[l-1]+y_old[l-1])
    return y

@njit
def diffuseP(y_old, diff):  # Returns the change only
    dy = np.zeros(np.shape(y_old))
    l = len(y_old)
    for i in prange(1,l-1):
        dy[i] = ((diff[i+1]-diff[i-1])/2)*((y_old[i+1]-y_old[i-1])/2) + diff[i]*(y_old[i+1]-2*y_old[i]+y_old[i-1])

    # Boundary Conditions (reflection at ends)
#    dy[0] = ((diff[1]-diff[0])/2)*((y_old[1]-y_old[0])/2)+diff[0]*(y_old[1]-2*y_old[0]+y_old[0]) 
#    dy[l-1] = ((diff[l-2]-diff[l-1])/2)*((y_old[l-2]-y_old[l-1])/2)+diff[l-1]*(y_old[l-2]-2*y_old[l-1]+y_old[l-1])

    # Boundary Conditions (periodic at ends)
    dy[0]   = ((diff[1]-diff[l-1])/2)*((y_old[1]-y_old[l-1])/2)     +diff[0]  *(y_old[1]-2*y_old[0]  +y_old[l-1]) 
    dy[l-1] = ((diff[0]-diff[l-2])/2)*((y_old[0]-y_old[l-2])/2)     +diff[l-1]*(y_old[0]-2*y_old[l-1]+y_old[l-2])

    return dy

@njit
def diffuseconstantD(y_old, diff):
    y = copy.copy(y_old) #doesn't diffuse properly if we say y = y_old
    l = len(y_old)
    for i in prange(1,l-1):
        y[i] = y_old[i] + diff[i]*(y_old[i+1]-2*y_old[i]+y_old[i-1])
    
    # Boundary Conditions (reflection at ends)
#    y[0] = y_old[0] + diff[0]*(y_old[1]-2*y_old[0]+y_old[0]) 
#    y[l-1] = y_old[l-1] + diff[l-1]*(y_old[l-2]-2*y_old[l-1]+y_old[l-1])

    # Boundary Conditions (periodic at ends)
    y[0] = y_old[0] + diff[0]*(y_old[1]-2*y_old[0]+y_old[l-1]) 
    y[l-1] = y_old[l-1] + diff[l-1]*(y_old[0]-2*y_old[l-1]+y_old[l-2])
     
    return y

@njit
def diffuseconstantDP(y_old, diff): # Returns the change only
    l = len(y_old)
    dy = np.zeros(np.shape(y_old))
    for i in prange(1,l-1):
        dy[i] = diff[i]*(y_old[i+1]-2*y_old[i]+y_old[i-1])
    
    # Boundary Conditions (reflection at ends)
#    y[0] = y_old[0] + diff[0]*(y_old[1]-2*y_old[0]+y_old[0]) 
#    y[l-1] = y_old[l-1] + diff[l-1]*(y_old[l-2]-2*y_old[l-1]+y_old[l-1])

    # Boundary Conditions (periodic at ends)
    dy[0] = diff[0]*(y_old[1]-2*y_old[0]+y_old[l-1]) 
    dy[l-1] = diff[l-1]*(y_old[0]-2*y_old[l-1]+y_old[l-2])
     
    return dy

@njit
def diffuseFast(Fliq_old, NIce_old, Ddt, term0, steps, Nbar, Nstar, Nmono, phi):
    # Rain    
    
    # Diffusion
    term1 = np.zeros(np.shape(Fliq_old))
    l = len(Fliq_old)
    for i in prange(1,l-1):
        term1[i] = ((Ddt[i+1]-Ddt[i-1])/2)*((Fliq_old[i+1]-Fliq_old[i-1])/2) + Ddt[i]*(Fliq_old[i+1]-2*Fliq_old[i]+Fliq_old[i-1])
    # Boundary Conditions 
    term1[0] = ((Ddt[1]-Ddt[0])/2)*((Fliq_old[1]-Fliq_old[0])/2)+Ddt[0]*(Fliq_old[1]-2*Fliq_old[0]+Fliq_old[0]) #assuming second derivative for x[0] is essentially the same as  it is for x[1]
    term1[l-1] = ((Ddt[l-2]-Ddt[l-1])/2)*((Fliq_old[l-2]-Fliq_old[l-1])/2)+Ddt[l-1]*(Fliq_old[l-2]-2*Fliq_old[l-1]+Fliq_old[l-1])
    
    # Perturbation due to rain and diffusion    
    term2 = term1 + term0
    
    # Conversion to ice
    F1Prime = getFliqPrime(NIce_old,Nbar,Nstar,Nmono,phi)
    dFliqdt = term2*F1Prime/(1+F1Prime)
    dNIcedt = term2 - dFliqdt
    Fliq = Fliq_old + dFliqdt
    NIce = NIce_old + dNIcedt
    return Fliq, NIce

@njit
def getNFliq(NIce,Nbar,Nstar,Nmono,phi):
    return Nbar+Nstar*np.sin(NIce/Nmono*2*np.pi-phi)

@njit
def getFliqPrime(NIce,Nbar,Nstar,Nmono,phi):
    return Nstar*np.cos(NIce/Nmono*2*np.pi-phi)*(1/Nmono*2*np.pi)

@njit
def getdeltaN(NIcep,NFliqp,Nbar,Nstar,Nmono,phi):
    deltaN = 0.0
    for i in range(10):
        deltaN = Nbar+Nstar*np.sin((NIcep-deltaN)/Nmono*2*np.pi-phi)-NFliqp
        #deltaN = getNFliq(NIcep-deltaN,Nbar,Nstar,Nmono,phi)-NFliqp
    return deltaN

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

@njit
def getNiceoffset(Nbar=None, Nstar=None, Nmono=None, phi=None):
    # to see the plots, use the getNiceoffset that is commented out
    #get the response curve
    Nicetest = np.linspace(0,1)
    Fliqtest = getNFliq(Nicetest,Nbar,Nstar,Nmono,phi)
    Imin = np.argmin(Fliqtest)
    return Nicetest[Imin]

@njit("f8[:](f8[:],f8,f8,i8)") #Ntot is ndarray of numbers (ints, become floats), Nstar and Nbar are floats, niter is an int literal
def getNliq_array(Ntot,Nstar,Nbar,niter):
    fqll_last = np.array([1.0])
    for i in range(niter):
        fqll_last = fqll_next_array(fqll_last,Ntot,Nstar,Nbar)
    return fqll_last*Nbar

@njit("f8(f8,f8,f8,i8)") #Ntot is ndarray of numbers (ints, become floats), Nstar and Nbar are floats, niter is an int literal
def getNliq(Ntot,Nstar,Nbar,niter):
    fqll_last = 1.0
    for i in range(niter):
        fqll_last = fqll_next(fqll_last,Ntot,Nstar,Nbar)
    return fqll_last*Nbar

@njit("f8[:](f8[:],f8[:],f8,f8)")
def fqllprime_next(fqll_last,Ntot,Nstar,Nbar):
    fstar = Nstar/Nbar
    return 1 + fstar*np.sin(2*np.pi*(Ntot-Nbar*fqll_last))

@njit#((types.float64(types.float64[:],types.float64,types.float64,types.int_)))
def getNliqprime(Ntot,Nstar,Nbar,niter):
    f1 = getNliq(Ntot,Nstar,Nbar,niter)
    f2 = getNliq(Ntot+.01,Nstar,Nbar,niter)
    return (f2-f1)/.01

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

@njit("f8[:](f8[:],f8,f8[:],i4[:],f8[:])")#, parallel = True) # in the current use case it is faster without paralellization
def f1d(y, t, float_params, int_params, sigmastep): #sigmastep is an array
     # unpack parameters
    Nbar, Nstar, sigma0, deprate, DoverdeltaX2 = float_params 
    niter, nx = int_params

    # unpack current values of y
    Fliq0, Ntot0 = np.reshape(np.ascontiguousarray(y),(2,500))
    
    # Deposition
    delta = (Fliq0 - (Nbar - Nstar))/(2*Nstar)
    sigD = (sigmastep - delta * sigma0)/(1+delta*sigma0)
    depsurf = deprate * sigD
    dFliq0_dt = getdNliq_dNtot_array(Ntot0,Nstar,Nbar,niter)*depsurf
    dNtot_dt = depsurf

    # Diffusion
    l = len(Fliq0)
    dy = np.zeros((l,))#np.shape(Fliq0))
    for i in prange(0,l):#(1,l-1):
        dy[i] = DoverdeltaX2*(Fliq0[i+1]-2*Fliq0[i]+Fliq0[i-1])
        # Boundary Conditions (periodic at ends)
        dy[0] = DoverdeltaX2*(Fliq0[1]-2*Fliq0[0]+Fliq0[l-1]) 
        dy[l-1] = DoverdeltaX2*(Fliq0[0]-2*Fliq0[l-1]+Fliq0[l-2])
     
    # Combined
    dFliq0_dt += dy
    dNtot_dt += dy

    # Package for output
    #derivs = np.reshape([dFliq0_dt, dNtot_dt],2*nx)
    derivs = np.concatenate((dFliq0_dt,dNtot_dt))
    return derivs

@njit
def f1dflux(Fliq0, Ntot0, dt, params):
    Nbar, Nstar, niter, sigmastep, sigma0, deprate, DoverdeltaX2, nx = params  # unpack parameters
    
    # Deposition
    delta = (Fliq0 - (Nbar - Nstar))/(2*Nstar)
    sigD = (sigmastep - delta * sigma0)/(1+delta*sigma0)
    depsurf = deprate * sigD
    dFliq0 = getdNliq_dNtot(Ntot0,Nstar,Nbar,niter)*depsurf*dt
    Fliq1 = Fliq0 + dFliq0
     
    # Package for output
    return Fliq1

@njit
def getsigmastep(x,xmax,center_reduction,sigmastepmax,method='sinusoid'):
    sigmapfac = 1-center_reduction/100
    xmid = max(x)/2
    if method == 'sinusoid':
        fsig = (np.cos(x/xmax*np.pi*2)+1)/2*(1-sigmapfac)+sigmapfac
    elif method == 'parabolic':
        fsig = (x-xmid)**2/xmid**2*(1-sigmapfac)+sigmapfac
    else:
        print('bad method')
    return fsig*sigmastepmax """