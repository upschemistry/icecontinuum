# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 15:01:47 2015

@author: nesh, jonathan
"""
import numpy as np
import copy

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

def convolution(nT,vKin,sigmaM,Nstar):
    """Computes Fourier transform of the nonlinear term in the QLL PDE
    
    2 pi N^* sigmaM vKin cos(Ntot)
    
    Computed in real space and then converted back
    to Fourier space.
    
    Parameters
    ----------
    nT : 1D Numpy Array (N,)
        Total water layers
        
    vKin : TBD
        TBD
        
    Nstar : TBD
        TBD

    Returns
    -------
    convo : 1D Numpy Array (N,)
        Fourier transform of the nonlinear term
    """
    
    # compute double sum in real space, then apply scalar multiplier
    convo = fftnorm(np.cos(ifftnorm(nT))
    convo = 2 * np.pi * Nstar * vKin * sigmaM * convo
    return convo

def markovKdV(u,M,alpha):
    """Computes nonlinear part of Markov term in KdV
    
    C_k(u,v) = -(alpha * 1i * k) / 2 * sum_{i+j = k} u_i v_j
    
    where the sum of i and j is over a "full" system with M positive modes (user specified)
    
    Computed in real space to avoid loops and then converted back
    to Fourier space.
    
    Parameters
    ----------
    u : 1D Numpy Array (N,)
        Positive modes of state vector whose RHS is being computed
        
    M : int
        Number of positive modes in "full" model for intermediary calculations
        
    alpha : float
        Degree of nonlinearity in KdV

    Returns
    -------
    nonlin0 : 1D Numpy Array (2*M,)
        Nonlinear part of Markov term for given state vector
        
    u_full : 1D Numpy array (2*M,)
        "full" state vector for use in later computations
    """
    
    # construct full Fourier vector from only the positive modes
    u_full = np.zeros(2*M) +1j*np.zeros(2*M)
    u_full[0:u.shape[0]] = u
    u_full[2*M-u.shape[0]+1:] = np.conj(np.flip(u[1:]))
    
    # compute the convolution sum
    nonlin0 = convolutionSumKdV(u_full,u_full,alpha)
    return nonlin0,u_full


def RHSKdV(t,u,params):
    """
    Computes the RHS for a full KdV or ROM simulation. For use in solver.
    
    Parameters
    ----------
    t : float
        Current time
        
    u : Numpy array (N,)
        Current state vector
              
    params : Dictionary
             Dictionary of relevant parameters (see below)
        N : float, number of positive modes in simulation
        M : float, number of positive modes in "full" intermediate compuation
        alpha : float, degree of nonlinearity in KdV
        epsilon : float, size of linear term (stiffness)
        tau : float, time decay modifier
        coeffs : Numpy array, renormalization coefficients for ROM (None if no ROM)

        
    Returns
    -------
    RHS : 1D Numpy array (N,)
          Derivative of each positive mode in state vector
    """
    
    # extract parameters from dictionary
    N = params['N']
    M = params['M']
    alpha = params['alpha']
    epsilon = params['epsilon']
    tau = params['tau']
    coeffs = params['coeffs']
    
    # construct wavenumber array
    k = np.concatenate([np.arange(0,M),np.arange(-M,0)])
    
    
    # Linear and Markov term
    nonlin0,u_full = markovKdV(u,M,alpha)
    RHS = 1j*k[0:N]**3*epsilon**2*u + nonlin0[0:N]
    
    if (np.any(coeffs == None)):
        order = 0
    else:
        order = coeffs.shape[0]
    
    if (order >= 1):
        # compute t-model term
        
        # define which modes are resolved / unresolved in full array
        F_modes = np.concatenate([np.arange(0,N),np.arange(2*N-1,M+N+2),np.arange(2*M-N+1,2*M)])
        G_modes = np.arange(N,2*M-N+1)
    
        # compute t-model term
        nonlin1,uuStar = tModelKdV(u_full,nonlin0,alpha,F_modes)
        RHS = RHS + coeffs[0]*nonlin1[0:N]*t**(1-tau)
        
        order = coeffs.shape[0]
    
    if (order >= 2):
        # compute t2-model term
        nonlin2,uk3,uu,A,AStar,B,BStar,C,CStar,D,DStar = t2ModelKdV(u_full,nonlin0,uuStar,alpha,F_modes,G_modes,k,epsilon)
        RHS = RHS + coeffs[1]*nonlin2[0:N]*t**(2*(1-tau))
    
    if (order >= 3):
        # compute t3-model term
        nonlin3,uk6,E,EStar,F,FStar = t3ModelKdV(alpha,F_modes,G_modes,k,epsilon,u_full,uu,uuStar,uk3,A,AStar,B,BStar,C,CStar,DStar)
        RHS = RHS + coeffs[2]*nonlin3[0:N]*t**(3*(1-tau))
    
    if (order == 4):
        # compute t4-model term
        nonlin4 = t4ModelKdV(alpha,F_modes,G_modes,k,epsilon,u_full,uu,uuStar,uk3,uk6,A,AStar,B,BStar,C,CStar,D,DStar,E,EStar,F,FStar)
        RHS = RHS + coeffs[3]*nonlin4[0:N]*t**(4*(1-tau))

    return RHS







def runSim(params):
    """
    Runs an actual ROM or non-ROM simulation of KdV
    
    Parameters
    ----------
    params : Dictionary
             Dictionary of relevant parameters (see below)
        N : float, number of positive modes in simulation
        M : float, number of positive modes in "full" intermediate compuation
        alpha : float, degree of nonlinearity in KdV
        epsilon : float, size of linear term (stiffness)
        tau : float, time decay modifier
        coeffs : Numpy array, renormalization coefficients for ROM (None if no ROM)
        IC : function handle, initial condition of simulation
        endtime : float, final time to simulate to
        timesteps: Numpy array, specific timesteps for which to save solution

        
    Returns
    -------
    uSim : ODE solver output
           Output solution from sp.integrate.solve_ivp (includes state vector at all timesteps, time vector, etc.)
    """
    
    # unpack parameters from dictionary
    N = params['N']
    IC = params['IC']
    endtime = params['endtime']
    timesteps = params['timesteps']
    
    # generate initial condition
    x = np.linspace(0,2*np.pi-2*np.pi/(2*N),2*N)
    y = IC(x)
    uFull = fftnorm(y)
    u = uFull[0:N]
    
    # define RHS in form appropriate for solve_ivp
    def myRHS(t,y):
        out = RHSKdV(t,y,params)
        return out
    
    # solve the IVP
    uSim = sp.integrate.solve_ivp(fun = myRHS, t_span = [0,endtime], y0 = u,method = "BDF", t_eval = timesteps)
    return uSim


def diffuse(y_old, diff):
    y = copy.copy(y_old) #doesn't diffuse properly if we say y = y_old
    l = len(y_old)
    for i in range(1,l-1):
        y[i] = y_old[i] + ((diff[i+1]-diff[i-1])/2)*((y_old[i+1]-y_old[i-1])/2) + diff[i]*(y_old[i+1]-2*y_old[i]+y_old[i-1])

#    # Boundary Conditions (reflection at ends)
    y[0] = y_old[0] + ((diff[1]-diff[0])/2)*((y_old[1]-y_old[0])/2)+diff[0]*(y_old[1]-2*y_old[0]+y_old[0]) #assuming second derivative for x[0] is essentially the same as  it is for x[1]
    y[l-1] = y_old[l-1] + ((diff[l-2]-diff[l-1])/2)*((y_old[l-2]-y_old[l-1])/2)+diff[l-1]*(y_old[l-2]-2*y_old[l-1]+y_old[l-1])
    return y
    
def diffuseP(y_old, diff):  # Returns the change only
    dy = np.zeros(np.shape(y_old))
    l = len(y_old)
    for i in range(1,l-1):
        dy[i] = ((diff[i+1]-diff[i-1])/2)*((y_old[i+1]-y_old[i-1])/2) + diff[i]*(y_old[i+1]-2*y_old[i]+y_old[i-1])

#    # Boundary Conditions (reflection at ends)
#    dy[0] = ((diff[1]-diff[0])/2)*((y_old[1]-y_old[0])/2)+diff[0]*(y_old[1]-2*y_old[0]+y_old[0]) 
#    dy[l-1] = ((diff[l-2]-diff[l-1])/2)*((y_old[l-2]-y_old[l-1])/2)+diff[l-1]*(y_old[l-2]-2*y_old[l-1]+y_old[l-1])

    # Boundary Conditions (periodic at ends)
    dy[0]   = ((diff[1]-diff[l-1])/2)*((y_old[1]-y_old[l-1])/2)     +diff[0]  *(y_old[1]-2*y_old[0]  +y_old[l-1]) 
    dy[l-1] = ((diff[0]-diff[l-2])/2)*((y_old[0]-y_old[l-2])/2)     +diff[l-1]*(y_old[0]-2*y_old[l-1]+y_old[l-2])

    return dy
    
def diffuseconstantD(y_old, diff):
    y = copy.copy(y_old) #doesn't diffuse properly if we say y = y_old
    l = len(y_old)
    for i in range(1,l-1):
        y[i] = y_old[i] + diff[i]*(y_old[i+1]-2*y_old[i]+y_old[i-1])
    
    # Boundary Conditions (reflection at ends)
#    y[0] = y_old[0] + diff[0]*(y_old[1]-2*y_old[0]+y_old[0]) 
#    y[l-1] = y_old[l-1] + diff[l-1]*(y_old[l-2]-2*y_old[l-1]+y_old[l-1])

    # Boundary Conditions (periodic at ends)
    y[0] = y_old[0] + diff[0]*(y_old[1]-2*y_old[0]+y_old[l-1]) 
    y[l-1] = y_old[l-1] + diff[l-1]*(y_old[0]-2*y_old[l-1]+y_old[l-2])
     
    return y

def diffuseconstantDP(y_old, diff): # Returns the change only
    l = len(y_old)
    dy = np.zeros(np.shape(y_old))
    for i in range(1,l-1):
        dy[i] = diff[i]*(y_old[i+1]-2*y_old[i]+y_old[i-1])
    
    # Boundary Conditions (reflection at ends)
#    y[0] = y_old[0] + diff[0]*(y_old[1]-2*y_old[0]+y_old[0]) 
#    y[l-1] = y_old[l-1] + diff[l-1]*(y_old[l-2]-2*y_old[l-1]+y_old[l-1])

    # Boundary Conditions (periodic at ends)
    dy[0] = diff[0]*(y_old[1]-2*y_old[0]+y_old[l-1]) 
    dy[l-1] = diff[l-1]*(y_old[0]-2*y_old[l-1]+y_old[l-2])
     
    return dy
    
def diffuseFast(Fliq_old, NIce_old, Ddt, term0, steps, Nbar, Nstar, Nmono, phi):
    # Rain    
    
    # Diffusion
    term1 = np.zeros(np.shape(Fliq_old))
    l = len(Fliq_old)
    for i in range(1,l-1):
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

    
def getNFliq(NIce,Nbar,Nstar,Nmono,phi):
    return Nbar+Nstar*np.sin(NIce/Nmono*2*np.pi-phi)

def getFliqPrime(NIce,Nbar,Nstar,Nmono,phi):
    return Nstar*np.cos(NIce/Nmono*2*np.pi-phi)*(1/Nmono*2*np.pi)

def getdeltaN(NIcep,NFliqp,Nbar,Nstar,Nmono,phi):
    deltaN = 0.0
    for i in range(10):
        deltaN = Nbar+Nstar*np.sin((NIcep-deltaN)/Nmono*2*np.pi-phi)-NFliqp
        #deltaN = getNFliq(NIcep-deltaN,Nbar,Nstar,Nmono,phi)-NFliqp
    return deltaN

def fqll_next(fqll_last,Ntot,Nstar,Nbar):
    fstar = Nstar/Nbar
    return 1 + fstar*np.sin(2*np.pi*(Ntot-Nbar*fqll_last))

def getNiceoffset(Nbar=None, Nstar=None, Nmono=None, phi=None):
    # to see the plots, use the getNiceoffset that is commented out
    #get the response curve
    Nicetest = np.linspace(0,1)
    Fliqtest = getNFliq(Nicetest,Nbar,Nstar,Nmono,phi)
    Imin = np.argmin(Fliqtest)
    return Nicetest[Imin]
    
def getNliq(Ntot,Nstar,Nbar,niter):
    fqll_last = 1.0
    for i in range(niter):
        fqll_last = fqll_next(fqll_last,Ntot,Nstar,Nbar)
    return fqll_last*Nbar

def fqllprime_next(fqll_last,Ntot,Nstar,Nbar):
    fstar = Nstar/Nbar
    return 1 + fstar*np.sin(2*np.pi*(Ntot-Nbar*fqll_last))

def getNliqprime(Ntot,Nstar,Nbar,niter):
    f1 = getNliq(Ntot,Nstar,Nbar,niter)
    f2 = getNliq(Ntot+.01,Nstar,Nbar,niter)
    return (f2-f1)/.01

def getdNliq_dNtot(Ntot,Nstar,Nbar,niter):
    dfqll_dNtot_last = 0.0
    fqll_last = 1.0
    for i in range(niter):
        dfqll_dNtot_last = getdfqll_dNtot_next(dfqll_dNtot_last,fqll_last,Ntot,Nstar,Nbar)
        fqll_last = fqll_next(fqll_last,Ntot,Nstar,Nbar)
    return dfqll_dNtot_last*Nbar
        
def getdfqll_dNtot_next(dfqll_dNtot_last,fqll_last,Ntot,Nstar,Nbar):
    fstar = Nstar/Nbar
    return fstar*np.cos(2*np.pi*(Ntot-fqll_last))*2*np.pi*(1-Nbar*dfqll_dNtot_last)
    
def f0d(y, t, params):
    Nbar, Nstar, niter, sigmastepmax, sigma0, deprate = params  # unpack parameters
    
    Fliq0, Ntot0 = y      # unpack current values of y
    
    delta = (Fliq0 - (Nbar - Nstar))/(2*Nstar)
    sigD = (sigmastepmax - delta * sigma0)/(1+delta*sigma0)
    depsurf = deprate * sigD

    #dFliq0_dt = getNliqprime(Ntot0,Nstar,Nbar,niter)*depsurf
    dFliq0_dt = getdNliq_dNtot(Ntot0,Nstar,Nbar,niter)*depsurf
    dNtot_dt = depsurf
    
    derivs = [dFliq0_dt, dNtot_dt]
    return derivs

def f1d(y, t, params):
    Nbar, Nstar, niter, sigmastep, sigma0, deprate, DoverdeltaX2, nx = params  # unpack parameters
    Fliq0, Ntot0 = np.reshape(y,(2,nx))      # unpack current values of y
    #print np.shape(Fliq0); print Fliq0[0:5]
    #print np.shape(Ntot0); print Ntot0[0:5]
    
    # Deposition
    delta = (Fliq0 - (Nbar - Nstar))/(2*Nstar)
    sigD = (sigmastep - delta * sigma0)/(1+delta*sigma0)
    depsurf = deprate * sigD
    #dFliq0_dt = getNliqprime(Ntot0,Nstar,Nbar,niter)*depsurf
    dFliq0_dt = getdNliq_dNtot(Ntot0,Nstar,Nbar,niter)*depsurf
    dNtot_dt = depsurf

    # Diffusion
    l = len(Fliq0)
    dy = np.zeros(np.shape(Fliq0))
    for i in range(1,l-1):
        dy[i] = DoverdeltaX2*(Fliq0[i+1]-2*Fliq0[i]+Fliq0[i-1])
    
        # Boundary Conditions (periodic at ends)
        dy[0] = DoverdeltaX2*(Fliq0[1]-2*Fliq0[0]+Fliq0[l-1]) 
        dy[l-1] = DoverdeltaX2*(Fliq0[0]-2*Fliq0[l-1]+Fliq0[l-2])
     
    # Combined
    dFliq0_dt += dy
    dNtot_dt += dy

    # Package for output
    derivs = list([dFliq0_dt, dNtot_dt])
    derivs = np.reshape(derivs,2*nx)
    return derivs

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

def getsigmastep(x,xmax,center_reduction,sigmastepmax,method='sinusoid'):
    sigmapfac = 1-center_reduction/100
    xmid = max(x)/2
    if method == 'sinusoid':
        fsig = (np.cos(x/xmax*np.pi*2)+1)/2*(1-sigmapfac)+sigmapfac
    elif method == 'parabolic':
        fsig = (x-xmid)**2/xmid**2*(1-sigmapfac)+sigmapfac
    else:
        print('bad method')
    return fsig*sigmastepmax
        

