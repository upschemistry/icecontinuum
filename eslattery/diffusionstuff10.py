# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 15:01:47 2015
@author: nesh, jonathan
@author: ella (as of 2023)
"""
import numpy as np
import copy
    
def getNQLL(Ntot,Nstar,Nbar):
    """Calculates NQLL at a given Ntot

    Parameters
    ----------
    Ntot :  1D Numpy Array (N,)
            Array containing total thickness of layers
    Nstar : float
            Best fit parameter to match amplitude of NQLL(Ntot)
    Nbar :  float
            Best fit parameter to match interval of NQLL(Ntot)

    Returns
    -------
    NQLL : 1D Numpy Array (N,)
        Array containing the thickness of the qll layer at a given Ntot
    """
    
    return Nbar - Nstar*np.sin(2*np.pi*Ntot)

    
def getsigmaM(N,params,isNqll=True):
    """Calculates sigma M from NQLL or Ntot

    N : 1D Numpy Array
        Array containing values of NQLL or Ntot, determined by boolean isNqll
    params : Dictionary 
             Dictionary containing relevant parameters (see below)
        Nbar : float, best fit parameter to match intercept of NQLL(Ntot)
        Nstar : float, best fit parameter to match amplitude of NQLL(Ntot)
        sigmaI : 1D Numpy Array (N,), TODO
        sigma0 : 1D Numpy Array (N,), TODO
    
    Returns
    -------
    sigmaM : 1D Numpy Array (N,)
        TODO
    """
    
    # unpack params
    Nbar = params['Nbar']
    Nstar = params['Nstar']
    sigmaI = params['sigmaI']
    sigma0 = params['sigma0']

    # determine Nqll
    if isNqll: # N is Nqll
        Nqll = N
    else: # N is Ntot
        Nqll = getNQLL(N,Nstar,Nbar)

    m = (Nqll - (Nbar - Nstar))/(2*Nstar)    
    return (sigmaI - m * sigma0)/(1+m*sigma0)
    
def f1d_sigma_m(y, t, params):
    """Calculates sigma M for 2 variable system??
    
    Parameters
    ----------
    y : 1D Numpy Array
        TODO
    t : float
        Only used in integrator

    Returns
    -------
    sigmaM : 1D Numpy Array
        TODO
    """
    
    Nbar, Nstar, sigmaI, sigma0, nu_kin_mlyperus, Doverdeltax2, nx = params
    NQLL0, Ntot0 = np.reshape(y,(2,nx))      # unpack current values of y
    
    # Deposition
    m = (NQLL0 - (Nbar - Nstar))/(2*Nstar)
    sigma_m = (sigmaI - m * sigma0)/(1+m*sigma0)
    return sigma_m

def getsigmaI(x,xmax,center_reduction,sigmaIcorner,method='sinusoid'):
    """Calculates sigma I using the passed method, either sinusoid or parabolic
    
    Parameters
    ----------
    x : 1D Numpy Array
            TODO
    xmax : float
           TODO
    center_reduction : float
            TODO
    sigmaIcorner : float
            TODO
    method : str
             Either sinusoid or parabolic to match the deposition"""
    
    sigmapfac = 1-center_reduction/100
    xmid = max(x)/2
    if method == 'sinusoid':
        fsig = (np.cos(x/xmax*np.pi*2)+1)/2*(1-sigmapfac)+sigmapfac
    elif method == 'parabolic':
        fsig = (x-xmid)**2/xmid**2*(1-sigmapfac)+sigmapfac
    else:
        print('bad method')
    return fsig*sigmaIcorner



def f0d(y, t, myparams):
    """0D 2 variable simulation

    Parameters
    ----------
    y : 1D Numpy Array
        Array containing thickness of the quasiliquid layer and total thickness, respectively
    t : float
        Only for use in integrator
    myparams : Dictionary
            Dictionary containing relevant parameters (see below)
        Nbar : Best fit parameter to match intercept of NQLL(Ntot)
        Nstar : float, best fit parameter to match amplitude of NQLL(Ntot)
        nu_kin_mlyperus : float, deposition rate in monolayers per microseconds
    
    Returns
    -------
    derivs : 2D Numpy Array (N,M)
        Array containing derivatives wrt t of NQLL and Ntot, respectively
    """

    Nbar = myparams['Nbar']
    Nstar = myparams['Nstar']
    nu_kin_mlyperus = myparams['nu_kin_mlyperus']

    NQLL0 = y[0]
    Ntot0 = y[1]      # unpack current values of y

    # Deposition
    twopi = 2*np.pi
    # sigma_m = getsigmaM(NQLL0,[Nbar,Nstar,sigmaI,sigma0])
    sigma_m = getsigmaM(NQLL0,myparams)
    depsurf = nu_kin_mlyperus * sigma_m

    dNQLL_dt = -depsurf*Nstar/Nbar*np.cos(twopi*Ntot0)*twopi
    dNtot_dt =  depsurf
    derivs = [dNQLL_dt, dNtot_dt]
    return derivs


def f0d_1var(y, t, params):
    """0D 1 variable simulation

    Parameters
    ----------
    y : 1D Numpy Array
        Array containing initial thickness of Ntot layer
    t : float
        Only used in the integrator
    params : Dictionary
             Dictionary containing all relevant parameters (see below)
        Nbar : float, best fit parameter to match intercept of NQLL(Ntot)
        Nstar : float, best fit parameter to match amplitude of NQLL(Ntot)
        nu_kin_mlyperus : float, deposition rate in monolayers per microsecond

    Returns
    -------
    dNtot_dt : 1D Numpy Array (N,)
        Derivatives of Ntot wrt t 
    """

    Nbar = params['Nbar']
    Nstar = params['Nstar']
    nu_kin_mlyperus = params['nu_kin_mlyperus']

    Ntot0 = y      # unpack current value of y
    NQLL0 = getNQLL(Ntot0,Nstar,Nbar)

    # Deposition
    sigma_m = getsigmaM(NQLL0,params)
    dNtot_dt = nu_kin_mlyperus * sigma_m
    return dNtot_dt

def f0d_solve_ivp(t, y, myparams):
    """0D 2 variable simulation, with signature formatted for solve_ivp integrator. See f0d(y,t,myparams)
    """
    
    return f0d(y,t,myparams)

def f0d_solve_ivp_1var(t, y, myparams):
    """0D 1 variable simulation, with signature formatted for solve_ivp integrator. See f0d_1var(y,t,myparams)
    """
    
    return f0d_1var(y,t,myparams)

def f1d(y, t, params):
    """1D 2 variable simulation

    Parameters
    ----------
    y : 1D Numpy Array
        Array containing initial thickness of NQLL and Ntot layers, respectively
    t : float
        Only used in the integrator
    params : Dictionary
             Dictionary containing all relevant parameters (see below)
        Nbar : float, best fit parameter to match intercept of NQLL(Ntot)
        Nstar : float, best fit parameter to match amplitude of NQLL(Ntot)
        nu_kin_mlyperus : float, deposition rate in monolayers per microsecond
        Doverdeltax2 : float, numerically calculated D nabla (diffusion coefficient / x^2)
        nx : int, discritization of x, NQLL, and Ntot arrays

    Returns
    -------
    derivs : 1D Numpy Array (N,)
        Derivatives of NQLL and Ntot wrt t, respectively
    """
    
    Nbar = params['Nbar']
    Nstar = params['Nstar']
    nu_kin_mlyperus = params['nu_kin_mlyperus']
    Doverdeltax2 = params['Doverdeltax2']
    nx = params['nx']

    NQLL0, Ntot0 = np.reshape(y,(2,nx))      # unpack current values of y
    
    # Deposition
    twopi = 2*np.pi
    sigma_m = getsigmaM(NQLL0,params)
    depsurf = nu_kin_mlyperus * sigma_m

    dNQLL_dt = -depsurf*Nstar*twopi/Nbar*np.cos(twopi*Ntot0)
    dNtot_dt =  depsurf

    # Diffusion
    dy = np.zeros(np.shape(NQLL0))
    for i in range(1,len(NQLL0)-1):
        dy[i] = Doverdeltax2*(NQLL0[i-1]-2*NQLL0[i]+NQLL0[i+1])
    dy[0]  = Doverdeltax2*(NQLL0[-1] -2*NQLL0[0] +NQLL0[1]) 
    dy[-1] = Doverdeltax2*(NQLL0[-2] -2*NQLL0[-1]+NQLL0[0])
     
    dNtot_dt += dy
    dNQLL_dt += dy

    # Package for output
    derivs = list([dNQLL_dt, dNtot_dt])
    derivs = np.reshape(derivs,2*nx)
    return derivs

def f1d_1var(y, t, params):
    """1D 1 variable simulation in terms of Ntot

    Parameters
    ----------
    y : 1D Numpy Array
        Array containing initial thickness of Ntot layer
    t : float
        Only used in the integrator
    params : Dictionary
             Dictionary containing all relevant parameters (see below)
        Nbar : float, best fit parameter to match intercept of NQLL(Ntot)
        Nstar : float, best fit parameter to match amplitude of NQLL(Ntot)
        nu_kin_mlyperus : float, deposition rate in monolayers per microsecond
        Doverdeltax2 : float, numerically calculated D nabla (diffusion coefficient / x^2)

    Returns
    -------
    derivs : 1D Numpy Array (N,)
        Derivatives of Ntot wrt t, respectively
    """
    
    Nbar = params['Nbar']
    Nstar = params['Nstar']
    nu_kin_mlyperus = params['nu_kin_mlyperus']
    Doverdeltax2 = params['Doverdeltax2']
 
    Ntot0 = y      # unpack current value of y
    NQLL0 = getNQLL(Ntot0,Nstar,Nbar)

    # Deposition
    sigma_m = getsigmaM(NQLL0,params)
    dNtot_dt = nu_kin_mlyperus * sigma_m

    # Diffusion
    dy = np.zeros(np.shape(NQLL0))
    for i in range(1,len(NQLL0)-1):
        dy[i] = Doverdeltax2*(NQLL0[i-1]-2*NQLL0[i]+NQLL0[i+1])
    dy[0]  = Doverdeltax2*(NQLL0[-1] -2*NQLL0[0] +NQLL0[1]) 
    dy[-1] = Doverdeltax2*(NQLL0[-2] -2*NQLL0[-1]+NQLL0[0])
    dNtot_dt += dy

    # Package for output
    derivs = list(dNtot_dt)
    return derivs

def f1d_solve_ivp(t, y, params):
    """1D 2 variable simulation, with signature formatted for solve_ivp integrator. See f1d(y,t,myparams)
    """

    return f1d(y,t,params)

def f1d_solve_ivp_1var(t, y, params):
    """1D 1 variable simulation, with signature formatted for solve_ivp integrator. See f1d_1var(y,t,myparams)
    """

    return f1d_1var(y,t,params)



def f1d_solve_ivp_1var_QLL(t,y,params):
    """1D 1 variable simulation in terms of NQLL

    Parameters
    ----------
    y : 1D Numpy Array
        Array containing initial thickness of NQLL layers, respectively
    t : float
        Only used in the integrator
    params : Dictionary
             Dictionary containing all relevant parameters (see below)
        Nbar : float, best fit parameter to match intercept of NQLL(Ntot)
        Nstar : float, best fit parameter to match amplitude of NQLL(Ntot)
        nu_kin_mlyperus : float, deposition rate in monolayers per microsecond
        Doverdeltax2 : float, numerically calculated D nabla (diffusion coefficient / x^2)
        # nx : int, discritization of x, NQLL, and Ntot arrays

    Returns
    -------
    derivs : 1D Numpy Array (N,)
        Derivatives of NQLL wrt t
    """
    
    Nbar = params['Nbar']
    Nstar = params['Nstar']
    nu_kin_mlyperus = params['nu_kin_mlyperus']
    Doverdeltax2 = params['Doverdeltax2']
    
    Nqll0 = y      # unpack current value of y
    dNqll = np.zeros(len(Nqll0))

    # Deposition
    sigma_m = getsigmaM(Nqll0,params)
    dNqll = 2*np.pi*nu_kin_mlyperus*sigma_m * np.sqrt(Nstar**2 - (Nbar - Nqll0)**2)##Nstar*(np.cos(np.arcsin((Nbar - Nqll0)/Nstar))) ##THIS IS WRONG FIX IT
    
    # Correct arcsin limitations
    for i in range (0,int(len(dNqll)/2)-1):
        if Nqll0[i-1] - Nqll0[i] > 0:
            dNqll[i] = -dNqll[i]
    for i in range (int(len(dNqll)/2)+1,len(dNqll)-1):
        if Nqll0[i-1] - Nqll0[i] < 0:
            dNqll[i] = -dNqll[i]
    # Correct center point to follow surrounding points
    n = int(len(dNqll)/2)
    for i in range (n-1, n+1):
        if dNqll[i-1] < 0 and dNqll[i+3] < 0:
            dNqll[i] = -dNqll[i]

    # Diffusion
    dy = np.zeros(np.shape(Nqll0))
    for i in range(1,len(Nqll0)-1):
        dy[i] = Doverdeltax2*(Nqll0[i-1]-2*Nqll0[i]+Nqll0[i+1])
    dy[0]  = Doverdeltax2*(Nqll0[-1] -2*Nqll0[0] +Nqll0[1]) 
    dy[-1] = Doverdeltax2*(Nqll0[-2] -2*Nqll0[-1]+Nqll0[0])
    dNqll += dy

    # Correct endpoints
    if Nqll0[2] - Nqll0[1] > 0 and Nqll0[0] > Nqll0[1]: #increasing slope, initial point is bigger than next
        dNqll[0] = -dNqll[0]
    if Nqll0[2] - Nqll0[1] < 0:# and dNqll[0] > 0:# and dNqll[0] < abs(dNqll[1]): #decreasing slope, force somehow???? second boolean isn't always right :(
        dNqll[0] = -dNqll[0]

    if Nqll0[-3] - Nqll0[-2] < 0 and Nqll0[-1] > Nqll0[-2]: #decreasing slope, final point is bigger than last
        dNqll[-1] = -dNqll[-1]
    elif Nqll0[-3] - Nqll0[-2] < 0 and Nqll0[-1] < abs(Nqll0[-2]): #increasing slope, another somehow force that isnt always right.....
        dNqll[-1] = -dNqll[-1]

    # Package for output
    derivs = list(dNqll)
    return derivs