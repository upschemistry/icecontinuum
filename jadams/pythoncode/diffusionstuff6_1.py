# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 15:01:47 2015
@author: nesh, jonathan
"""
import numpy as np
import copy
    
def getNliq(Ntot,Nstar,Nbar):
    fstar = Nstar/Nbar
    return Nbar - Nstar*np.sin(2*np.pi*Ntot)
    
def f0d(y, t, params):
    Nbar, Nstar, sigmaIcorner, sigma0, deprate = params  # unpack parameters
    NQLL0, Ntot0 = y      # unpack current values of y

    # Deposition
    twopi = 2*np.pi
    delta = (NQLL0 - (Nbar - Nstar))/(2*Nstar)
    sigD = (sigmaIcorner - delta * sigma0)/(1+delta*sigma0)
    depsurf = deprate * sigD
    dNQLL_dt = -depsurf*Nstar/Nbar*np.cos(twopi*Ntot0)*twopi
    dNtot_dt =  depsurf
    derivs = [dNQLL_dt, dNtot_dt]
    return derivs

def f1d(y, t, params):
    Nbar, Nstar, sigmaI, sigma0, deprate, Doverdeltax2, nx = params
    NQLL0, Ntot0 = np.reshape(y,(2,nx))      # unpack current values of y
    
    # Deposition
    twopi = 2*np.pi
    delta = (NQLL0 - (Nbar - Nstar))/(2*Nstar)
    sigD = (sigmaI - delta * sigma0)/(1+delta*sigma0)
    depsurf = deprate * sigD
    dNQLL_dt = -depsurf*Nstar*twopi/Nbar*np.cos(twopi*Ntot0)
    dNtot_dt =  depsurf
    
    # Diffusion
    l = len(NQLL0)
    dy = np.zeros(np.shape(NQLL0))
    for i in range(1,l-1):
        dy[i] = Doverdeltax2*(NQLL0[i-1]-2*NQLL0[i]+NQLL0[i+1])
    
    # Periodic boundary Conditions
    dy[0]  = Doverdeltax2*(NQLL0[-1] -2*NQLL0[0] +NQLL0[1]) 
    dy[-1] = Doverdeltax2*(NQLL0[-2] -2*NQLL0[-1]+NQLL0[0])
    
#     # Seems to be an equivalent alternative: non-periodic boundary conditions: 
#     dy[0]  = Doverdeltax2*(-NQLL0[0] +NQLL0[1]) 
#     dy[-1] = Doverdeltax2*(NQLL0[-2] -NQLL0[-1])
     
    dNtot_dt += dy

    # Updating the liquid derivative
    dNQLL_dt += dy

    # Package for output
    derivs = list([dNQLL_dt, dNtot_dt])
    derivs = np.reshape(derivs,2*nx)
    return derivs

def f1d_sigma_m(y, t, params):
    Nbar, Nstar, sigmaI, sigma0, deprate, Doverdeltax2, nx = params
    NQLL0, Ntot0 = np.reshape(y,(2,nx))      # unpack current values of y
    
    # Deposition
    m = (NQLL0 - (Nbar - Nstar))/(2*Nstar)
    sigma_m = (sigmaI - m * sigma0)/(1+m*sigma0)
    return sigma_m

def getsigmaI(x,xmax,center_reduction,sigmaIcorner,method='sinusoid'):
    sigmapfac = 1-center_reduction/100
    xmid = max(x)/2
    if method == 'sinusoid':
        fsig = (np.cos(x/xmax*np.pi*2)+1)/2*(1-sigmapfac)+sigmapfac
    elif method == 'parabolic':
        fsig = (x-xmid)**2/xmid**2*(1-sigmapfac)+sigmapfac
    else:
        print('bad method')
    return fsig*sigmaIcorner