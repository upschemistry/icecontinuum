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
    Nbar, Nstar, sigmastepmax, sigma0, deprate = params  # unpack parameters
    
    Fliq0, Ntot0 = y      # unpack current values of y
    
    delta = (Fliq0 - (Nbar - Nstar))/(2*Nstar)
    sigD = (sigmastepmax - delta * sigma0)/(1+delta*sigma0)
    depsurf = deprate * sigD

    dFliq0_dt = -Nstar/Nbar*np.cos(2*np.pi*Ntot0)*2*np.pi*depsurf
    dNtot_dt = depsurf
    
    derivs = [dFliq0_dt, dNtot_dt]
    return derivs

def f1d(y, t, params):
    Nbar, Nstar, sigmastep, sigma0, deprate, DoverdeltaX2, nx = params
    Fliq0, Ntot0 = np.reshape(y,(2,nx))      # unpack current values of y
    
    # Deposition
    delta = (Fliq0 - (Nbar - Nstar))/(2*Nstar)
    sigD = (sigmastep - delta * sigma0)/(1+delta*sigma0)
    depsurf = deprate * sigD
    dFliq0_dt = -Nstar/Nbar*np.cos(2*np.pi*Ntot0)*2*np.pi*depsurf
    dNtot_dt = depsurf

    # Diffusion
    l = len(Fliq0)
    dy = np.zeros(np.shape(Fliq0))
    for i in range(1,l-1):
        dy[i] = DoverdeltaX2*(Fliq0[i-1]-2*Fliq0[i]+Fliq0[i+1])
    
    # Periodic boundary Conditions
    dy[0]  = DoverdeltaX2*(Fliq0[-1] -2*Fliq0[0] +Fliq0[1]) 
    dy[-1] = DoverdeltaX2*(Fliq0[-2] -2*Fliq0[-1]+Fliq0[0])
    
#     # Seems to be an equivalent alternative: non-periodic boundary conditions: 
#     dy[0]  = DoverdeltaX2*(-Fliq0[0] +Fliq0[1]) 
#     dy[-1] = DoverdeltaX2*(Fliq0[-2] -Fliq0[-1])
     
    # Updating the total (ice+liq) derivative
    dNtot_dt += dy

    # Updating the liquid derivative
    # Option 1: the original formlation
    dFliq0_dt += dy

#     # Option 2: Tayor approximation of the constraint getNliq(Ntot,Nstar,Nbar,niter)
#     twopi = 2*np.pi
#     dFliq0_dt = -dNtot_dt*Nstar*twopi*np.cos(twopi*Ntot0)

#     # Option 3 - Numerical evaluation of the constraint equation (much too slow as it stands)
#     twopi = 2*np.pi
#     dtprime = 1e-3
#     dFliq0_dt = (getNliq(Ntot0+dNtot_dt*dtprime,Nstar,Nbar,niter)-Fliq0)/dtprime
#     # print('from diffusionstuff: ', dFliq0_dt[0], dNtot_dt[0]*Nstar*twopi*np.cos(twopi*(Ntot0[0]-Nbar)))
    
    # Package for output
    derivs = list([dFliq0_dt, dNtot_dt])
    derivs = np.reshape(derivs,2*nx)
    return derivs

def f1d_sigD(y, t, params):
    Nbar, Nstar, niter, sigmastep, sigma0, deprate, DoverdeltaX2, nx = params
    Fliq0, Ntot0 = np.reshape(y,(2,nx))      # unpack current values of y
    
    # Deposition
    delta = (Fliq0 - (Nbar - Nstar))/(2*Nstar)
    sigD = (sigmastep - delta * sigma0)/(1+delta*sigma0)
    return sigD

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