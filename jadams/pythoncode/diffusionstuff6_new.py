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
    NQLL0, Ntot0 = y      # unpack current values of y

    # Deposition
    twopi = 2*np.pi
    delta = (NQLL0 - (Nbar - Nstar))/(2*Nstar)
    sigD = (sigmastepmax - delta * sigma0)/(1+delta*sigma0)
    depsurf = deprate * sigD
    dNQLL_dt = -depsurf*Nstar/Nbar*np.cos(twopi*Ntot0)*twopi
    dNtot_dt =  depsurf
    derivs = [dNQLL_dt, dNtot_dt]
    return derivs

def f1d(y, t, params):
    Nbar, Nstar, sigmastep, sigma0, deprate, DoverdeltaX2, nx, gamma = params
    NQLL0, Ntot0 = np.reshape(y,(2,nx))      # unpack current values of y
    
    # Deposition
    twopi = 2*np.pi
    delta = (NQLL0 - (Nbar - Nstar))/(2*Nstar)
    sigD = (sigmastep - delta * sigma0)/(1+delta*sigma0)
    depsurf = deprate * sigD
    dNQLL_dt = -depsurf*Nstar*twopi/Nbar*np.cos(twopi*Ntot0)
    dNtot_dt =  depsurf
    
#     # A kluge
#     DoverdeltaX2 *= gamma
    
    # Diffusion
    l = len(NQLL0)
    dy = np.zeros(np.shape(NQLL0))
    for i in range(1,l-1):
        dy[i] = DoverdeltaX2*(NQLL0[i-1]-2*NQLL0[i]+NQLL0[i+1])
    
    # Periodic boundary Conditions
    dy[0]  = DoverdeltaX2*(NQLL0[-1] -2*NQLL0[0] +NQLL0[1]) 
    dy[-1] = DoverdeltaX2*(NQLL0[-2] -2*NQLL0[-1]+NQLL0[0])
    
#     # Seems to be an equivalent alternative: non-periodic boundary conditions: 
#     dy[0]  = DoverdeltaX2*(-NQLL0[0] +NQLL0[1]) 
#     dy[-1] = DoverdeltaX2*(NQLL0[-2] -NQLL0[-1])
     
    # Updating the total (ice+liq) derivative
#     ix = 5
#     print('From inside f1d, diff/react = ',  Ntot0[ix], dy[ix]/dNQLL_dt[ix])
    dNtot_dt += dy

    # Updating the liquid derivative
    
    # Option 1: the original formlation
    dNQLL_dt += dy
    
    # Option 2: Tayor approximation of the constraint getNliq(Ntot,Nstar,Nbar)
#     dNQLL_dt = -dNtot_dt*Nstar*twopi*np.cos(twopi*Ntot0)

#     # Option 3 - Exact (but this doesn't work) 
#     twopi = 2*np.pi
#     Ntot0mod = np.mod(Ntot0,1)
#     smallnumber = 1e-4
#     dtprime = np.max([smallnumber*Ntot0[0]/dNtot_dt[0],smallnumber])
# #     dNQLL_dt = ( Nbar-NQLL0-Nstar*np.sin(twopi*(Ntot0+dNtot_dt*dtprime)) )/dtprime
#     dNQLL_dt = ( Nbar-NQLL0-Nstar*np.sin(twopi*(Ntot0mod+dNtot_dt*dtprime)) )/dtprime
# #     dNQLL0_dt_test = -dNtot_dt*Nstar*twopi*np.cos(twopi*Ntot0)
# #     mytest = dNQLL_dt-dNQLL0_dt_test
# #     print('From inside f1d: ', dtprime, Ntot0[200], mytest[200])
    
#     # Option 4 - a hybrid of options 1 and 2
#     dNQLL_dt_orig = dNQLL_dt + dy
#     dNQLL_dt_fastequil = -dNtot_dt*Nstar*twopi*np.cos(twopi*Ntot0)
#     dNQLL_dt = (1-gamma)*dNQLL_dt_orig + gamma*dNQLL_dt_fastequil

    # Package for output
    derivs = list([dNQLL_dt, dNtot_dt])
    derivs = np.reshape(derivs,2*nx)
    return derivs

def f1d_sigD(y, t, params):
    Nbar, Nstar, sigmastep, sigma0, deprate, DoverdeltaX2, nx, gamma = params
    NQLL0, Ntot0 = np.reshape(y,(2,nx))      # unpack current values of y
    
    # Deposition
    delta = (NQLL0 - (Nbar - Nstar))/(2*Nstar)
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