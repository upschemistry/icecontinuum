# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 15:01:47 2015
@author: nesh, jonathan
"""
import numpy as np
import copy

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
    ans = 1 + fstar*np.sin(2*np.pi*(Ntot-Nbar*fqll_last))
    return ans

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
    ans =fqll_last*Nbar
    return ans

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

    #dFliq0_dt = getNliqprime(Ntot0,Nstar,Nbar,niter)*depsurf # This is a numerical derivative (and inaccurate)
    #dFliq0_dt = getdNliq_dNtot(Ntot0,Nstar,Nbar,niter)*depsurf # This is unnecessary as long as niter=1
    dFliq0_dt = Nstar/Nbar*np.cos(2*np.pi*Ntot0)*2*np.pi*depsurf
    dNtot_dt = depsurf
    
    derivs = [dFliq0_dt, dNtot_dt]
    return derivs

def f0d_trial1(y, t, params):
    Nbar, Nstar, niter, sigmastepmax, sigma0, deprate = params  # unpack parameters
    
    Fliq0, Ntot0 = y      # unpack current values of y
    
    delta = (Fliq0 - (Nbar - Nstar))/(2*Nstar)
    sigD = (sigmastepmax - delta * sigma0)/(1+delta*sigma0)
    depsurf = deprate * sigD

    dFliq0_dt = Nstar/Nbar*np.cos(2*np.pi*Ntot0)*2*np.pi*depsurf
    dNtot_dt = depsurf
    
    derivs = [dFliq0_dt, dNtot_dt]
    return derivs

def f1d(y, t, params):
    Nbar, Nstar, niter, sigmastep, sigma0, deprate, DoverdeltaX2, nx = params
    Fliq0, Ntot0 = np.reshape(y,(2,nx))      # unpack current values of y
    
    # Deposition
    delta = (Fliq0 - (Nbar - Nstar))/(2*Nstar)
    sigD = (sigmastep - delta * sigma0)/(1+delta*sigma0)
    depsurf = deprate * sigD
    dFliq0_dt = getdNliq_dNtot(Ntot0,Nstar,Nbar,niter)*depsurf
    dNtot_dt = depsurf
    print('from f1d: ', dFliq0_dt[0],dNtot_dt[0])

    # Diffusion
    l = len(Fliq0)
    dy = np.zeros(np.shape(Fliq0))
    for i in range(1,l-1):
        dy[i] = DoverdeltaX2*(Fliq0[i+1]-2*Fliq0[i]+Fliq0[i-1])
    
        # Boundary Conditions (periodic at ends)
        dy[0] = DoverdeltaX2*(Fliq0[1]-2*Fliq0[0]+Fliq0[l-1]) 
        dy[-1] = DoverdeltaX2*(Fliq0[0]-2*Fliq0[-1]+Fliq0[-2])
     
    # Combined
    dNtot_dt += dy
    dFliq0_dt += dy

    # Package for output
    derivs = list([dFliq0_dt, dNtot_dt])
    derivs = np.reshape(derivs,2*nx)
    return derivs

def f1d_trial1(y, t, params):
    Nbar, Nstar, niter, sigmastep, sigma0, deprate, DoverdeltaX2, nx = params
    Fliq0, Ntot0 = np.reshape(y,(2,nx))      # unpack current values of y
    
    # Deposition
    delta = (Fliq0 - (Nbar - Nstar))/(2*Nstar)
    sigD = (sigmastep - delta * sigma0)/(1+delta*sigma0)
    depsurf = deprate * sigD
    dFliq0_dt = Nstar/Nbar*np.cos(2*np.pi*Ntot0)*2*np.pi*depsurf
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
#     # Option 1: the original formlation
#     dFliq0_dt += dy

    # Option 2: Tayor approximation of the constraint getNliq(Ntot,Nstar,Nbar,niter)
    twopi = 2*np.pi
    dFliq0_dt = dNtot_dt*Nstar*twopi*np.cos(twopi*(Ntot0-Nbar))

#     # Option 3 - Numerical evaluation of the constraint equation (much too slow as it stands)
#     twopi = 2*np.pi
#     dtprime = 1e-3
#     dFliq0_dt = (getNliq(Ntot0+dNtot_dt*dtprime,Nstar,Nbar,niter)-Fliq0)/dtprime
#     # print('from diffusionstuff: ', dFliq0_dt[0], dNtot_dt[0]*Nstar*twopi*np.cos(twopi*(Ntot0[0]-Nbar)))
    
    # Package for output
    derivs = list([dFliq0_dt, dNtot_dt])
    derivs = np.reshape(derivs,2*nx)
    return derivs

def f1d_trial1_sigD(y, t, params):
    Nbar, Nstar, niter, sigmastep, sigma0, deprate, DoverdeltaX2, nx = params
    Fliq0, Ntot0 = np.reshape(y,(2,nx))      # unpack current values of y
    
    # Deposition
    delta = (Fliq0 - (Nbar - Nstar))/(2*Nstar)
    sigD = (sigmastep - delta * sigma0)/(1+delta*sigma0)
    return sigD

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