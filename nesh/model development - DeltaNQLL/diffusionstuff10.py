# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 15:01:47 2015
@author: nesh, jonathan
"""
import numpy as np
import copy
from numba import njit, float64, int32, types
from scipy.integrate import solve_ivp

def getsigma_m(NQLL0,Nbar,Nstar,sigmaI,sigma0):
    twopi = 2*np.pi
    m = (NQLL0 - (Nbar - Nstar))/(2*Nstar)
    sigma_m = (sigmaI - m * sigma0)/(1+m*sigma0)
    return sigma_m

@njit
def getNQLL(Ntot,Nstar,Nbar):
    return Nbar - Nstar*np.sin(2*np.pi*Ntot)
    
@njit
def getDeltaNQLL(Ntot,Nstar,Nbar,NQLL):
    return NQLL-getNQLL(Ntot,Nstar,Nbar)

@njit
def f1d_sigma_m(y, t, params):
    Nbar, Nstar, sigmaI, sigma0, nu_kin_mlyperus, Doverdeltax2, nx = params
    NQLL0, Ntot0 = np.reshape(y,(2,nx))      # unpack current values of y
    
    # Deposition
    m = (NQLL0 - (Nbar - Nstar))/(2*Nstar)
    sigma_m = (sigmaI - m * sigma0)/(1+m*sigma0)
    return sigma_m

@njit
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
    
@njit
def f0d_solve_ivp(t, y, myparams):
    Nbar, Nstar, sigmaI, sigma0, nu_kin_mlyperus, tau_eq = myparams  # unpack parameters
    NQLL0 = y[0]
    Ntot0 = y[1]      # unpack current values of y

    # Ntot deposition
    twopi = 2*np.pi
    m = (NQLL0 - (Nbar - Nstar))/(2*Nstar)
    sigma_m = (sigmaI - m * sigma0)/(1+m*sigma0)
    depsurf = nu_kin_mlyperus * sigma_m
    dNtot_dt = depsurf
    
    # NQLL
    dNQLL_dt = dNtot_dt - getDeltaNQLL(Ntot0,Nstar,Nbar,NQLL0)/tau_eq
    
    # Packaging up for output
    derivs = [dNQLL_dt, dNtot_dt]
    return derivs

# @njit("f8[:](f8[:],f8)")
# def diffuse_1d(NQLL0,DoverdeltaX2):
#     l = len(NQLL0)
# #     l = 501
#     dy = np.zeros((l,))
#     for i in range(1,l):
#         dy[i] = DoverdeltaX2*(NQLL0[i+1]-2*NQLL0[i]+NQLL0[i-1])
#     dy[0] = DoverdeltaX2*(NQLL0[1]-2*NQLL0[0]+NQLL0[l-1]) 
#     dy[l-1] = DoverdeltaX2*(NQLL0[0]-2*NQLL0[l-1]+NQLL0[l-2])
#     return dy

@njit("f8[:](f8,f8[:],f8[:],f8[:])")
def f1d_solve_ivp(t, y, scalar_params, sigmaI):
    Nbar, Nstar, sigma0, nu_kin_mlyperus, DoverdeltaX2, tau_eq = scalar_params
    l = int(len(y)/2)
    NQLL0 = y[:l]
    Ntot0 = y[l:]
    
    # Ntot deposition
    twopi = 2*np.pi
    m = (NQLL0 - (Nbar - Nstar))/(2*Nstar)
    sigma_m = (sigmaI - m * sigma0)/(1+m*sigma0)
    depsurf = nu_kin_mlyperus * sigma_m
    dNtot_dt = depsurf

    # Ntot diffusion
    dy = np.empty(np.shape(NQLL0))
    for i in range(1,len(NQLL0)-1):
        dy[i] = DoverdeltaX2*(NQLL0[i-1]-2*NQLL0[i]+NQLL0[i+1])
    dy[0]  = DoverdeltaX2*(NQLL0[-1] -2*NQLL0[0] +NQLL0[1]) 
    dy[-1] = DoverdeltaX2*(NQLL0[-2] -2*NQLL0[-1]+NQLL0[0])

    # Combined
    dNtot_dt += dy

    # NQLL    
    dNQLL_dt = dNtot_dt - getDeltaNQLL(Ntot0,Nstar,Nbar,NQLL0)/tau_eq
    
    # Package for output
    derivs = np.empty(2*l)
    derivs[:l] = dNQLL_dt
    derivs[l:] = dNtot_dt

    return derivs


def run_f0d(NQLL_init_0D,Ntot_init_0D,times,params,odemethod):
    # Call the ODE solver
    ylast = np.array([NQLL_init_0D,Ntot_init_0D])
    ykeep_0D = [ylast]
    lastprogress = 0

    nt = len(times)
    for i in range(0,nt-1):

        # Specify the time interval of this step
        tinterval = [times[i],times[i+1]]
        
        # Integrate up to next time step
        sol = solve_ivp(f0d_solve_ivp, tinterval, ylast, dense_output=True, args=(params,),rtol=1e-12,method=odemethod)
        ylast = sol.y[:,-1]

        # Stuff into keeper arrays
        ykeep_0D.append(ylast)
        
        # Progress reporting
        progress = int(i/nt*100)
        if np.mod(progress,10) == 0:
            if progress > lastprogress:
                print(progress,'% done')
                lastprogress = progress

    print('100% done')
    ykeep_0D = np.array(ykeep_0D, np.float64)
    NQLLkeep_0D = ykeep_0D[:,0]
    Ntotkeep_0D = ykeep_0D[:,1]

    return Ntotkeep_0D, NQLLkeep_0D 


def run_f1d(NQLL_init_1D,Ntot_init_1D,times,scalar_params,sigmaI,odemethod):
    # Call the ODE solver
    nt = len(times)
    nx = len(NQLL_init_1D)
    ylast = np.array([NQLL_init_1D,Ntot_init_1D])
    ylast = np.reshape(ylast,2*nx)
    ykeep_1D = [ylast]
    lastprogress = 0


    for i in range(0,nt-1):

        # Specify the time interval of this step
        tinterval = [times[i],times[i+1]]
        
        # Integrate up to next time step
        sol = solve_ivp(f1d_solve_ivp, tinterval, ylast, args=(scalar_params,sigmaI),rtol=1e-12,method=odemethod)
        ylast = sol.y[:,-1]
        
        # Stuff into keeper arrays
        ykeep_1D.append(ylast)
        
        # Progress reporting
        progress = int(i/nt*100)
        if np.mod(progress,10) == 0:
            if progress > lastprogress:
                print(progress,'% done')
                lastprogress = progress

    print('100% done')
    ykeep_1D = np.array(ykeep_1D, np.float64)
    ykeep_1Darr = np.array(ykeep_1D, np.float64)
    ykeep_1Darr_reshaped = np.reshape(ykeep_1Darr,(nt,2,nx))
    Ntotkeep_1D = ykeep_1Darr_reshaped[:,1,:]
    NQLLkeep_1D = ykeep_1Darr_reshaped[:,0,:]
    return Ntotkeep_1D, NQLLkeep_1D


# def f0d_solve_ivp_1var(t, y, myparams):
#     Nbar, Nstar, sigmaI, sigma0, nu_kin_mlyperus = myparams  # unpack parameters
#     Ntot0 = y[0]      # unpack current values of y

#     # Deposition
#     NQLL0 = getNQLL(Ntot0,Nstar,Nbar)
#     m = (NQLL0 - (Nbar - Nstar))/(2*Nstar)
#     sigma_m = (sigmaI - m * sigma0)/(1+m*sigma0)
#     dNtot_dt = nu_kin_mlyperus * sigma_m
#     return dNtot_dt

# def f1d(y, t, params):
#     Nbar, Nstar, sigmaI, sigma0, nu_kin_mlyperus, Doverdeltax2, nx = params
#     NQLL0, Ntot0 = np.reshape(y,(2,nx))      # unpack current values of y
    
#     # Deposition
#     twopi = 2*np.pi
#     m = (NQLL0 - (Nbar - Nstar))/(2*Nstar)
#     sigma_m = (sigmaI - m * sigma0)/(1+m*sigma0)
#     depsurf = nu_kin_mlyperus * sigma_m
#     dNQLL_dt = -depsurf*Nstar*twopi/Nbar*np.cos(twopi*Ntot0)
#     dNtot_dt =  depsurf
    
#     # Diffusion
#     dy = np.zeros(np.shape(NQLL0))
#     for i in range(1,len(NQLL0)-1):
#         dy[i] = Doverdeltax2*(NQLL0[i-1]-2*NQLL0[i]+NQLL0[i+1])
#     dy[0]  = Doverdeltax2*(NQLL0[-1] -2*NQLL0[0] +NQLL0[1]) 
#     dy[-1] = Doverdeltax2*(NQLL0[-2] -2*NQLL0[-1]+NQLL0[0])
    
# #     # Seems to be an equivalent alternative: non-periodic boundary conditions: 
# #     dy[0]  = Doverdeltax2*(-NQLL0[0] +NQLL0[1]) 
# #     dy[-1] = Doverdeltax2*(NQLL0[-2] -NQLL0[-1])
     
#     dNtot_dt += dy
#     dNQLL_dt += dy

#     # Package for output
#     derivs = list([dNQLL_dt, dNtot_dt])
#     derivs = np.reshape(derivs,2*nx)
#     return derivs

# def f1d_1var(y, t, params):
#     Nbar, Nstar, sigmaI, sigma0, nu_kin_mlyperus, Doverdeltax2, nx = params
#     Ntot0 = y      # unpack current value of y
    
#     # Deposition
#     NQLL0 = getNQLL(Ntot0,Nstar,Nbar)
#     m = (NQLL0 - (Nbar - Nstar))/(2*Nstar)
#     sigma_m = (sigmaI - m * sigma0)/(1+m*sigma0)
#     dNtot_dt = nu_kin_mlyperus * sigma_m
    
#     # Diffusion
#     dy = np.zeros(np.shape(NQLL0))
#     for i in range(1,len(NQLL0)-1):
#         dy[i] = Doverdeltax2*(NQLL0[i-1]-2*NQLL0[i]+NQLL0[i+1])
#     dy[0]  = Doverdeltax2*(NQLL0[-1] -2*NQLL0[0] +NQLL0[1]) 
#     dy[-1] = Doverdeltax2*(NQLL0[-2] -2*NQLL0[-1]+NQLL0[0])
#     dNtot_dt += dy

#     # Package for output
#     derivs = list(dNtot_dt)
#     return derivs

# def f1d_solve_ivp_1var(t, y, params):
#     Nbar, Nstar, sigmaI, sigma0, nu_kin_mlyperus, Doverdeltax2, nx = params
#     Ntot0 = y     # unpack current values of y
    
#     # Deposition
#     NQLL0 = getNQLL(Ntot0,Nstar,Nbar)
#     m = (NQLL0 - (Nbar - Nstar))/(2*Nstar)
#     sigma_m = (sigmaI - m * sigma0)/(1+m*sigma0)
#     dNtot_dt = nu_kin_mlyperus * sigma_m
    
#     # Diffusion
#     dy = np.zeros(np.shape(NQLL0))
#     for i in range(1,len(NQLL0)-1):
#         dy[i] = Doverdeltax2*(NQLL0[i-1]-2*NQLL0[i]+NQLL0[i+1])
#     dy[0]  = Doverdeltax2*(NQLL0[-1] -2*NQLL0[0] +NQLL0[1]) 
#     dy[-1] = Doverdeltax2*(NQLL0[-2] -2*NQLL0[-1]+NQLL0[0])
#     dNtot_dt += dy

#     # Package for output
#     derivs = list(dNtot_dt)
#     return derivs

# def f0d(y, t, myparams):
#     Nbar, Nstar, sigmaI, sigma0, nu_kin_mlyperus, tau_eq = myparams  # unpack parameters
#     NQLL0 = y[0]
#     Ntot0 = y[1]      # unpack current values of y

#     # Deposition
#     twopi = 2*np.pi
#     m = (NQLL0 - (Nbar - Nstar))/(2*Nstar)
#     sigma_m = (sigmaI - m * sigma0)/(1+m*sigma0)
#     depsurf = nu_kin_mlyperus * sigma_m
#     dNtot_dt = depsurf
#     dNQLL_dt = dNtot_dt + getDeltaNQLL(Ntot0,Nstar,Nbar,NQLL0)/tau_eq
#     derivs = [dNQLL_dt, dNtot_dt]
#     return derivs

# def f0d_1var(y, t, params):
#     Nbar, Nstar, sigmaI, sigma0, nu_kin_mlyperus = params  # unpack parameters
#     Ntot0 = y      # unpack current value of y

#     # Deposition
#     NQLL0 = getNQLL(Ntot0,Nstar,Nbar)
#     m = (NQLL0 - (Nbar - Nstar))/(2*Nstar)
#     sigma_m = (sigmaI - m * sigma0)/(1+m*sigma0)
#     dNtot_dt = nu_kin_mlyperus * sigma_m
#     return dNtot_dt

