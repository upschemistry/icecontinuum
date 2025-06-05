import numpy as np
from copy import copy as cp
import matplotlib.pylab as plt
from numba import njit, float64, int32, types
from matplotlib import rcParams
from time import time
from scipy.fft import fft, ifft, rfft, irfft, fftfreq
from scipy.interpolate import CubicSpline
from scipy.integrate import solve_ivp
from pint import UnitRegistry; AssignQuantity = UnitRegistry().Quantity

# import copy
from numba import njit, float64, int32, types


import sys
import f90nml

ticklabelsize = 15
linewidth = 1
fontsize = 15
titlefontsize = 8
color = 'k'
markersize = 10

def get_nu_kin(T,AssignQuantity):
    P3 = AssignQuantity(611,'Pa')
    T3 = AssignQuantity(273,'K')
    R = AssignQuantity(8.314,'J/mol/K')
    M = AssignQuantity(18,'g/mol')
    NA = AssignQuantity(6.02e23,'1/mol')
    rho = AssignQuantity(0.9,'g/cm^3')
    
    # Clausius-Clapeyron
    Delta_H_sub = AssignQuantity(50,'kJ/mol')
    P_vapor_eq = P3*np.exp(-Delta_H_sub/R*(1/T-1/T3))
    
    # Hertz-Knudsen
    nu_kin = P_vapor_eq*M**.5/(2*np.pi*R*T)**.5
    nu_kin.ito('gram / micrometer ** 2 / second')
    nu_kin /= rho
    nu_kin.ito('micrometer/second')
    return(nu_kin)

def get_D_of_T(T,AssignQuantity):
    """ Based on a log/inverse T fit to Price's data for supercooled liquid water """
    T_inverse_Temperature = 1e3/T; #print(T_inverse_Temperature)
    p = [-2.74653072, 9.97737468]
    logD = np.polyval(p,T_inverse_Temperature.magnitude)
    D = AssignQuantity(np.exp(logD)*1e-5*100,'micrometers^2/microsecond')
    return D

def get_L_cr_of_T(Temperature,AssignQuantity):
    aT = 9.341579E-09
    bT = 9.857504E-02
    L_cr_of_T = aT * np.exp(bT*Temperature.magnitude)
    return L_cr_of_T

def get_L_cr_of_P(Pressure,AssignQuantity): 
    aP = 18041.86122836
    bP = -1.09599342
    L_cr_of_P = aP * Pressure.magnitude**(bP)
    return L_cr_of_P

def get_L_cr_of_TP(Temperature,Pressure,AssignQuantity): 
    L_cr_of_T = get_L_cr_of_T(Temperature,AssignQuantity)
    Temperature_ref = AssignQuantity(240,'K')
    L_cr_of_T_ref = get_L_cr_of_T(Temperature_ref,AssignQuantity)
    L_cr_of_P = get_L_cr_of_P(Pressure,AssignQuantity)
    L_cr_of_TP = L_cr_of_T/L_cr_of_T_ref*L_cr_of_P
    return L_cr_of_TP

def get_cr_of_T(L,Temperature,AssignQuantity): 
    cr = L/get_L_cr_of_T(Temperature,AssignQuantity)
    return cr

def get_cr_of_P(L,Pressure,AssignQuantity): 
    cr = L/get_L_cr_of_P(Pressure,AssignQuantity)
    return cr

def get_cr_of_TP(L,Temperature,Pressure,AssignQuantity):     
#     Temperature_ref = AssignQuantity(240,'K')
#     L_cr_T = get_L_cr_of_T(Temperature,AssignQuantity)
#     L_cr_Tref = get_L_cr_of_T(Temperature_ref,AssignQuantity)
#     L_cr_P = get_L_cr_of_P(Pressure,AssignQuantity)
#     L_cr = L_cr_T/L_cr_Tref*L_cr_P
    L_cr_of_TP = get_L_cr_of_TP(Temperature,Pressure,AssignQuantity)
    cr_of_TP = L/L_cr_of_TP
    return cr_of_TP

@njit
def getNQLL(Ntot,Nstar,Nbar):
    return Nbar - Nstar*np.sin(2*np.pi*Ntot)

# @njit
# def getDeltaNQLL(Ntot,Nstar,Nbar,NQLL):
#     return NQLL - (Nbar - Nstar*np.sin(2*np.pi*Ntot))

@njit
def getDeltaNQLL(Ntot,Nstar,Nbar,NQLL):
    return NQLL-getNQLL(Ntot,Nstar,Nbar)


def f1d_solve_ivp_dimensionless(t, y, scalar_params, sigmaI, j2_list):
    Nbar, Nstar, sigma0, omega_kin, deltax, D, t_0, with_diffusion = scalar_params
    l = int(len(y)/2)
    NQLL0 = y[:l]
    Ntot0 = y[l:]

    # Ntot deposition
    m = (NQLL0 - (Nbar - Nstar))/(2*Nstar)
    sigma_m = (sigmaI - m * sigma0)
    dNtot_dt = omega_kin * sigma_m

    if with_diffusion:
        # Diffusion term based on FT
        Dcoefficient1 =   4 * np.pi**2 / (deltax * l)**2  #print('Dcoefficient1', Dcoefficient1)
        bj_list = rfft(NQLL0)
        cj_list = bj_list*j2_list
        dy = -Dcoefficient1  * irfft(cj_list)
        dNtot_dt += dy

    # NQLL    
    dNQLL_dt = dNtot_dt - (NQLL0 - (Nbar - Nstar*np.sin(2*np.pi*Ntot0)))
    
    # Package for output
    return np.concatenate((dNQLL_dt, dNtot_dt))

@njit("f8[:](f8,f8[:],f8[:],f8[:])")
def f1d_solve_ivp(t, y, scalar_params, sigmaI):
    Nbar, Nstar, sigma0, nu_kin_mlyperus, DoverdeltaX2, tau_eq = scalar_params
    l = int(len(y)/2)
    NQLL0 = y[:l]
    Ntot0 = y[l:]
    
    # Ntot deposition
    twopi = 2*np.pi
    m = (NQLL0 - (Nbar - Nstar))/(2*Nstar)
    sigma_m = (sigmaI - m * sigma0)
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

    
#Now takes deltax and D as input (not combined DoverdeltaX2)
def run_f1d_dimensionless(\
           NQLL_init_1D,Ntot_init_1D,times,\
           Nbar, Nstar, sigma0, nu_kin_mlyperus, deltaX, D, tau_eq, sigmaI,\
           AssignQuantity,\
           verbose=0, odemethod='LSODA', with_diffusion=True):
    """
    Takes dimensional arrays for NQLL, Ntot, and time steps but not dimensional quantities for scalars
    """
    
    #convert to nondimensional?
    times_nondim = times / tau_eq
    deltaX_nondim = deltaX / (np.sqrt(D * tau_eq))

    # Prep for the integration
    nt = len(times_nondim)
    nx = len(NQLL_init_1D)
    ylast = np.array([NQLL_init_1D,Ntot_init_1D])
    ylast = np.reshape(ylast,2*nx)
    ykeep_1D = [ylast]
    lastprogress = 0
    sigmaI_mag = sigmaI.magnitude
    t1 = time()
    bj_list = rfft(NQLL_init_1D)
    j_list = np.array([j for j in range(len(bj_list))])
    j2_list = np.array(j_list)**2

    #new nondimensionalized params
    omega_kin = nu_kin_mlyperus * tau_eq

    # Bundle params for ODE solver
    scalar_params = np.array(\
        [Nbar, Nstar, sigma0, omega_kin, deltaX_nondim, D, tau_eq, with_diffusion])
    
    # Loop over times
    for i in range(0,nt-1):
                
        # Specify the time interval of this step
        tinterval = [times_nondim[i].magnitude,times_nondim[i+1].magnitude]
        
        if verbose > 0:
            print(tinterval)
            print(ylast)
            print(scalar_params)
            print(sigmaI_mag)
            print(odemethod)
        
        # Integrate up to next time step
        sol = solve_ivp(\
            f1d_solve_ivp_dimensionless, tinterval, ylast, args=(scalar_params, sigmaI_mag, j2_list), \
            rtol=1e-12, method=odemethod) 
        ylast = sol.y[:,-1]
        
        # Symmetrizing
        ylast = np.array(ylast, dtype=np.complex_)
        ylast_reshaped = np.reshape(ylast,(2,nx))
        NQLL_last = ylast_reshaped[0,:]
        Ntot_last = ylast_reshaped[1,:]
        nx_mid = int(nx/2)
        for j in range(0,nx_mid):
            jp = nx -j -1
            Ntot_last[j] = Ntot_last[jp]
            NQLL_last[j] = NQLL_last[jp]
        ylast = np.array([NQLL_last,Ntot_last])
        ylast = np.reshape(ylast,2*nx)
        ylast = np.real(ylast)

        # Stuff into keeper arrays
        ykeep_1D.append(ylast)
        
        # Progress reporting
        progress = int(i/nt*100)
        if np.mod(progress,10) == 0:
            if progress > lastprogress:
                t2 = time()
                elapsed = (t2 - t1)/60
                print(progress,'%'+' elapsed time is %.3f minutes' %elapsed)
                lastprogress = progress

                
    print('100% done')
    print('status = ', sol.status)
    print('message = ', sol.message)
    print(dir(sol))
    
    ykeep_1D = np.array(ykeep_1D, np.float64)
    ykeep_1Darr = np.array(ykeep_1D, np.float64)
    ykeep_1Darr_reshaped = np.reshape(ykeep_1Darr,(nt,2,nx))
    Ntotkeep_1D = ykeep_1Darr_reshaped[:,1,:]
    NQLLkeep_1D = ykeep_1Darr_reshaped[:,0,:]
    
    # This would be how we would scale back to dimensionalized space
    Ntotkeep_1D = Ntotkeep_1D #* np.sqrt(D * tau_eq)
    NQLLkeep_1D = NQLLkeep_1D #* np.sqrt(D * tau_eq)
    
    return Ntotkeep_1D, NQLLkeep_1D

#I copied this and made no changes
def report_1d_growth_results_dimensionless(\
         x_QLC,tkeep_1Darr,NQLLkeep_1D,Ntotkeep_1D,Nicekeep_1D,nmpermonolayer,lastfraction=0, title_params='', \
         graphics=True,itime=-1,Liquid=True,IceAndLiquid=True,tgraphics=True,xlim=[],vlayers=0):
    
    # Parameters of the data
    ntimes = len(NQLLkeep_1D)

    if graphics:
        
        # Titles on graphs
        title_entire = title_params

        # Plot ice and total profile
        if IceAndLiquid:
            plt.figure()
            plt.plot(x_QLC.magnitude, Nicekeep_1D[itime,:], 'k', label='ice', lw=linewidth)
            plt.plot(x_QLC.magnitude, Ntotkeep_1D[itime,:], 'b', label='total', lw=linewidth)
            plt.xlabel('$x \ (\mu m$)',fontsize=fontsize)
            plt.ylabel('$ice \ & \ liquid \ layers$',fontsize=fontsize)
            rcParams['xtick.labelsize'] = ticklabelsize 
            rcParams['ytick.labelsize'] = ticklabelsize
            plt.legend()
            plt.title(title_entire,fontsize=titlefontsize)
            plt.grid('on')
            if len(xlim) > 0:
                plt.xlim(xlim)
                i = np.where( (x_QLC.magnitude > xlim[0]) &  (x_QLC.magnitude < xlim[1]) )[0]
                ymin = Nicekeep_1D[itime,:][i].min()
                ymax = Ntotkeep_1D[itime,:][i].max()
                plt.ylim( ymin, ymax ) 
            if vlayers != 0:
                ymin = Nicekeep_1D[itime,:][i].min()
                ymax = ymin + vlayers
                plt.ylim( ymin, ymax ) 

        # Plot liquid
        if Liquid:
            plt.figure()
            plt.plot(x_QLC.magnitude, NQLLkeep_1D[itime,:], 'b', label='liquid', lw=linewidth)
            plt.xlabel('$x \ (\mu m$)',fontsize=fontsize)
            plt.ylabel('$liquid \ layers$',fontsize=fontsize)
            rcParams['xtick.labelsize'] = ticklabelsize 
            rcParams['ytick.labelsize'] = ticklabelsize
            plt.title(title_entire,fontsize=titlefontsize)
            plt.grid('on')
            if len(xlim) > 0:
                plt.xlim(xlim)
                i = np.where( (x_QLC.magnitude > xlim[0]) &  (x_QLC.magnitude < xlim[1]) )[0]
                plt.ylim( NQLLkeep_1D[itime,:][i].min(), NQLLkeep_1D[itime,:][i].max() ) 

    if tgraphics:
        # Plot number of steps over time
        plt.figure()
        rcParams['xtick.labelsize'] = ticklabelsize 
        rcParams['ytick.labelsize'] = ticklabelsize
        f = np.max(Ntotkeep_1D,axis=1) - np.min(Ntotkeep_1D,axis=1)
        plt.plot(tkeep_1Darr.magnitude/1e3,f,lw=linewidth)
        plt.xlabel('t ($m s$)',fontsize=fontsize)
        plt.ylabel('Number of steps',fontsize=fontsize)
        plt.title(title_entire,fontsize=titlefontsize)
        plt.grid('on')

    # Some analysis
    if lastfraction == 0:
        lastfraction = 0.3
    itimes_almost_end = int(ntimes*(1-lastfraction))
    icorner = 0
    delta_N = Ntotkeep_1D[itime,icorner]-Ntotkeep_1D[itimes_almost_end,icorner]
    delta_t = tkeep_1Darr[itime]-tkeep_1Darr[itimes_almost_end]
    g_ice_QLC = delta_N/delta_t*nmpermonolayer
    
    return g_ice_QLC


####### From diffusionstuff11.py

def getsigma_m(NQLL0,Nbar,Nstar,sigmaI,sigma0):
    twopi = 2*np.pi
    m = (NQLL0 - (Nbar - Nstar))/(2*Nstar)
    sigma_m = (sigmaI - m * sigma0)
    return sigma_m

@njit
def f1d_sigma_m(y, t, params):
    Nbar, Nstar, sigmaI, sigma0, nu_kin_mlyperus, Doverdeltax2, nx = params
    NQLL0, Ntot0 = np.reshape(y,(2,nx))      # unpack current values of y
    
    # Deposition
    m = (NQLL0 - (Nbar - Nstar))/(2*Nstar)
    sigma_m = (sigmaI - m * sigma0)
    return sigma_m

@njit
def getsigmaI(x,xmax,center_reduction,sigmaIcorner,method='sinusoid',nsinusoid=1):
    sigmapfac = 1-center_reduction/100
    xmid = max(x)/2
    if method == 'sinusoid':
        fsig = (np.cos(x/xmax*np.pi*2*nsinusoid)+1)/2*(1-sigmapfac)+sigmapfac
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
    sigma_m = (sigmaI - m * sigma0)
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





######################### main for debugging ###########

""" 
inputfile = "./2024 - Spencer/GI parameters - Reference limit cycle (for testing).nml"

# For readability ...
print('Using parameter file '+inputfile+' ...')

print('\nFrom the GrowthInstability (GI) namelist:')
GI=f90nml.read(inputfile)['GI'] # Read the main parameter namelist

# Supersaturation at the corner of a facet
sigmaI_corner = GI['sigmaI_corner']
sigmaI_corner_units = GI['sigmaI_corner_units']
sigmaI_corner = AssignQuantity(sigmaI_corner,sigmaI_corner_units)
print('sigmaI_corner =', sigmaI_corner)

# Difference in equilibrium supersaturation between microsurfaces I and II
sigma0 = GI['sigma0']
sigma0_units = GI['sigma0_units']
sigma0 = AssignQuantity(sigma0,sigma0_units)
print('sigma0 =',sigma0)

# Reduction of supersaturation at the facet cental
c_r = GI['c_r']
c_r_units = GI['c_r_units']
c_r = AssignQuantity(c_r,c_r_units)
print('c_r =',c_r)

# Properties of the QLL
Nbar = GI['Nbar']; print('Nbar', Nbar)
Nstar = GI['Nstar']; print('Nstar', Nstar)

# Thickness of monolayers
h_pr = GI['h_pr']
h_pr_units = GI['h_pr_units']
h_pr = AssignQuantity(h_pr,h_pr_units) 
print('h_pr =', h_pr)

# Diffusion coeficient
D = GI['D']
D_units = GI['D_units']
D = AssignQuantity(D,D_units)
print('D =', D)

# Deposition velocity
nu_kin = GI['nu_kin']
nu_kin_units = GI['nu_kin_units']
nu_kin = AssignQuantity(nu_kin,nu_kin_units)
print('nu_kin =', nu_kin)

# Size of the facet
L = GI['L']
L_units = GI['L_units']
L = AssignQuantity(L,L_units)
print('L =', L)

# Crystal size -- needs to be an even number
nx_crystal = GI['nx_crystal']
print('nx (crystal) =', nx_crystal)

# Time constant for freezing/thawing
tau_eq = GI['tau_eq']
tau_eq_units = GI['tau_eq_units']
tau_eq = AssignQuantity(tau_eq,tau_eq_units)
print('tau_eq =',tau_eq)

# Integration algorithm (possibilities: RK45, BDF, RK23, DOP853, LSODA, and Radau)
odemethod = GI['odemethod']
print('odemethod =',odemethod)

# Conversions (in case inputs are in other units)
sigma0.ito('dimensionless')
h_pr.ito('micrometer')
D.ito('micrometer^2/microsecond')
nu_kin.ito('micrometer/second')
L.ito('micrometer')
sigmaI_corner.ito('dimensionless')
c_r.ito('dimensionless')
tau_eq.ito('microsecond')

x_QLC = np.linspace(-L,L,nx_crystal)
deltax = x_QLC[1]-x_QLC[0]
print('Spacing of points on the ice surface =', deltax)
sigmaI_QLC = sigmaI_corner*(c_r*(x_QLC/L)**2+1-c_r)
nu_kin_mlyperus = nu_kin/h_pr
nu_kin_mlyperus.ito('1/microsecond')
Doverdeltax2 = D/deltax**2

RT=f90nml.read(inputfile)['RT'] # Read the main parameter namelist

# How long
runtime = RT['runtime']
runtime = 10
runtime_units = RT['runtime_units']
runtime = AssignQuantity(runtime,runtime_units)
print('runtime =', runtime)
runtime.ito('microsecond')

# Number of time steps to keep for reporting later
ntimes = RT['ntimes']

# Flag if we want more output
verbose = RT['verbose']

# Specify the time interval and initial conditions
tkeep_1Darr = np.linspace(0,runtime,ntimes)
Ntot_init_1D = np.ones(nx_crystal)
NQLL_init_1D = getNQLL(Ntot_init_1D,Nstar,Nbar)

print('This is a run from time', tkeep_1Darr[0].to('msec'),'to', tkeep_1Darr[-1].to('msec'))
print('dt =', tkeep_1Darr[1]-tkeep_1Darr[0])

Ntotkeep_1D, NQLLkeep_1D = run_f1d_dimensionless(\
    NQLL_init_1D,Ntot_init_1D,tkeep_1Darr,\
    Nbar, 
    Nstar, 
    sigma0.magnitude, 
    nu_kin_mlyperus.magnitude,
    deltax.magnitude,
    D.magnitude,
    tau_eq.magnitude, 
    sigmaI_QLC,\
    AssignQuantity,\
    verbose=0, odemethod='RK45')
Nicekeep_1D = Ntotkeep_1D-NQLLkeep_1D

# Reporting and graphing
# Label for graphs
title_params = \
        "{:.0f}".format(L.magnitude)+' '+str(L.units)+\
        ", "+np.format_float_scientific(D.magnitude,precision=2)+" "+str(D.units)+\
        "\n"+\
        "{:.0f}".format(nu_kin.magnitude)+' '+str(nu_kin.units)+\
        "\n"+\
        "{:.3f}".format(sigmaI_corner.magnitude)+' '+str(sigmaI_corner.units)+\
        ", "+"{:.1f}".format(tau_eq.magnitude)+' '+str(tau_eq.units)+\
        ", "+"{:.3f}".format(c_r * 100)+'%'+\
        ", "+odemethod+\
        "\n"
    
print(":)")

g_ice_QLC = report_1d_growth_results_dimensionless(\
        x_QLC,tkeep_1Darr,NQLLkeep_1D,Ntotkeep_1D,Nicekeep_1D,h_pr, \
        graphics=True,title_params=title_params) """