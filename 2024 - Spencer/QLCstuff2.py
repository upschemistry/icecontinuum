import numpy as np
from copy import copy as cp
import matplotlib.pylab as plt
from scipy.integrate import solve_ivp
from numba import njit, float64, int32, types
from matplotlib import rcParams
from time import time
from scipy.fft import fft, ifft, rfft, irfft, fftfreq
from scipy.interpolate import CubicSpline
from pint import UnitRegistry; AssignQuantity = UnitRegistry().Quantity

import sys
import f90nml

ticklabelsize = 15
linewidth = 1
fontsize = 15
titlefontsize = 8
color = 'k'
markersize = 10

@njit
def getNQLL(Ntot,Nstar,Nbar):
    return Nbar - Nstar*np.sin(2*np.pi*Ntot)

@njit
def getDeltaNQLL(Ntot,Nstar,Nbar,NQLL):
    return NQLL - (Nbar - Nstar*np.sin(2*np.pi*Ntot))

def f1d_solve_ivp_dimensionless(t, y, scalar_params, sigmaI, j2_list):
    Nbar, Nstar, sigma0, omega_kin, deltax = scalar_params
    l = int(len(y)/2)
    NQLL0 = y[:l]
    Ntot0 = y[l:]

    # Diffusion term based on FT
    Dcoefficient1 = 4 / (deltax * l * np.pi)**2  #print('Dcoefficient1', Dcoefficient1)
    bj_list = rfft(NQLL0)
    cj_list = bj_list*j2_list
    dy = -Dcoefficient1  * irfft(cj_list)

    # Ntot deposition
    m = (NQLL0 - (Nbar - Nstar))/(2*Nstar)
    sigma_m = (sigmaI - m * sigma0)
    dNtot_dt = omega_kin * sigma_m

    # Combined
    dNtot_dt += dy

    # NQLL    
    dNQLL_dt = dNtot_dt - getDeltaNQLL(Ntot0,Nstar,Nbar,NQLL0)
    
    # Package for output
    return np.concatenate((dNQLL_dt, dNtot_dt))

#Now takes just deltaX as input
def run_f1d_dimensionless(\
           NQLL_init_1D,Ntot_init_1D,times,\
           Nbar, Nstar, sigma0, nu_kin_mlyperus, deltaX, D, tau_eq, sigmaI,\
           AssignQuantity,\
           verbose=0, odemethod='LSODA'):
    
    # Prep for the integration
    nt = len(times)
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
        [Nbar, Nstar, sigma0, omega_kin, deltaX])
    
    # Loop over times
    for i in range(0,nt-1):
                
        # Specify the time interval of this step
        tinterval = [times[i].magnitude,times[i+1].magnitude]
        
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
    
    #scale back to dimensionalized space
    Ntotkeep_1D = Ntotkeep_1D * np.sqrt(D * tau_eq)
    NQLLkeep_1D = NQLLkeep_1D * np.sqrt(D * tau_eq)
    
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
"""