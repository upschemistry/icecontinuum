import numpy as np
from scipy.integrate import solve_ivp
from time import time
from scipy.fft import rfft, irfft
from pint import UnitRegistry; AssignQuantity = UnitRegistry().Quantity

import f90nml


def generate_reference_solution(inputfile = "GI parameters.nml", runtime = 50, num_steps = 50, verbose = False):
    """ 
    Generates a reference solution for the QLC problem using a non-dimensional model.

    Args:
        runtime: int - number of ms to run model for
        num_steps: int - number of time steps to generate over 'runtime'
        verbose: Boolean - True to print parameter information, False otherwise

    Returns:
        A Numpy ndarray of shape (3, num_steps, nx_crystal).
        Corresponds to: (N-Tot, N-QLL, N-Ice)
    
    """
    

    # For readability ...
    print('Using parameter file '+inputfile+' ...\n')
    GI=f90nml.read(inputfile)['GI'] # Read the main parameter namelist

    ### Load model parameters from inputfile ###

    # Supersaturation at the corner of a facet
    sigmaI_corner = GI['sigmaI_corner']
    sigmaI_corner_units = GI['sigmaI_corner_units']
    sigmaI_corner = AssignQuantity(sigmaI_corner,sigmaI_corner_units)
    
    # Difference in equilibrium supersaturation between microsurfaces I and II
    sigma0 = GI['sigma0']
    sigma0_units = GI['sigma0_units']
    sigma0 = AssignQuantity(sigma0,sigma0_units)

    # Reduction of supersaturation at the facet center
    c_r = GI['c_r']
    c_r_units = GI['c_r_units']
    c_r = AssignQuantity(c_r,c_r_units)

    # Properties of the QLL
    Nbar = GI['Nbar']
    Nstar = GI['Nstar']

    # Thickness of monolayers
    h_pr = GI['h_pr']
    h_pr_units = GI['h_pr_units']
    h_pr = AssignQuantity(h_pr,h_pr_units) 

    # Diffusion coefficient
    D = GI['D']
    D_units = GI['D_units']
    D = AssignQuantity(D,D_units)

    # Deposition velocity
    nu_kin = GI['nu_kin']
    nu_kin_units = GI['nu_kin_units']
    nu_kin = AssignQuantity(nu_kin,nu_kin_units)

    # Size of the facet
    L = GI['L']
    L_units = GI['L_units']
    L = AssignQuantity(L,L_units)

    # Crystal size -- needs to be an even number
    nx_crystal = GI['nx_crystal']

    # Time constant for freezing/thawing
    t_eq = GI['t_eq']
    t_eq_units = GI['t_eq_units']
    t_eq = AssignQuantity(t_eq,t_eq_units)

    # Integration algorithm (possibilities: RK45, BDF, RK23, DOP853, LSODA, and Radau)
    odemethod = GI['odemethod']

    # Conversions (in case inputs are in other units)
    sigma0.ito('dimensionless')
    h_pr.ito('micrometer')
    D.ito('micrometer^2/microsecond')
    nu_kin.ito('micrometer/second')
    L.ito('micrometer')
    sigmaI_corner.ito('dimensionless')
    c_r.ito('dimensionless')
    t_eq.ito('microsecond')
    
    # Create evenly-spaced x values
    x_QLC = np.linspace(-L,L,nx_crystal)
    deltax = (x_QLC[1]-x_QLC[0])

    # Compute sigmaI_QLC and nu_kin_mlyperus
    sigmaI_QLC = sigmaI_corner*(c_r*(x_QLC/L)**2+1-c_r)
    nu_kin_mlyperus = nu_kin/h_pr
    nu_kin_mlyperus.ito('1/microsecond')

    # Specify runtime values
    runtime_units = 'ms'
    runtime = AssignQuantity(runtime,runtime_units)
    runtime.ito('microsecond')

    # Specify the time interval and initial conditions
    tkeep_1Darr = np.linspace(0,runtime,num_steps)
    Ntot_init_1D = np.ones(nx_crystal)
    NQLL_init_1D = getNQLL(Ntot_init_1D,Nstar,Nbar) 

    if verbose:
        # Print parameter information
        print('From the GrowthInstability (GI) namelist:')
        print('sigmaI_corner =', sigmaI_corner)
        print('sigma0 =',sigma0)
        print('c_r =',c_r)
        print('Nbar', Nbar) 
        print('Nstar', Nstar)
        print('h_pr =', h_pr)
        print('D =', D)
        print('nu_kin =', nu_kin)
        print('L =', L)
        print('nx (crystal) =', nx_crystal)
        print('t_eq =',t_eq)
        print('odemethod =', odemethod)
        print('Spacing of points on the ice surface =', deltax)
        print('runtime =', runtime, "\n")

    print('This is a run from time', tkeep_1Darr[0].to('msec'),'to', tkeep_1Darr[-1].to('msec'))
    print('dt =', tkeep_1Darr[1]-tkeep_1Darr[0])

    # Generate reference solution
    Ntotkeep_1D_nondimensional, NQLLkeep_1D_nondimensional = run_f1d_dimensionless(\
        NQLL_init_1D,Ntot_init_1D,tkeep_1Darr,\
        Nbar, 
        Nstar, 
        sigma0.magnitude, 
        nu_kin_mlyperus.magnitude,
        deltax.magnitude,
        D.magnitude,
        t_eq.magnitude, 
        sigmaI_QLC,\
        AssignQuantity,\
        verbose=0, odemethod='RK45')
    
    
    Nicekeep_1D_nondimensional = Ntotkeep_1D_nondimensional-NQLLkeep_1D_nondimensional

    val = np.asarray(a=(Ntotkeep_1D_nondimensional, NQLLkeep_1D_nondimensional, Nicekeep_1D_nondimensional))
    
    return val

def getNQLL(Ntot,Nstar,Nbar):
    return Nbar - Nstar*np.sin(2*np.pi*Ntot)

def f1d_solve_ivp_dimensionless(t, y, scalar_params, sigmaI, j2_list):
    Nbar, Nstar, sigma0, omega_kin, deltax, D, t_0 = scalar_params
    l = int(len(y)/2)
    NQLL0 = y[:l]
    Ntot0 = y[l:]

    # Diffusion term based on FT
    Dcoefficient1 =   4 * np.pi**2 / (deltax * l)**2  #print('Dcoefficient1', Dcoefficient1)
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
    dNQLL_dt = dNtot_dt - (NQLL0 - (Nbar - Nstar*np.sin(2*np.pi*Ntot0)))
    
    # Package for output
    return np.concatenate((dNQLL_dt, dNtot_dt))

def run_f1d_dimensionless(\
           NQLL_init_1D,Ntot_init_1D,times,\
           Nbar, Nstar, sigma0, nu_kin_mlyperus, deltaX, D, t_eq, sigmaI,\
           AssignQuantity,\
           verbose=0, odemethod='LSODA'):
    
    #convert times to nondimensional tau
    times_nondim = times / t_eq
    deltaX_nondim = deltaX / (np.sqrt(D * t_eq))

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
    omega_kin = nu_kin_mlyperus * t_eq

    # Bundle params for ODE solver
    scalar_params = np.array(\
        [Nbar, Nstar, sigma0, omega_kin, deltaX_nondim, D, t_eq])
    
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
    
    return Ntotkeep_1D, NQLLkeep_1D