import numpy as np
from copy import copy as cp
import matplotlib.pylab as plt
from scipy.integrate import solve_ivp
from numba import njit, float64, int32, types
from matplotlib import rcParams
from time import time
from scipy.fft import fft, ifft, rfft, irfft, fftfreq
from scipy.interpolate import CubicSpline
myinterpolator = CubicSpline

ticklabelsize = 15
linewidth = 1
fontsize = 15
titlefontsize = 8
color = 'k'
markersize = 10

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

def get_nu_kin(T,AssignQuantity):
    """ Hertz-Knudsen deposition velocity """

    # Reference values
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

@njit
def get_alpha(beta,beta_trans,delta_beta):
    alpha = 1-1/(1+np.exp(-(beta-beta_trans)/delta_beta))
    return alpha

@njit
def getNQLL(Ntot,Nstar,Nbar):
    return Nbar - Nstar*np.sin(2*np.pi*Ntot)

@njit
def getDeltaNQLL(Ntot,Nstar,Nbar,NQLL):
    return NQLL - (Nbar - Nstar*np.sin(2*np.pi*Ntot))

@njit
def pypr_getDeltaNQLL(Ntot_pr,Ntot_pyneg,Ntot_pypos,alpha_pr,alpha_pyneg,alpha_pypos,Nstar_pr,Nstar_py,Nbar,NQLL_pr):
    NQLL_eq_pr    = Nbar - Nstar_pr*np.sin(2*np.pi*Ntot_pr)
    NQLL_eq_pyneg = Nbar - Nstar_py*np.sin(2*np.pi*Ntot_pyneg)
    NQLL_eq_pypos = Nbar - Nstar_py*np.sin(2*np.pi*Ntot_pypos)
    NQLL_eq = alpha_pr*NQLL_eq_pr + alpha_pyneg*NQLL_eq_pyneg + alpha_pypos*NQLL_eq_pypos
    return NQLL_pr - NQLL_eq

def pypr_solve_ivp(t, y, scalar_params, sigmaI, j_list, j2_list, x_QLC):
    Nbar, Nstar_pr, Nstar_py, sigma0_pr, sigma0_py, nu_kin_mlyperus, DoverdeltaX2, tau_eq, \
    theta, beta_trans, delta_beta, \
    h_pr, h_py, microfacets = scalar_params
    l = int(len(y)/2)
    NQLL0 = NQLL_pr = y[:l]
    Ntot0 = Ntot_pr = y[l:]
    
    # Using numpy's gradient method to get the first derivative of (scaled) Ntot
    z_pr = h_pr * Ntot_pr
    dx = x_QLC[1]-x_QLC[0]
    beta = np.gradient(z_pr,dx)
    
    # Deposition from air
    if microfacets == 1.0:
        # Calculating the weights
        alpha_pyneg = get_alpha(beta,-beta_trans,delta_beta)
        alpha_pypos = 1-get_alpha(beta, beta_trans,delta_beta)
        alpha_pr = 1 - alpha_pyneg - alpha_pypos
 
        # Ntot deposition
        m_pr = (NQLL0 -(Nbar-Nstar_pr))/(2*Nstar_pr)
        sigma_m_pr = (sigmaI - m_pr * sigma0_pr)    
        m_py = (NQLL0 -(Nbar-Nstar_py))/(2*Nstar_py); 
        sigma_m_py = (sigmaI - m_py * sigma0_py)
        sigma_m = alpha_pyneg*sigma_m_py + alpha_pypos*sigma_m_py + alpha_pr*sigma_m_pr  
    else:
        m_pr = (NQLL0 -(Nbar-Nstar_pr))/(2*Nstar_pr)
        sigma_m = (sigmaI - m_pr * sigma0_pr) 
    dNtot_dt = nu_kin_mlyperus * sigma_m

    # Add in surface diffusion (based on FT)
    angles = np.arctan(beta) # Default is radians
    costerm = np.cos(angles)
    Dcoefficient1 = 4*DoverdeltaX2/l**2*np.pi**2 *costerm**2
    bj_list = rfft(NQLL0)
    cj_list = bj_list*j2_list
    dy = -Dcoefficient1  * irfft(cj_list)
    dNtot_dt += dy

    # NQLL
    if microfacets == 1.0:
        Ntot_pyneg = 1/h_py * (np.cos(theta)*h_pr* Ntot_pr -np.sin(theta)*x_QLC)
        Ntot_pypos = 1/h_py * (np.cos(theta)*h_pr* Ntot_pr +np.sin(theta)*x_QLC)
        dNQLL_dt = dNtot_dt - pypr_getDeltaNQLL(\
            Ntot_pr,Ntot_pyneg,Ntot_pypos,alpha_pr,alpha_pyneg,alpha_pypos,Nstar_pr,Nstar_py,Nbar,NQLL_pr)\
            /tau_eq
    else:
        dNQLL_dt = dNtot_dt - getDeltaNQLL(Ntot0,Nstar_pr,Nbar,NQLL0)/tau_eq
    
    # Package for output
    return np.concatenate((dNQLL_dt, dNtot_dt))


def smoothout(x_QLC,Ntot_pr,deltax,d2Ntot_dx2_threshold,verbose=0):
    dNtot_dx = np.gradient(Ntot_pr,deltax)#; print(dNtot_dx.units)
    d2Ntot_dx2 = np.gradient(dNtot_dx,deltax)#; print(d2Ntot_dx2.units)
    ismoothlist = np.argwhere(d2Ntot_dx2<-d2Ntot_dx2_threshold)
    ismoothlist = ismoothlist.reshape(-1)
    if verbose > 0:
        print('Shape of the smooth list is ', ismoothlist.shape)
    Ntot_pr_smoothed = np.copy(Ntot_pr)
    nbefore = 2; #print(nbefore)
    nafter = nbefore+1; #print(nafter)
    nx = len(x_QLC)

    for ismooth in ismoothlist:
        if ismooth >= nbefore and ismooth <= nx-nafter:
            x = x_QLC[ismooth-nbefore:ismooth+nafter]; #print("here is x", x)
            x = np.delete(x,nbefore); #print("here is x", x)
            y = Ntot_pr[ismooth-nbefore:ismooth+nafter]; #print("here is y",y)
            y = np.delete(y,nbefore); #print("here is y", y)
            spl = myinterpolator(x,y)
            ynew = spl(x[nbefore]) # same as spl(x_QLC[ismooth])
            Ntot_pr_smoothed[ismooth] = ynew

        elif ismooth == nx-2:
            if verbose > 0:
                print('Linear smoothing at the end of the array')
            Ntot_pr_smoothed[ismooth] = (Ntot_pr[-3]+Ntot_pr[-1])/2

        elif ismooth == 1:
            if verbose > 0:
                print('Linear smoothing at the beginning of the array')
            Ntot_pr_smoothed[ismooth] = (Ntot_pr[0]+Ntot_pr[2])/2
                 
    return d2Ntot_dx2, Ntot_pr_smoothed

def run_pypr(\
           NQLL_init_1D,Ntot_init_1D,times,\
           Nbar, Nstar, sigma0, nu_kin_mlyperus, Doverdeltax2, tau_eq, \
           theta, beta_trans_factor, Nstarfactor, h_pr, h_pyfactor, sigma0factor,\
           sigmaI, x_QLC, d2Ntot_dx2_threshold,\
           AssignQuantity,\
           verbose=0, odemethod='RK45', microfacets=0):

    """ Solves the QLC-2 problem with pyramidal as well as prismatic facet possibilities. """

    # Prep for the integration
    nt = len(times)
    nx = len(NQLL_init_1D)
    ylast = np.array([NQLL_init_1D,Ntot_init_1D])
    ylast = np.reshape(ylast,2*nx)
    ykeep_1D = [ylast]
    lastprogress = 0
    sigmaI_mag = sigmaI.magnitude
    x_QLC_mag = x_QLC.magnitude
    t1 = time()
    bj_list = rfft(NQLL_init_1D)
    j_list = np.array([j for j in range(len(bj_list))])
    j2_list = np.array(j_list)**2
    
    # Integration prep having to do with multiple microfacets
    theta.ito('radian')
    beta_trans = np.sin(theta/2)/np.cos(theta/2)
    delta_beta = beta_trans/beta_trans_factor
    h_pr.ito('micrometer')
    h_py = h_pr*h_pyfactor
    Nstar_pr = Nstar
    Nstar_py = Nstar_pr*Nstarfactor
    sigma0_pr = sigma0.magnitude
    sigma0_py = sigma0.magnitude*sigma0factor
    
    # This is prep for attempting to smooth Ntot
    deltax_mag = x_QLC_mag[1]-x_QLC_mag[0]
    d2Ntot_dx2_threshold_mag = d2Ntot_dx2_threshold.magnitude

    # Bundle parameters for ODE solver
    scalar_params = np.array(\
      [Nbar, Nstar_pr, Nstar_py, sigma0_pr, sigma0_py, nu_kin_mlyperus.magnitude, Doverdeltax2.magnitude, tau_eq.magnitude, \
       theta.magnitude, beta_trans.magnitude, delta_beta.magnitude,
       h_pr.magnitude, h_py.magnitude, microfacets])

    # Loop over times
    for i in range(0,nt-1):
                
        # Specify the time interval of this step
        tinterval = [times[i].magnitude,times[i+1].magnitude]
        
        # Integrate up to next time step
        sol = solve_ivp(\
            pypr_solve_ivp, tinterval, ylast, args=(scalar_params, sigmaI_mag, j_list, j2_list, x_QLC_mag), \
            rtol=1e-12, method=odemethod) 
        ylast = sol.y[:,-1]
        
        # Smoothing
        ylast_1Darray = np.array(ylast, np.float64); 
        ylast_1Darray_reshaped = np.reshape(ylast_1Darray,(2,nx))
        Ntot_pr = ylast_1Darray_reshaped[1,:]; #print('Ntot_pr has shape', np.shape(Ntot_pr))
        NQLL_pr = ylast_1Darray_reshaped[0,:]; #print('NQLL_pr has shape', np.shape(NQLL_pr))
        d2Ntot_dx2, Ntot_pr_smoothed = smoothout(x_QLC_mag,Ntot_pr,deltax_mag,d2Ntot_dx2_threshold_mag,verbose)
        ylast = np.array([NQLL_pr,Ntot_pr_smoothed])
        ylast = np.reshape(ylast,2*nx)

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

# @njit("f8[:](f8,f8[:],f8[:],f8[:])")
def f1d_solve_ivp(t, y, scalar_params, sigmaI, j2_list):
    Nbar, Nstar, sigma0, nu_kin_mlyperus, DoverdeltaX2, tau_eq = scalar_params
    l = int(len(y)/2)
    NQLL0 = y[:l]
    Ntot0 = y[l:]
    
    # Ntot deposition
    m = (NQLL0 - (Nbar - Nstar))/(2*Nstar)
    sigma_m = (sigmaI - m * sigma0)
    dNtot_dt = nu_kin_mlyperus * sigma_m

    # Ntot diffusion in x-space (replaced by the FT code below this)
#     dy = np.empty(np.shape(NQLL0))
#     for i in range(1,len(NQLL0)-1):
#         dy[i] = DoverdeltaX2*(NQLL0[i-1]-2*NQLL0[i]+NQLL0[i+1])
#     dy[0]  = DoverdeltaX2*(NQLL0[-1] -2*NQLL0[0] +NQLL0[1]) # Periodic BC
#     dy[-1] = DoverdeltaX2*(NQLL0[-2] -2*NQLL0[-1]+NQLL0[0])

    # Diffusion term based on FT
    Dcoefficient1 = 4*DoverdeltaX2/l**2*np.pi**2; #print('Dcoefficient1', Dcoefficient1)
    bj_list = rfft(NQLL0)
    cj_list = bj_list*j2_list
    dy = -Dcoefficient1  * irfft(cj_list)

    # Combined
    dNtot_dt += dy

    # NQLL    
    dNQLL_dt = dNtot_dt - getDeltaNQLL(Ntot0,Nstar,Nbar,NQLL0)/tau_eq
    
    # Package for output
    return np.concatenate((dNQLL_dt, dNtot_dt))

def run_f1d(\
           NQLL_init_1D,Ntot_init_1D,times,\
           Nbar, Nstar, sigma0, nu_kin_mlyperus, Doverdeltax2, tau_eq, sigmaI,\
           AssignQuantity,\
           verbose=0, odemethod='LSODA'):

    """ Solves the QLC-2 problem. Branched from the code in diffusionstuff11.py, it has units """

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

    # Bundle parameters for ODE solver
    scalar_params = np.array(\
            [Nbar, Nstar, sigma0.magnitude, nu_kin_mlyperus.magnitude, Doverdeltax2.magnitude, tau_eq.magnitude])

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
            f1d_solve_ivp, tinterval, ylast, args=(scalar_params, sigmaI_mag, j2_list), \
            rtol=1e-12, method=odemethod) 
        ylast = sol.y[:,-1]
        
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

@njit
def f0d_solve_ivp(t, y, scalar_params, sigmaIcorner):
    Nbar, Nstar, sigma0, nu_kin_mlyperus, tau_eq = scalar_params  # unpack parameters
    NQLL0 = y[0]
    Ntot0 = y[1]      # unpack current values of y

    # Ntot deposition
    twopi = 2*np.pi
    m = (NQLL0 - (Nbar - Nstar))/(2*Nstar)
    sigma_m = (sigmaIcorner - m * sigma0)
    dNtot_dt = nu_kin_mlyperus * sigma_m
    
    # NQLL
    dNQLL_dt = dNtot_dt - getDeltaNQLL(Ntot0,Nstar,Nbar,NQLL0)/tau_eq
    
    # Packaging up for output
    derivs = [dNQLL_dt, dNtot_dt]
    return derivs

def run_f0d(NQLL_init_0D, Ntot_init_0D, times,\
            Nbar, Nstar, sigma0, nu_kin_mlyperus, tau_eq, sigmaI_corner,
            verbose=0, odemethod='LSODA'):
    
    # Prep for the integration
    scalar_params = np.array([Nbar, Nstar, sigma0, nu_kin_mlyperus.magnitude, tau_eq.magnitude])
    ylast = np.array([NQLL_init_0D,Ntot_init_0D])
    ykeep_0D = [ylast]
    lastprogress = 0
    
    nt = len(times)
    for i in range(0,nt-1):

        # Specify the time interval of this step
        tinterval = [times[i].magnitude,times[i+1].magnitude]
        
        # Integrate up to next time step
        sol = solve_ivp(\
              f0d_solve_ivp, tinterval, ylast, dense_output=True, args=(scalar_params,sigmaI_corner.magnitude),\
              rtol=1e-12,method=odemethod)
        ylast = sol.y[:,-1]

        # Stuff into keeper arrays
        ykeep_0D.append(ylast)
        
        # Progress reporting
        progress = int(i/nt*100)
        if np.mod(progress,10) == 0:
            if progress > lastprogress:
                #print(progress,'% done')
                lastprogress = progress

    #print('100% done')
    ykeep_0D = np.array(ykeep_0D, np.float64)
    NQLLkeep_0D = ykeep_0D[:,0]
    Ntotkeep_0D = ykeep_0D[:,1]

    return Ntotkeep_0D, NQLLkeep_0D          

def get_D_of_T(T,AssignQuantity):
    """ Based on a log/inverse T fit to Price's data for supercooled liquid water """
    E_a =  AssignQuantity(22.83465640608,'kilojoule / mole')
    R = AssignQuantity(8.314e-3,'kjoule/mol/K')
    T_o = AssignQuantity(273,'K')
    D_o = AssignQuantity(0.0009201878841272197,'micrometer ** 2 / microsecond')    
    arg_of_exp = -E_a/R * (1/T-1/T_o)
    D = D_o * np.exp(arg_of_exp)
    return D

def report_0d_growth_results(\
         tkeep_0Darr,NQLLkeep_0D,Ntotkeep_0D,Nicekeep_0D,Nbar,Nstar,nmpermonolayer, \
         graphics=True,itime=-1):
    
    # Growth statistics
    delta_N = Ntotkeep_0D[itime]-Ntotkeep_0D[0]
    delta_t = tkeep_0Darr[itime]-tkeep_0Darr[0]
    g_ice_QLC = delta_N/delta_t*nmpermonolayer; g_ice_QLC.ito('micrometer/second')
    
    # Plot results
    if graphics:
        plt.figure()
        rcParams['xtick.labelsize'] = ticklabelsize 
        rcParams['ytick.labelsize'] = ticklabelsize
        plt.plot(tkeep_0Darr.magnitude,NQLLkeep_0D,lw=linewidth,label='NQLL')
        plt.plot(tkeep_0Darr.magnitude,NQLLkeep_0D-getNQLL(Ntotkeep_0D,Nstar,Nbar),lw=linewidth,label='NQLL bias')
        plt.xlabel(r't ($\mu s$)',fontsize=fontsize)
        plt.ylabel(r'$N_{QLL} $',fontsize=fontsize)
        plt.grid('on')
        plt.legend()
    
    return g_ice_QLC


def report_1d_growth_results(\
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
    g_ice_QLC = delta_N/delta_t*nmpermonolayer; g_ice_QLC.ito('micrometer/second')
    
    return g_ice_QLC

def getsigmaI(x,center_reduction,sigmaIcorner):
    """ Assume x is already centered """
    sigmapfac = 1-center_reduction/100
#     xmid = max(x)/2
    fsig = x**2*(1-sigmapfac)+sigmapfac
    return fsig*sigmaIcorner

# Not sure we use either of these so commenting them out
# def getsigma_m(NQLL0,Nbar,Nstar,sigmaI,sigma0):
#     m = (NQLL0 - (Nbar - Nstar))/(2*Nstar)
#     sigma_m = (sigmaI - m * sigma0)
#     return sigma_m

# def pypr_getsigma_m(NQLL0,Nbar,Nstar,sigmaI,sigma0):
#     m = (NQLL0 - (Nbar - Nstar))/(2*Nstar)
#     sigma_m = (sigmaI - m * sigma0)
#     return sigma_m

# Commented out because I'm not sure we use it
# @njit
# def f1d_sigma_m(y, t, params):
#     Nbar, Nstar, sigmaI, sigma0, nu_kin_mlyperus, Doverdeltax2, nx = params
#     NQLL0, Ntot0 = np.reshape(y,(2,nx))      # unpack current values of y
    
#     # Deposition
#     m = (NQLL0 - (Nbar - Nstar))/(2*Nstar)
#     sigma_m = (sigmaI - m * sigma0)
#     return sigma_m

def getDofTpow(T,AssignQuantity):
    """ Returns D in micrometers^2/microsecond """
    """ Assumes temperature in degrees K """
    """ Looks like three parameters, but actually D0 = np.exp(b)*T0**m """
    """ Based on https://www.engineeringtoolbox.com/air-diffusion-coefficient-gas-mixture-temperature-d_2010.html """
    m = 1.86121271
    b = -7.35421981
    T0 = 273.15
    D0 = 21.91612692493907
    D = (T.magnitude/T0)**m * D0
    D = AssignQuantity(D,'micrometers^2/microsecond')
    return D

def getDofTP(T,P,AssignQuantity):
    """ Returns D in micrometers^2/microsecond """
    """ Assumes temperature in degrees K """
    """ Based on https://www.engineeringtoolbox.com/air-diffusion-coefficient-gas-mixture-temperature-d_2010.html """
    """ The pressure dependence is ~1/P """
    DofT = getDofTpow(T,AssignQuantity); # print(DofT)
    P0 = AssignQuantity(1,'atm') 
    D = DofT/(P.to('atm')/P0)
    return D

def fillin(un,ixbox,iybox,overrideflag=0,overrideval=0):
    border = cp(un[ixbox.start-1,iybox.start])
    if(overrideflag == 1):
        border = overrideval
    un[ixbox,iybox] = border
    return un

def run_f1d_FT(\
           NQLL_init_1D,Ntot_init_1D,times,\
           Nbar, Nstar, sigma0, nu_kin_mlyperus, DoverL2pi2, tau_eq, sigmaI,\
           AssignQuantity,\
           verbose=0, odemethod='RK45'):

    """ Solves the QLC problem, with units, in Fourier space """

    # Prep for the integration
    nt = len(times)
    lastprogress = 0
    sigmaI_mag = sigmaI.magnitude
    SigmaI_mag = rfft(sigmaI_mag)
    t1 = time()
    
    # This is to get j2_list because it's more efficient to pre-calculate it (but we'll use bj_list later too)
    bj_list = rfft(NQLL_init_1D)
    n_list = len(bj_list)
    j_list = np.array([j for j in range(n_list)])
    j2_list = np.array(j_list)**2
    
    l = int(len(NQLL_init_1D)/2)
    cos_series = 1
    for i in range(len(NQLL_init_1D) - 1):
        cos_series += np.cos((i + 1) * j_list * np.pi / l)
#     print('cos_series',cos_series)

    # Bundle parameters for ODE solver
    scalar_params = np.array([\
       Nbar, Nstar, sigma0.magnitude, nu_kin_mlyperus.magnitude, DoverL2pi2.magnitude, tau_eq.magnitude])

    # Package up the dynamical variables as FT 
    aj_list = rfft(Ntot_init_1D)
    Ylast = np.array([bj_list,aj_list])
    Ylast = np.reshape(Ylast,2*n_list)
    Ykeep_1D = [Ylast]
        
    # Loop over times
    for i in range(0,nt-1):
                
        # Specify the time interval of this step
        tinterval = [times[i].magnitude,times[i+1].magnitude]
        
        # Integrate up to next time step
        sol = solve_ivp(\
            f1d_solve_ivp_FT, tinterval, Ylast, args=(scalar_params, SigmaI_mag, j2_list), \
            rtol=1e-12, method=odemethod) 
        Ylast = sol.y[:,-1]
        
        # Stuff into keeper arrays
        Ykeep_1D.append(Ylast)
        
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
    
    # Packaging up the dynamical variables over time
    Ykeep_1Darr = np.array(Ykeep_1D, dtype=np.complex_)
    Ykeep_1Darr_reshaped = np.reshape(Ykeep_1Darr,(nt,2,n_list))
    Ykeep_Ntot_1D = Ykeep_1Darr_reshaped[:,1,:]
    Ykeep_NQLL_1D = Ykeep_1Darr_reshaped[:,0,:]
    
    # Convert to Cartesian values
    Ntotkeep_1D = irfft(Ykeep_Ntot_1D)
    NQLLkeep_1D = irfft(Ykeep_NQLL_1D)
    
    # Return the Cartesian values    
    return Ntotkeep_1D, NQLLkeep_1D

def f1d_solve_ivp_FT(t, Y, scalar_params, SigmaI, j2_list):

    # Unpack parameters
    Nbar, Nstar, sigma0, nu_kin_mlyperus, Dcoefficient1, tau_eq = scalar_params
    l = int(len(Y)/2)

    # Extract the dynamical variables
    bj_list = Y[:l]
    aj_list = Y[l:]
    
    # Convert some variables to position space
    NQLL0 = irfft(bj_list)
    Ntot0 = irfft(aj_list)
    nx_crystal = len(Ntot0)

    # Start with the diffusion term for Ntot
    cj_list = bj_list*j2_list
    daj_list_dt = -Dcoefficient1 * cj_list
    
    # Add in the deposition term
    M = bj_list/(2*Nstar)
    M[0] -= (Nbar - Nstar)/(2*Nstar)*nx_crystal  
    Sigma_m = (SigmaI - M * sigma0)
    Deposition_term = nu_kin_mlyperus * Sigma_m
    daj_list_dt += Deposition_term

    # Freezing/melting for NQLL (awkwardly done by reverse/forward FT -- but we may be stuck with this)
    deltaNQLL = getDeltaNQLL(Ntot0,Nstar,Nbar,NQLL0)
    DeltaNQLL = rfft(deltaNQLL)
    dbj_list_dt = daj_list_dt - DeltaNQLL/tau_eq

    # Package up and return
    return np.concatenate((dbj_list_dt, daj_list_dt))


# @njit
# def pypr_getNQLL(Ntot_pr,Ntot_pyneg,Ntot_pypos,alpha_pr,alpha_pyneg,alpha_pypos,Nstar_pr,Nstar_py,Nbar):
#     NQLL_eq_pr    = Nbar - Nstar_pr*np.sin(2*np.pi*Ntot_pr)
#     NQLL_eq_pyneg = Nbar - Nstar_py*np.sin(2*np.pi*Ntot_pyneg)
#     NQLL_eq_pypos = Nbar - Nstar_py*np.sin(2*np.pi*Ntot_pypos)
#     NQLL_eq = alpha_pr*NQLL_eq_pr + alpha_pyneg*NQLL_eq_pyneg + alpha_pypos*NQLL_eq_pypos
#     return NQLL_eq
    
#     Using a Fourier transform method to get the first derivative of (scaled) Ntot
#     Z_pr = rfft(z_pr)
#     dZpr_dx = 1j*Z_pr*j_list
#     L = x_QLC[-1]
#     beta = irfft(dZpr_dx)*np.pi/L


