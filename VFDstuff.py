import numpy as np
from copy import copy as cp
import matplotlib.pylab as plt
from scipy.integrate import solve_ivp
from numba import njit, float64, int32, types
from matplotlib import rcParams
from time import time
from scipy.fft import fft, ifft, rfft, irfft, fftfreq

ticklabelsize = 15
linewidth = 1
fontsize = 15
titlefontsize = 8
color = 'k'
markersize = 10

def getDofT(T,AssignQuantity):
    """ This produces D in micrometers^2/microsecond """
    """ Assumes temperature in degrees K """

    m = 1.86121271
    b = -7.35421981
    logD = m*np.log(T.magnitude)+b
    D = np.exp(logD)
    D = AssignQuantity(D,'micrometers^2/microsecond')
    return D

def getDofTpow(T,AssignQuantity):
    """ This produces D in micrometers^2/microsecond """
    """ Assumes temperature in degrees K """

    m = 1.86121271
    b = -7.35421981
    T0 = 273
    D0 = np.exp(b)*T0**m; print('D0 = ', D0)
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

def propagate(u0,ixbox,iybox,udirichlet,uneumannx,uneumanny,Dxeff,Dyeff):
    
    # Diffusion
    un = np.empty(np.shape(u0))
    un[1:-1, 1:-1] = u0[1:-1, 1:-1] + ( \
    (u0[2:, 1:-1] - 2*u0[1:-1, 1:-1] + u0[:-2, 1:-1])*Dxeff + \
    (u0[1:-1, 2:] - 2*u0[1:-1, 1:-1] + u0[1:-1, :-2])*Dyeff )

    # Dirichlet outer boundary
    un[[0,-1],:]=udirichlet
    un[:,[0,-1]]=udirichlet
        
    # Pull out the stop and start indices
    ixmin = ixbox.start
    ixmax = ixbox.stop-1
    iymin = iybox.start
    iymax = iybox.stop-1

    # Inner boundary: diffusion and Neumann
    un[ixmin-1,iybox] = u0[ixmin-1,iybox] +(u0[ixmin-2,iybox] - u0[ixmin-1,iybox])*Dxeff -uneumannx
    un[ixmax+1,iybox] = u0[ixmax+1,iybox] +(u0[ixmax+2,iybox] - u0[ixmax+1,iybox])*Dxeff -uneumannx

    un[ixbox,iymin-1] = u0[ixbox,iymin-1] +(u0[ixbox,iymin-2] - u0[ixbox,iymin-1])*Dyeff -uneumanny
    un[ixbox,iymax+1] = u0[ixbox,iymax+1] +(u0[ixbox,iymax+2] - u0[ixbox,iymax+1])*Dyeff -uneumanny
        
    return un

def propagate_asymmetric(u0,ixbox,iybox,udirichlet,uneumannx_left,uneumannx_right,uneumanny,Dxeff,Dyeff):
    
    # Diffusion
    un = np.empty(np.shape(u0))
    un[1:-1, 1:-1] = u0[1:-1, 1:-1] + ( \
    (u0[2:, 1:-1] - 2*u0[1:-1, 1:-1] + u0[:-2, 1:-1])*Dxeff + \
    (u0[1:-1, 2:] - 2*u0[1:-1, 1:-1] + u0[1:-1, :-2])*Dyeff )

    # Dirichlet outer boundary
    un[[0,-1],:]=udirichlet
    un[:,[0,-1]]=udirichlet
        
    # Pull out the stop and start indices
    ixmin = ixbox.start
    ixmax = ixbox.stop-1
    iymin = iybox.start
    iymax = iybox.stop-1

    # Inner boundary: diffusion and Neumann
    un[ixmin-1,iybox] = u0[ixmin-1,iybox] +(u0[ixmin-2,iybox] - u0[ixmin-1,iybox])*Dxeff -uneumannx_left
    un[ixmax+1,iybox] = u0[ixmax+1,iybox] +(u0[ixmax+2,iybox] - u0[ixmax+1,iybox])*Dxeff -uneumannx_right

    un[ixbox,iymin-1] = u0[ixbox,iymin-1] +(u0[ixbox,iymin-2] - u0[ixbox,iymin-1])*Dyeff -uneumanny
    un[ixbox,iymax+1] = u0[ixbox,iymax+1] +(u0[ixbox,iymax+2] - u0[ixbox,iymax+1])*Dyeff -uneumanny
        
    return un

# xvec = (x[ixboxmax].magnitude,x[ixboxmax_new].magnitude)
# yvec = (y[iyboxmax_new].magnitude,0)
# plt.plot(xvec,yvec,color='k',linewidth=linewidth)
# xvec = (x[ixboxmin_new].magnitude,x[ixboxmin].magnitude)
# yvec = (0,y[iyboxmax_new].magnitude)
# plt.plot(xvec,yvec,color='b',linewidth=linewidth)
# xvec = (x[ixboxmin_new].magnitude,x[ixboxmin].magnitude)
# yvec = (0,y[iyboxmin_new].magnitude)
# plt.plot(xvec,yvec,color='r',linewidth=linewidth)
# xvec = (x[ixboxmax].magnitude,x[ixboxmax_new].magnitude)
# yvec = (y[iyboxmin_new].magnitude,0)
# plt.plot(xvec,yvec,color='g',linewidth=linewidth)
# xvec = (x[ixboxmin].magnitude,x[ixboxmax].magnitude)
# yvec = (y[iyboxmin_new].magnitude,y[iyboxmin_new].magnitude)
# plt.plot(xvec,yvec,color='c',linewidth=linewidth)
# xvec = (x[ixboxmin].magnitude,x[ixboxmax].magnitude)
# yvec = (y[iyboxmax_new].magnitude,y[iyboxmax_new].magnitude)
# plt.plot(xvec,yvec,color='y',linewidth=linewidth)

def propagate_hexagon(u0,ix01,ix12,ix32,ix43,ix54,ix50,iy01,iy12,iy32,iy43,iy54,iy50,udirichlet,uneumann,Deff):
    
    # Diffusion
    un = np.empty(np.shape(u0))
    un[1:-1, 1:-1] = u0[1:-1, 1:-1] + ( \
    (u0[2:, 1:-1] - 2*u0[1:-1, 1:-1] + u0[:-2, 1:-1])*Deff + \
    (u0[1:-1, 2:] - 2*u0[1:-1, 1:-1] + u0[1:-1, :-2])*Deff )

    # Dirichlet outer boundary
    un[[0,-1],:]=udirichlet
    un[:,[0,-1]]=udirichlet
        
    # Pull out the stop and start indices
#     ixmin = ixbox.start
#     ixmax = ixbox.stop-1
#     iymin = iybox.start
#     iymax = iybox.stop-1
#     ixmin_new = ixbox_new.start
#     ixmax_new = ixbox_new.stop-1
#     iymin_new = iybox_new.start
#     iymax_new = iybox_new.stop-1

    # Inner boundary: diffusion and Neumann
#     un[ixmin-1,iybox] = u0[ixmin-1,iybox] +(u0[ixmin-2,iybox] - u0[ixmin-1,iybox])*Deff -uneumann
#     un[ixmax+1,iybox] = u0[ixmax+1,iybox] +(u0[ixmax+2,iybox] - u0[ixmax+1,iybox])*Deff -uneumann

    un[ix43,iy43-1] = u0[ix43,iy43-1] +(u0[ix43,iy43-2] - u0[ix43,iy43-1])*Deff -uneumann
    un[ix01,iy01+1] = u0[ix01,iy01+1] +(u0[ix01,iy01+2] - u0[ix01,iy01+1])*Deff -uneumann
    un[ix12+1,iy12+1] = u0[ix12+1,iy12+1] +(u0[ix12+2,iy12+2] - u0[ix12+1,iy12+1])*Deff -uneumann
    un[ix32+1,iy32-1] = u0[ix32+1,iy32-1] +(u0[ix32+2,iy32-2] - u0[ix32+1,iy32-1])*Deff -uneumann
    un[ix54-1,iy54-1] = u0[ix54-1,iy54-1] +(u0[ix54-2,iy54-2] - u0[ix54-1,iy54-1])*Deff -uneumann
    un[ix50-1,iy50+1] = u0[ix50-1,iy50+1] +(u0[ix50-2,iy50+2] - u0[ix50-1,iy50+1])*Deff -uneumann
    
#     un[ixboxnew_right,iymax_new+1] = u0[ixboxnew_right,iymax_new+1] +(u0[ixboxnew_right,iymax_new+2] - u0[ixboxnew_right,iymax_new+1])*Deff -uneumann
#     un[ixbox,iymax_new+1] = u0[ixbox,iymax_new+1] +(u0[ixbox,iymax_new+2] - u0[ixbox,iymax_new+1])*Deff -uneumann
        
    return un

def propagate_vaporfield_Euler_x1d(u0,udirichlet,uneumann,Deff):
    
    # Diffusion ... indices [1:-1] exclude the first and the last ...
    un = np.empty(np.shape(u0))
    un[1:-1] = u0[1:-1] + (u0[2:] - 2*u0[1:-1] + u0[:-2]) *Deff

    # Dirichlet outer boundary
    un[-1]=udirichlet

    # Inner boundary: diffusion and Neumann
    un[0] = u0[0]
    un[0] += (u0[1] - u0[0])*Deff
    un[0] -= uneumann
    
    # Return
    return un

def VF2d_x1d(Temperature,Pressure,g_ice,sigmaI_far_field,L,\
         AssignQuantity,verbose=0,\
         tmax_mag=0.5, dt=0, nx=151, xmax_mag=1000):
    
    # Times
    tmax = AssignQuantity(tmax_mag,'microsecond')

    # Box size
    xmax = AssignQuantity(cmax_mag,'micrometer')
    x = np.linspace(L,xmax,nx); dx = x[1]-x[0]
    dx2 = dx**2

    # Compute diffusion coefficient of water through air at this temperature and pressure
    # This is using trends from engineering toolbox, with the log-log correction
    D = getDofTP(Temperature,Pressure,AssignQuantity)

    # Getting a suitable time step
    if dt == 0:
        dt = dx2/D/30

    # Computing effective diffusion coefficents (without dt)
    Deff = D/dr2
    
    # Calculating the Neumann condition at the vapor/ice boundary (starting with ice density)
    rho_ice = AssignQuantity(0.9,'g/cm^3')
    Mvap = AssignQuantity(18,'g/mol')
    R = AssignQuantity(8.314,'J/mol/kelvin')
    uneumann = rho_ice*g_ice*R*Temperature/(Mvap*dr); uneumann.ito('pascal/microsecond')
    if verbose>0:
        print('uneumann = ',uneumann)
    
    # Converting this into pressures
    P3 = AssignQuantity(611,'Pa')
    T3 = AssignQuantity(273,'kelvin')
    Delta_H_sub = AssignQuantity(50,'kJ/mol')
    P_vapor_eq = P3*np.exp(-Delta_H_sub/R*(1/Temperature-1/T3))
    if verbose > 0: print('Vapor pressure at this temperature = ', P_vapor_eq)

    # Dirichlet conditions at the far-field boundary
    udirichlet = P_vapor_eq*(sigmaI_far_field+1)
    if verbose > 0: print('udirichlet = ', udirichlet)
        
    # Calculating how many time steps we'll do
    ntimes = int(tmax/dt)
    if verbose > 0:
        print('Integrating steps = ', ntimes)
        print('Integrating out to ', ntimes*dt) # This is a check -- it should be very close to the tmax specified above

    # Initialize u0 and un as ones/zeros matrices 
    u0 = np.ones([nx])*udirichlet # starting u values        
    un_mag = u0.magnitude
    udirichlet_mag = udirichlet.magnitude
    uneumann_mag = uneumann.magnitude

    # Propagate forward a bunch of times
    uneumann_Euler_mag = uneumann_mag*dt.magnitude
    Deff_Euler_mag = Deff.magnitude*dt.magnitude
    x_mag = x.magnitude

    for i in range(ntimes):
        un_mag = propagate_vaporfield_Euler_x1d(\
             un_mag, udirichlet_mag, uneumann_Euler_mag,\
             Deff2_Euler_mag)
 
    # Re-dimensionalize
    un = AssignQuantity(un_mag,'pascal')
    
    # Return
    return [r,un]

def propagate_vaporfield_Euler_r1d(u0,udirichlet,uneumann,Deff1,Deff2,r):
    
    # Diffusion ... indices [1:-1] exclude the first and the last ...
    un = np.empty(np.shape(u0))
    un[1:-1] = u0[1:-1] + (u0[2:] - 2*u0[1:-1] + u0[:-2]) *Deff2
    un[1:-1] += 2 *(u0[1:-1]-u0[0:-2]) *Deff1/r[1:-1]

    # Dirichlet outer boundary
    un[-1]=udirichlet

    # Inner boundary: diffusion and Neumann
    un[0] = u0[0]
    un[0] +=   (u0[1] - u0[0])*Deff2
    un[0] += 2*(u0[1] - u0[0])*Deff1/r[0]
    un[0] -= uneumann
    
    return un

def VF2d_r1d(Temperature,Pressure,g_ice,sigmaI_far_field,L,\
         AssignQuantity,verbose=0,\
         tmax_mag=0.5, dt=0, nr=151, rmax_mag=1000):
    
    # Times
    tmax = AssignQuantity(tmax_mag,'microsecond')

    # Box size
    rmax = AssignQuantity(rmax_mag,'micrometer')
    r = np.linspace(L,rmax,nr); dr = r[1]-r[0]
    dr2 = dr**2

    # Compute diffusion coefficient of water through air at this temperature and pressure
    # This is using trends from engineering toolbox, with the log-log correction
    D = getDofTP(Temperature,Pressure,AssignQuantity)

    # Getting a suitable time step
    if dt == 0:
        dt = dr2/D/30

    # Computing effective diffusion coefficents (without dt)
    Deff1 = D/dr
    Deff2 = D/dr2
    
    # Calculating the Neumann condition at the vapor/ice boundary (starting with ice density)
    rho_ice = AssignQuantity(0.9,'g/cm^3')
    Mvap = AssignQuantity(18,'g/mol')
    R = AssignQuantity(8.314,'J/mol/kelvin')
    uneumann = rho_ice*g_ice*R*Temperature/(Mvap*dr); uneumann.ito('pascal/microsecond')
    if verbose>0:
        print('uneumann = ',uneumann)
    
    # Converting this into pressures
    P3 = AssignQuantity(611,'Pa')
    T3 = AssignQuantity(273,'kelvin')
    Delta_H_sub = AssignQuantity(50,'kJ/mol')
    P_vapor_eq = P3*np.exp(-Delta_H_sub/R*(1/Temperature-1/T3))
    if verbose > 0: print('Vapor pressure at this temperature = ', P_vapor_eq)

    # Dirichlet conditions at the far-field boundary
    udirichlet = P_vapor_eq*(sigmaI_far_field+1)
    if verbose > 0: print('udirichlet = ', udirichlet)
        
    # Calculating how many time steps we'll do
    ntimes = int(tmax/dt)
    if verbose > 0:
        print('Integrating steps = ', ntimes)
        print('Integrating out to ', ntimes*dt) # This is a check -- it should be very close to the tmax specified above

    # Initialize u0 and un as ones/zeros matrices 
    u0 = np.ones([nr])*udirichlet*.9999 # starting u values        
    un_mag = u0.magnitude
    udirichlet_mag = udirichlet.magnitude
    uneumann_mag = uneumann.magnitude

    # Propagate forward a bunch of times
    uneumann_Euler_mag = uneumann_mag*dt.magnitude
    Deff1_Euler_mag = Deff1.magnitude*dt.magnitude
    Deff2_Euler_mag = Deff2.magnitude*dt.magnitude
    r_mag = r.magnitude

    for i in range(ntimes):
        un_mag = propagate_vaporfield_Euler_r1d(\
             un_mag, udirichlet_mag, uneumann_Euler_mag,\
             Deff1_Euler_mag, Deff2_Euler_mag,\
             r_mag)
 
    # Re-dimensionalize
    un = AssignQuantity(un_mag,'pascal')
    
    # Return
    return [r,un]

                           
def propagate_vaporfield_Euler(u0,ixbox,iybox,udirichlet,uneumannx,uneumanny,Dxeff,Dyeff):
    
    # Diffusion ... indices [1:-1] exclude the first and the last
    un = np.empty(np.shape(u0))
    un[1:-1, 1:-1] = u0[1:-1, 1:-1] + ( \
    (u0[2:, 1:-1] - 2*u0[1:-1, 1:-1] + u0[:-2, 1:-1])*Dxeff + \
    (u0[1:-1, 2:] - 2*u0[1:-1, 1:-1] + u0[1:-1, :-2])*Dyeff )

    # Dirichlet outer boundary
    un[0,:]=udirichlet
    un[-1,:]=udirichlet
    un[:,0]=udirichlet
    un[:,-1]=udirichlet
        
    # Pull out the stop and start indices
    ixmin = ixbox.start
    ixmax = ixbox.stop-1
    iymin = iybox.start
    iymax = iybox.stop-1

    # Inner boundary: diffusion and Neumann
    un[ixmin-1,iybox] = u0[ixmin-1,iybox] +(u0[ixmin-2,iybox] - u0[ixmin-1,iybox])*Dxeff -uneumannx
    un[ixmax+1,iybox] = u0[ixmax+1,iybox] +(u0[ixmax+2,iybox] - u0[ixmax+1,iybox])*Dxeff -uneumannx

    un[ixbox,iymin-1] = u0[ixbox,iymin-1] +(u0[ixbox,iymin-2] - u0[ixbox,iymin-1])*Dyeff -uneumanny
    un[ixbox,iymax+1] = u0[ixbox,iymax+1] +(u0[ixbox,iymax+2] - u0[ixbox,iymax+1])*Dyeff -uneumanny
        
    return un
         
@njit
def solve_ivp_VF2d(t, y, slice_params, integer_params, float_params):
    
    # Parameters
    ixmin,ixmax,iymin,iymax = slice_params; #print(slice_params)
    nx, ny = integer_params; #print(integer_params)
    udirichlet, uneumannx, uneumanny, Dxeff, Dyeff = float_params; #print(float_params)
    
    # Reshape
    u0 = np.reshape(y,(nx,ny))

    # Diffusion
    dun_dt = np.empty(np.shape(u0))
    dun_dt[1:-1, 1:-1] = ( \
    (u0[2:, 1:-1] - 2*u0[1:-1, 1:-1] + u0[:-2, 1:-1])*Dxeff + \
    (u0[1:-1, 2:] - 2*u0[1:-1, 1:-1] + u0[1:-1, :-2])*Dyeff )

    # Dirichlet outer boundary
    dun_dt[0,:]=0
    dun_dt[-1,:]=0
    dun_dt[:,0]=0
    dun_dt[:,-1]=0
        
#   Make slices
    ixbox = slice(ixmin,ixmax)
    iybox = slice(iymin,iymax)

    # Inner boundary: diffusion and Neumann
    dun_dt[ixmin-1,iybox] = (u0[ixmin-2,iybox] - u0[ixmin-1,iybox])*Dxeff -uneumannx
    dun_dt[ixmax+1,iybox] = (u0[ixmax+2,iybox] - u0[ixmax+1,iybox])*Dxeff -uneumannx

    dun_dt[ixbox,iymin-1] = (u0[ixbox,iymin-2] - u0[ixbox,iymin-1])*Dyeff -uneumanny
    dun_dt[ixbox,iymax+1] = (u0[ixbox,iymax+2] - u0[ixbox,iymax+1])*Dyeff -uneumanny
    
    dun_dt = np.reshape(dun_dt,(nx*ny,))
    return dun_dt

def VF2DSquare(Temperature,Pressure,g_ice,sigmaI_far_field,Ldesired,\
         AssignQuantity,verbose=0,Integration_method='Euler',\
         tmax_mag=0.5, dt=0, aspect_ratio=1, nx=151, ny=151, xmax_mag=1000):
    """
    Numerical solution of the vapor field surrounding a square crystal (with Neumann conditions)
    inside a far-field Dirichlet box
    """
    
    # Times
    tmax = AssignQuantity(tmax_mag,'microsecond')

    # Box size
    xmax = AssignQuantity(xmax_mag,'micrometer')
    ymax = AssignQuantity(xmax_mag,'micrometer')
    x = np.linspace(0,xmax,nx); dx = x[1]-x[0]
    if verbose>0:
        print('dx', dx)
    y = np.linspace(0,ymax,ny); dy = y[1]-y[0]
    if verbose>0:
        print('dy',dy)
    dx2 = dx**2
    dy2 = dy**2
    nxmid = int(nx/2)
    nymid = int(ny/2)
    x = x-x[nxmid]
    y = y-y[nymid]

    # Compute diffusion coefficient of water through air at this temperature and pressure
    # This is using trends from engineering toolbox, with the log-log correction
    D = getDofTP(Temperature,Pressure,AssignQuantity)

    # Getting a suitable time step
    if dt == 0:
        dt = (dx2+dy2)/D/10
        if verbose>0:
            print('Using the default dt =', dt)
    else:
        if verbose>0:
            print('Using the user-specified dt =',dt)

    # Computing effective diffusion coefficents (without dt)
    Dxeff = D/dx2
    if verbose>0: print('Dxeff = ', Dxeff)
    Dyeff = D/dy2
    if verbose>0: print('Dyeff = ', Dyeff)
    
    # Calculating the Neumann condition at the vapor/ice boundary (starting with ice density)
    rho_ice = AssignQuantity(0.9,'g/cm^3')
    Mvap = AssignQuantity(18,'g/mol')
    R = AssignQuantity(8.314,'J/mol/kelvin')

    # Neumann (w/o dt)
    uneumannx = rho_ice*g_ice*R*Temperature/(Mvap*dy); uneumannx.ito('pascal/microsecond')
    uneumanny = rho_ice*g_ice*R*Temperature/(Mvap*dx); uneumanny.ito('pascal/microsecond')
    if verbose>0:
        print('uneumannx = ',uneumannx)
        print('uneumanny = ',uneumanny)
    
    # Converting this into pressures
    P3 = AssignQuantity(611,'Pa')
    T3 = AssignQuantity(273,'kelvin')
    Delta_H_sub = AssignQuantity(50,'kJ/mol')
    P_vap_eq = P3*np.exp(-Delta_H_sub/R*(1/Temperature-1/T3))
    if verbose > 0: print('Vapor pressure at this temperature = ', P_vap_eq)

    # Dirichlet conditions at the far-field boundary
    udirichlet = P_vap_eq*(sigmaI_far_field+1)
    if verbose > 0: print('udirichlet = ', udirichlet)
        
    # Calculating how many time steps we'll do
    ntimes = int(tmax/dt)
    if verbose > 0:
        print('Integrating steps = ', ntimes)
        print('Integrating out to ', ntimes*dt) # This is a check -- it should be very close to the tmax specified above

    # Define the box inside
    Ldesiredx = Ldesired # Doesn't always work out to this because the grid is discretized
    boxradx = int(Ldesiredx/dx)
    Lx = boxradx*dx
    if verbose > 0: print('    box Lx = ', Lx)

    Ldesiredy = Ldesiredx*aspect_ratio
    boxrady = int(Ldesiredy/dy)
    Ly = boxrady*dy
    if verbose > 0: print('    box Ly = ', Ly)

    # Indices defining the crystal
    ixboxmin = nxmid-boxradx
    ixboxmax = nxmid+boxradx
    iyboxmin = nymid-boxrady
    iyboxmax = nymid+boxrady
    if verbose > 0: print('    box length (y) = ', iyboxmax-iyboxmin)

    # Setting up to slice through the volume
    ixbox = slice(ixboxmin,ixboxmax)
    if verbose>0: print(ixbox)
    iybox = slice(iyboxmin,iyboxmax)
    if verbose>0: print(iybox)

    # Initialize u0 and un as ones/zeros matrices 
    u0 = np.ones([nx, ny])*udirichlet # starting u values
    if verbose > 0: 
        nxtest, nytest = np.shape(u0)
        print('Shape of u0:')
        print('   nx =', nxtest)
        print('   ny =', nytest)
        
    un_mag = u0.magnitude
    udirichlet_mag = udirichlet.magnitude
    uneumannx_mag = uneumannx.magnitude
    uneumanny_mag = uneumanny.magnitude
    Dxeff_mag = Dxeff.magnitude
    Dyeff_mag = Dyeff.magnitude

    # Propagate forward a bunch of times
    if Integration_method == 'Euler':
        if verbose > 0: print("Solving using "+Integration_method)
        uneumannx_Euler_mag = uneumannx_mag*dt.magnitude
        uneumanny_Euler_mag = uneumanny_mag*dt.magnitude
        Dxeff_Euler_mag = Dxeff_mag*dt.magnitude
        Dyeff_Euler_mag = Dyeff_mag*dt.magnitude
        for i in range(ntimes):
            un_mag = propagate_vaporfield_Euler(\
               un_mag,ixbox,iybox,udirichlet_mag,uneumannx_Euler_mag,uneumanny_Euler_mag,Dxeff_Euler_mag,Dyeff_Euler_mag)

    else:                
        if verbose > 0: print("Solving using "+Integration_method)
        print('Not implemented')
#         # Dirichlet outer boundary
#         un_mag[[0,-1],:]=udirichlet.magnitude
#         un_mag[:,[0,-1]]=udirichlet.magnitude
        
#         # This is the starting state
#         ylast = np.reshape(un_mag,(nx*ny,1))
#         if verbose > 0: print('shape of ylast =', np.shape(ylast))
#         ylast = np.squeeze(ylast)
#         if verbose > 0: print('shape of ylast =', np.shape(ylast))
        
#         # Indices for the crystal inside
#         ixmin = ixbox.start
#         ixmax = ixbox.stop-1
#         iymin = iybox.start
#         iymax = iybox.stop-1
        
#         # Packaging up parameters
#         slice_params = np.array([ixmin,ixmax,iymin,iymax])
#         integer_params = np.array([nx, ny])
#         float_params = \
#              np.array([udirichlet.magnitude, uneumannx.magnitude, uneumanny.magnitude, Dxeff.magnitude, Dyeff.magnitude])
        
#         # Integrating
#         tinterval = [0.0,tmax.magnitude]
#         sol = solve_ivp(\
#               solve_ivp_VF2d, tinterval, ylast, args=(slice_params, integer_params, float_params),\
#               rtol=1e-8,method=Integration_method)
#         ylast = sol.y[:,-1]
#         un_mag = np.reshape(ylast,(nx,ny))
        
    # Re-dimensionalize
    un = AssignQuantity(un_mag,'pascal')
        
    # Now a slice just across one of the box surfaces (in the x dimension)
    uslicex = un[ixbox,nymid+boxrady]
    c_rx_percent = (max(uslicex)-min(uslicex))/uslicex[0]*100
    sigmaDx = uslicex/P_vap_eq-1
    xshifted = x[ixbox]-x[nxmid]+dx/2

    # Now a slice just across one of the box surfaces (in the y dimension)
    uslicey = un[nxmid+boxradx, iybox]
    c_ry_percent = (max(uslicey)-min(uslicey))/uslicey[0]*100
    sigmaDy = uslicey/P_vap_eq-1
    yshifted = y[iybox]-y[nymid]+dy/2
    
    # Filling in where the crystal is
    fillin(un,ixbox,iybox)

    # Reporting
    if verbose > 1:

        # Plotting from far afield up to the box
        iextend = 6
        fontsize = 25
        color = 'k'
        linewidth = 1
        markersize = 10

        ixbox_pre = slice(0,ixboxmin)
        ixbox_post = slice(ixboxmax,nx)
        plt.figure()
        plt.plot(x[ixbox_pre], un[ixbox_pre,nymid], 'blue')
        plt.plot(x[ixbox_post],un[ixbox_post,nymid],'blue')
        plt.xlabel('x')
        plt.grid(True)

        iybox_pre = slice(0,iyboxmin)
        iybox_post = slice(iyboxmax,ny)
        plt.figure()
        plt.plot(y[iybox_pre], un[nxmid,iybox_pre], 'green')
        plt.plot(y[iybox_post],un[nxmid,iybox_post],'green')
        plt.xlabel('y')
        plt.grid(True)

        # This is pressure right "above" the surface (i.e., the next y-bin)
        plt.figure()
        plt.plot(xshifted,uslicex,'ob',label='Just above the crystal',lw=linewidth,ms=markersize)
        bigixbox = [ix for ix in range(nxmid-boxradx-iextend,nxmid+boxradx+iextend)]
        biguslice = un[bigixbox,nymid+boxrady]
        bigxshifted = x[bigixbox]-x[nxmid]+dx/2
        plt.plot(bigxshifted,biguslice,'xb', label='Away from crystal',lw=linewidth)
        plt.xlabel(r'$x$ ($\mu m$)', fontsize=fontsize)
        plt.ylabel(r'$P_{vap}$',fontsize=fontsize)
        plt.legend()
        plt.grid(True)

        # This is supersaturation right "above" the surface (i.e., the next y-bin)
        plt.figure()        
        plt.plot(xshifted,sigmaDx,'ob', label='Above crystal',ms=markersize)
        p = np.polyfit(xshifted.magnitude,sigmaDx.magnitude,2); #print(p)
        xshifted_theory = np.linspace(min(xshifted),max(xshifted))
        plt.plot(xshifted_theory,np.polyval(p,xshifted_theory.magnitude),'-r',label='Parabolic fit',lw=linewidth)
        plt.xlabel(r'$y$ ($\mu m$)', fontsize=fontsize)
        plt.ylabel(r'$\sigma_I(x)$',fontsize=fontsize)
        plt.legend()
        plt.grid(True)

        # This is pressure right "to the right" of the surface (i.e., the next x-bin)
        plt.figure()
        plt.plot(yshifted,uslicey,'ob',label='Just to the right of the crystal',lw=linewidth,ms=markersize)
        bigiybox = [iy for iy in range(nymid-boxrady-iextend,nymid+boxrady+iextend)]
        biguslice = un[nxmid+boxradx,bigiybox]
        bigyshifted = y[bigiybox]-y[nymid]+dy/2
        plt.plot(bigyshifted,biguslice,'xb', label='Away from crystal',lw=linewidth)
        plt.xlabel(r'$y$ ($\mu m$)', fontsize=fontsize)
        plt.ylabel(r'$P_{vap}$',fontsize=fontsize)
        plt.legend()
        plt.grid(True)
        
        # Graph as contour plot
        fig,ax = plt.subplots()
        CS = ax.contour(x.magnitude+dx.magnitude/2,y.magnitude+dy.magnitude/2,un.T.magnitude)
        ax.set_xlabel(r'$x$ ($\mu m$)', fontsize=fontsize)
        ax.set_ylabel(r'$y$ ($\mu m$)', fontsize=fontsize)
        fig.colorbar(CS)
        xvec = (x[ixboxmin].magnitude,x[ixboxmin].magnitude)
        yvec = (y[iyboxmin].magnitude,y[iyboxmax].magnitude)
        plt.plot(xvec,yvec,color=color,linewidth=linewidth)
        xvec = (x[ixboxmax].magnitude,x[ixboxmax].magnitude)
        yvec = (y[iyboxmin].magnitude,y[iyboxmax].magnitude)
        plt.plot(xvec,yvec,color=color,linewidth=linewidth)
        xvec = (x[ixboxmin].magnitude,x[ixboxmax].magnitude)
        yvec = (y[iyboxmin].magnitude,y[iyboxmin].magnitude)
        plt.plot(xvec,yvec,color=color,linewidth=linewidth)
        xvec = (x[ixboxmin].magnitude,x[ixboxmax].magnitude)
        yvec = (y[iyboxmax].magnitude,y[iyboxmax].magnitude)
        plt.plot(xvec,yvec,color=color,linewidth=linewidth)
        ax.axis('equal')
  
    # Return
    return [xshifted, sigmaDx], [yshifted, sigmaDy], [x, y, un], [Lx, Ly]

def removedups(mylist):
    
    # Removes duplicates in mylist, preserving the order
    
    seen = set()
    no_dups = []
    for lst in mylist.T:

        # convert to hashable type
        current = tuple(lst)

        # If element not in seen, add it to both
        if current not in seen:
            no_dups.append(lst)
            seen.add(current)
        else:
            print('not keeping', current)
            
    npts_unique, dummy = np.shape(no_dups); #print(npts_unique)
    ixedges_unique = []
    iyedges_unique = []

    for i in range(npts_unique):
        ixedges_unique.append(no_dups[i][0])
        iyedges_unique.append(no_dups[i][1])

    no_dups = np.vstack((ixedges_unique,iyedges_unique))

    return no_dups, npts_unique

def removedups2(mylist1,mylist2):
    
    # Removes any tuple in mylist1 that appears in mylist2
    
    seen = set()
    for lst in mylist2.T:
        current = tuple(lst)
        seen.add(current)
    
    no_dups = []
    for lst in mylist1.T:

        # convert to hashable type
        current = tuple(lst)

        # If element not in seen, add it to both
        if current not in seen:
            no_dups.append(lst)
        else:
            print('not keeping', current)
            
    npts_unique, dummy = np.shape(no_dups); #print(npts_unique)
    ixedges_unique = []
    iyedges_unique = []

    for i in range(npts_unique):
        ixedges_unique.append(no_dups[i][0])
        iyedges_unique.append(no_dups[i][1])

    no_dups = np.vstack((ixedges_unique,iyedges_unique))

    return no_dups, npts_unique


def autoscale(ax=None, axis='y', margin=0.1):
    '''Autoscales the x or y axis of a given matplotlib ax object
    to fit the margins set by manually limits of the other axis,
    with margins in fraction of the width of the plot

    Defaults to current axes object if not specified.
    '''
    import matplotlib.pyplot as plt
    import numpy as np
    if ax is None:
        ax = plt.gca()
    newlow, newhigh = np.inf, -np.inf

    for artist in ax.collections + ax.lines:
        x,y = get_xy(artist)
        if axis == 'y':
            setlim = ax.set_ylim
            lim = ax.get_xlim()
            fixed, dependent = x, y
        else:
            setlim = ax.set_xlim
            lim = ax.get_ylim()
            fixed, dependent = y, x

        low, high = calculate_new_limit(fixed, dependent, lim)
        newlow = low if low < newlow else newlow
        newhigh = high if high > newhigh else newhigh

    margin = margin*(newhigh - newlow)

    setlim(newlow-margin, newhigh+margin)

def calculate_new_limit(fixed, dependent, limit):
    '''Calculates the min/max of the dependent axis given 
    a fixed axis with limits
    '''
    if len(fixed) > 2:
        mask = (fixed>limit[0]) & (fixed < limit[1])
        window = dependent[mask]
        low, high = window.min(), window.max()
    else:
        low = dependent[0]
        high = dependent[-1]
        if low == 0.0 and high == 1.0:
            # This is a axhline in the autoscale direction
            low = np.inf
            high = -np.inf
    return low, high

def get_xy(artist):
    '''Gets the xy coordinates of a given artist
    '''
    if "Collection" in str(artist):
        x, y = artist.get_offsets().T
    elif "Line" in str(artist):
        x, y = artist.get_xdata(), artist.get_ydata()
    else:
        raise ValueError("This type of object isn't implemented yet")
    return x, y

def Cartesian_position_to_index(x,L,dx):
    index = np.rint((x+L)/dx).astype(int)
    return(index)

def Cartesian_index_to_position(index,L,dx):
    position = index*dx - L
    return(position)

def Cartesian_propagate_hexagon(u0,udirichlet,uneumann,Deff,Dmatrix,iedges,npts_total):

    # Propagates the vapor field forward one time step
    
    # Initialize starting values
    un = np.zeros(np.shape(u0))

    # Diffusion throughout (this is wrong)
    un[1:-1, 1:-1] = u0[1:-1, 1:-1] + ( \
    ( (u0[2:, 1:-1] - u0[1:-1, 1:-1])*Dmatrix[1:,1:] + (u0[0:-2, 1:-1] -u0[1:-1, 1:-1])*Dmatrix[0:-1,1:] ) + \
    ( (u0[1:-1, 2:] - u0[1:-1, 1:-1])*Dmatrix[1:,1:] + (u0[1:-1, 0:-2] -u0[1:-1, 1:-1])*Dmatrix[1:,0:-1] ) )
    
    # Assigning Dirichlet conditions at the perimeter of the simulation box
    un[[0,-1],:] = udirichlet
    un[:,[0,-1]] = udirichlet
    
    # Assigning Neumann conditions above the crystal surface
    for i in range(npts_total):
        un[iedges[0,i],iedges[1,i]] -= uneumann
            
    return(un)