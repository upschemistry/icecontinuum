import numpy as np
from copy import copy as cp
import matplotlib.pylab as plt
from scipy.integrate import solve_ivp
from numba import njit, float64, int32, types
from matplotlib import rcParams
from time import time

# See about the Fourier components of some of these solutions
from scipy.fft import fft, ifft, rfft, irfft, fftfreq


ticklabelsize = 15
linewidth = 1
fontsize = 15
color = 'k'
markersize = 10

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
#     test = propagate_vaporfield_Euler_r1d(un_mag,udirichlet_mag,uneumann_Euler_mag,Deff1_Euler_mag,Deff2_Euler_mag,r_mag)
#     print(test)
#     return

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

def fillin(un,ixbox,iybox,overrideflag=0,overrideval=0):
    border = cp(un[ixbox.start-1,iybox.start])
    if(overrideflag == 1):
        border = overrideval
    un[ixbox,iybox] = border
    return un

def getsigma_m(NQLL0,Nbar,Nstar,sigmaI,sigma0):
    m = (NQLL0 - (Nbar - Nstar))/(2*Nstar)
    sigma_m = (sigmaI - m * sigma0)
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
    sigma_m = (sigmaI - m * sigma0)
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
    dNtot_dt = nu_kin_mlyperus * sigma_m

    # Ntot diffusion
    dy = np.empty(np.shape(NQLL0))
    for i in range(1,len(NQLL0)-1):
        dy[i] = DoverdeltaX2*(NQLL0[i-1]-2*NQLL0[i]+NQLL0[i+1])
    dy[0]  = DoverdeltaX2*(NQLL0[-1] -2*NQLL0[0] +NQLL0[1]) # Periodic BC
    dy[-1] = DoverdeltaX2*(NQLL0[-2] -2*NQLL0[-1]+NQLL0[0])

    # Diffusion term based on a zeroed-out FT
#     Dcoefficient1 = 4*DoverdeltaX2/l**2*np.pi**2; #print('Dcoefficient1', Dcoefficient1)
#     bj_list = rfft(NQLL0)
#     jmax = 130
#     j_list = np.array([j for j in range(len(bj_list))])
#     j2_list = np.array(j_list)**2
#     j2_list[(j_list>jmax)] = 0
#     cj_zeroed_list = bj_list*j2_list
#     dy = -Dcoefficient1  * irfft(cj_zeroed_list)

    # Diffusion term based on a truncated FT (but this runs into trouble because of its shorter length)
#     Dcoefficient1 = 4*DoverdeltaX2/l**2*np.pi**2; #print('Dcoefficient1', Dcoefficient1)
#     bj_list = rfft(NQLL0)
#     jmax = 130
#     bj_list_head = bj_list[0:jmax]; #print('head', aj_list_head)
#     bj_list_tail = bj_list[-jmax:-1]; #print('tail', aj_list_tail)
#     bj_tr_list = np.append(bj_list_head,bj_list_tail)
#     rescale2 = len(bj_tr_list)/len(bj_list)
#     cj_tr_list = bj_tr_list*[j**2 for j in range(len(bj_tr_list))]
#     dy = -Dcoefficient1 * irfft(cj_tr_list)* rescale2
    

    # Combined
    dNtot_dt += dy

    # NQLL    
    dNQLL_dt = dNtot_dt - getDeltaNQLL(Ntot0,Nstar,Nbar,NQLL0)/tau_eq
    
    # Package for output
    derivs = np.empty(2*l)
    derivs[:l] = dNQLL_dt
    derivs[l:] = dNtot_dt

    return derivs

# def run_f1d(\
#            NQLL_init_1D,Ntot_init_1D,times,\
#            Nbar, Nstar, sigma0, nu_kin_mlyperus, Doverdeltax2, tau_eq, sigmaI,\
#            AssignQuantity,\
#            verbose=0, odemethod='LSODA', max_step=max_step):
def run_f1d(\
           NQLL_init_1D,Ntot_init_1D,times,\
           Nbar, Nstar, sigma0, nu_kin_mlyperus, Doverdeltax2, tau_eq, sigmaI,\
           AssignQuantity,\
           verbose=0, odemethod='LSODA'):

    """ Solves the QLC-2 problem. Branched from the code in diffusionstuff11.py, it has units """

    # Bundle parameters for ODE solver
    scalar_params = np.array([Nbar, Nstar, sigma0, nu_kin_mlyperus.magnitude, Doverdeltax2.magnitude, tau_eq.magnitude])

    # Prep for the integration
    nt = len(times)
    nx = len(NQLL_init_1D)
    ylast = np.array([NQLL_init_1D,Ntot_init_1D])
    ylast = np.reshape(ylast,2*nx)
    ykeep_1D = [ylast]
    lastprogress = 0
    sigmaI_mag = sigmaI.magnitude
    t1 = time()

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
#         sol = solve_ivp(\
#             f1d_solve_ivp, tinterval, ylast, args=(scalar_params,sigmaI_mag), \
#             rtol=1e-12, method=odemethod, max_step=max_step) 
        sol = solve_ivp(\
            f1d_solve_ivp, tinterval, ylast, args=(scalar_params,sigmaI_mag), \
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
         graphics=True,itime=-1,tgraphics=True,xlim=[]):
    
    # Parameters of the data
    ntimes = len(NQLLkeep_1D)

    if graphics:
        
        # Titles on graphs
        title_entire = title_params

        # Plot ice and total profile
        plt.figure()
        plt.plot(x_QLC.magnitude, Nicekeep_1D[itime,:], 'k', label='ice', lw=linewidth)
        plt.plot(x_QLC.magnitude, Ntotkeep_1D[itime,:], 'b', label='total', lw=linewidth)
        plt.xlabel('$x \ (\mu m$)',fontsize=fontsize)
        plt.ylabel('$ice \ & \ liquid \ layers$',fontsize=fontsize)
        rcParams['xtick.labelsize'] = ticklabelsize 
        rcParams['ytick.labelsize'] = ticklabelsize
        plt.legend()
        plt.title(title_entire,fontsize=fontsize)
        plt.grid('on')
        if len(xlim) > 0:
            plt.xlim(xlim)
            i = np.where( (x_QLC.magnitude > xlim[0]) &  (x_QLC.magnitude < xlim[1]) )[0]
            plt.ylim( Nicekeep_1D[itime,:][i].min(), Ntotkeep_1D[itime,:][i].max() ) 

        # Plot liquid
        plt.figure()
        plt.plot(x_QLC.magnitude, NQLLkeep_1D[itime,:], 'b', label='liquid', lw=linewidth)
        plt.xlabel('$x \ (\mu m$)',fontsize=fontsize)
        plt.ylabel('$liquid \ layers$',fontsize=fontsize)
        rcParams['xtick.labelsize'] = ticklabelsize 
        rcParams['ytick.labelsize'] = ticklabelsize
        plt.title(title_entire,fontsize=fontsize)
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
        plt.title(title_entire,fontsize=fontsize)
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


