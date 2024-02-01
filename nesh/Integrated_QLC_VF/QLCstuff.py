import numpy as np
from copy import copy as cp
import matplotlib.pylab as plt
from scipy.integrate import solve_ivp
from numba import njit, float64, int32, types


@njit
def propagate_vaporfield_Euler(u0,ixbox,iybox,udirichlet,uneumannx,uneumanny,Dxeff,Dyeff):
    
    # Diffusion
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

def VF2d(Temperature,Pressure,g_ice,sigmaI_far_field,Ldesired,AssignQuantity,verbose=0,Integration_method='Euler',tmax=0):
    
    # Times
    if tmax == 0:
        tmax = AssignQuantity(0.5,'microsecond')

    # Box size
    nx = 353
    ny = 355
    xmax = AssignQuantity(1000,'micrometer')
    ymax = AssignQuantity(1000,'micrometer')
    x = np.linspace(0,xmax,nx); dx = x[1]-x[0]; print('dx', dx)
    y = np.linspace(0,ymax,ny); dy = y[1]-y[0]; print('dy',dy)
    dx2 = dx**2
    dy2 = dy**2
    nxmid = int(nx/2); # print('nxmid =', nxmid); # print('x(nxmid) =',x[nxmid])
    nymid = int(ny/2)
    x = x-x[nxmid]
    y = y-y[nymid]

    # Compute diffusion coefficient of water through air at this temperature and pressure
    # This is using trends from engineering toolbox, with the log-log correction
    D = getDofTP(Temperature,Pressure,AssignQuantity)

    # Getting a suitable time step
    dt = (dx2+dy2)/D/10; print('dt = ', dt)

    # Computing effective diffusion coefficents (without dt)
    Dxeff = D/dx2; print('Dxeff = ', Dxeff)
    Dyeff = D/dy2; print('Dyeff = ', Dyeff)
    
    # Calculating the Neumann condition at the vapor/ice boundary (starting with ice density)
    rho_ice = AssignQuantity(0.9,'g/cm^3')
    Mvap = AssignQuantity(18,'g/mol')
    R = AssignQuantity(8.314,'J/mol/kelvin')

    # Neumann (w/o dt)
    uneumannx = rho_ice*g_ice*R*Temperature/(Mvap*dy); uneumannx.ito('pascal/microsecond')
    uneumanny = rho_ice*g_ice*R*Temperature/(Mvap*dx); uneumanny.ito('pascal/microsecond')
    print('uneumannx = ',uneumannx)
    print('uneumanny = ',uneumanny)
    
    # Converting this into pressures
    P3 = AssignQuantity(611,'Pa')
    T3 = AssignQuantity(273,'kelvin')
    Delta_H_sub = AssignQuantity(50,'kJ/mol')
    P_vapor_eq = P3*np.exp(-Delta_H_sub/R*(1/Temperature-1/T3)); print('Vapor pressure at this temperature = ', P_vapor_eq)

    # Dirichlet conditions at the far-field boundary
    udirichlet = P_vapor_eq*(sigmaI_far_field+1)
    print('udirichlet = ', udirichlet)
    
    # Shape of the crystal
    aspect_ratio = 1
    
    # Calculating how many time steps we'll do
    ntimes = int(tmax/dt)
    print('Integrating steps = ', ntimes)
    print('Integrating out to ', ntimes*dt) # This is a check -- it should be very close to the tmax specified above

    # Define the box inside
    Ldesiredx = Ldesired # Doesn't always work out to this because the grid is discretized
    boxradx = int(Ldesiredx/dx)
    Lx = boxradx*dx; 
    Ldesiredy = Ldesiredx*aspect_ratio
    boxrady = int(Ldesiredy/dy)
    Ly = boxrady*dy; 

    # Indices defining the crystal
    ixboxmin = nxmid-boxradx
    ixboxmax = nxmid+boxradx
    iyboxmin = nymid-boxrady
    iyboxmax = nymid+boxrady
    print('    box length (y) = ', iyboxmax-iyboxmin)

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
        print("Solving using "+Integration_method)
        uneumannx_Euler_mag = uneumannx_mag*dt.magnitude
        uneumanny_Euler_mag = uneumanny_mag*dt.magnitude
        Dxeff_Euler_mag = Dxeff_mag*dt.magnitude
        Dyeff_Euler_mag = Dyeff_mag*dt.magnitude
        for i in range(ntimes):
            un_mag = propagate_vaporfield_Euler(\
               un_mag,ixbox,iybox,udirichlet_mag,uneumannx_Euler_mag,uneumanny_Euler_mag,Dxeff_Euler_mag,Dyeff_Euler_mag)

    else:                
        print("Solving using "+Integration_method)
        # Dirichlet outer boundary
        un_mag[[0,-1],:]=udirichlet.magnitude
        un_mag[:,[0,-1]]=udirichlet.magnitude
        
        # This is the starting state
        ylast = np.reshape(un_mag,(nx*ny,1))
        print('shape of ylast =', np.shape(ylast))
        ylast = np.squeeze(ylast)
        print('shape of ylast =', np.shape(ylast))
        
        # Indices for the crystal inside
        ixmin = ixbox.start
        ixmax = ixbox.stop-1
        iymin = iybox.start
        iymax = iybox.stop-1
        
        # Packaging up parameters
        slice_params = np.array([ixmin,ixmax,iymin,iymax])
        integer_params = np.array([nx, ny])
        float_params = \
             np.array([udirichlet.magnitude, uneumannx.magnitude, uneumanny.magnitude, Dxeff.magnitude, Dyeff.magnitude])
        
        # Testing the derivative function ...
#         print('calling solve_ivp_VF2d ...')
#         solve_ivp_VF2d(0.0, ylast, slice_params, integer_params, float_params)

        # Integrating
        tinterval = [0.0,tmax.magnitude]
        print('calling solve_ivp ...')
        sol = solve_ivp(\
              solve_ivp_VF2d, tinterval, ylast, args=(slice_params, integer_params, float_params),\
              rtol=1e-8,method=Integration_method)
        ylast = sol.y[:,-1]
        un_mag = np.reshape(ylast,(nx,ny))
        
    # Re-dimensionalize
    un = AssignQuantity(un_mag,'pascal')
        
    # Now a slice just across one of the box surfaces (in the x dimension)
    uslicex = un[ixbox,nymid+boxrady]
    c_rx_percent = (max(uslicex)-min(uslicex))/uslicex[0]*100
    sigmaDx = uslicex/P_vapor_eq-1
    xshifted = x[ixbox]-x[nxmid]+dx/2

    # Now a slice just across one of the box surfaces (in the y dimension)
    uslicey = un[nxmid+boxradx, iybox]
    c_ry_percent = (max(uslicey)-min(uslicey))/uslicey[0]*100
    sigmaDy = uslicey/P_vapor_eq-1
    yshifted = y[iybox]-y[nymid]+dy/2
    
    # Filling in where the crystal is
    fillin(un,ixbox,iybox)

    # Reporting
    if verbose > 1:
        # Time reporting
        print('dt, tmax = ',dt, dt*ntimes)


        # Plotting from far afield up to the box
        iextend = 6
        fontsize = 25
        color = 'k'
        linewidth = 4
        markersize = 10

        ixbox_pre = slice(0,ixboxmin)
        ixbox_post = slice(ixboxmax,nx)
        plt.figure()
        plt.plot(x[ixbox_pre], un[ixbox_pre,nymid], 'blue')
        plt.plot(x[ixbox_post],un[ixbox_post,nymid],'blue')
        plt.xlabel('x')
        plt.title(Integration_method)
        plt.grid(True)

        iybox_pre = slice(0,iyboxmin)
        iybox_post = slice(iyboxmax,ny)
        plt.figure()
        plt.plot(y[iybox_pre], un[nxmid,iybox_pre], 'green')
        plt.plot(y[iybox_post],un[nxmid,iybox_post],'green')
        plt.xlabel('y')
        plt.title(Integration_method)
        plt.grid(True)

        # This is pressure right "above" the surface (i.e., the next y-bin)
        plt.figure()
        plt.plot(xshifted,uslicex,'ob',label='Just above the crystal',lw=linewidth,ms=markersize)
        p = np.polyfit(xshifted.magnitude,uslicex.magnitude,2); #print(p)
        xshifted_theory = np.linspace(min(xshifted),max(xshifted))
        plt.plot(xshifted_theory,np.polyval(p,xshifted_theory.magnitude),'-r',label='Parabolic fit',lw=linewidth)
        bigixbox = [ix for ix in range(nxmid-boxradx-iextend,nxmid+boxradx+iextend)]
        biguslice = un[bigixbox,nymid+boxrady]
        bigxshifted = x[bigixbox]-x[nxmid]+dx/2
        plt.plot(bigxshifted,biguslice,'xb', label='Away from crystal',lw=linewidth)
        plt.xlabel(r'$x$ ($\mu m$)', fontsize=fontsize)
        plt.ylabel(r'$P_{vap}$',fontsize=fontsize)
        plt.legend()
        plt.title(Integration_method)
        plt.grid(True)

        # This is supersaturation right "above" the surface (i.e., the next y-bin)
        plt.figure()        
        plt.plot(xshifted,sigmaDx,'ob', label='Above crystal',ms=markersize)
        plt.xlabel(r'$y$ ($\mu m$)', fontsize=fontsize)
        plt.ylabel(r'$\sigma_I(x)$',fontsize=fontsize)
        plt.legend()
        plt.title(Integration_method)
        plt.grid(True)

        # This is pressure right "to the right" of the surface (i.e., the next x-bin)
        plt.figure()
        plt.plot(yshifted,uslicey,'ob',label='Just to the right of the crystal',lw=linewidth,ms=markersize)
        p = np.polyfit(yshifted.magnitude,uslicey.magnitude,2); #print(p)
        yshifted_theory = np.linspace(min(yshifted),max(yshifted))
        plt.plot(yshifted_theory,np.polyval(p,yshifted_theory.magnitude),'-r',label='Parabolic fit',lw=linewidth)
        bigiybox = [iy for iy in range(nymid-boxrady-iextend,nymid+boxrady+iextend)]
        biguslice = un[nxmid+boxradx,bigiybox]
        bigyshifted = y[bigiybox]-y[nymid]+dy/2
        plt.plot(bigyshifted,biguslice,'xb', label='Away from crystal',lw=linewidth)
        plt.xlabel(r'$y$ ($\mu m$)', fontsize=fontsize)
        plt.ylabel(r'$P_{vap}$',fontsize=fontsize)
        plt.legend()
        plt.title(Integration_method)
        plt.grid(True)
        
        # Graph as contour plot
        fig,ax = plt.subplots()
        CS = ax.contour(x.magnitude,y.magnitude,un.T.magnitude)
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
        plt.title(Integration_method)
        plt.plot(xvec,yvec,color=color,linewidth=linewidth)
  
    # Return
    return [xshifted, sigmaDx], [yshifted, sigmaDy]

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
    DofT = getDofTpow(T,AssignQuantity); # print(DofT)
    P0 = AssignQuantity(1,'atm') 
    D = DofT/(P.to('atm')/P0)
    return D

def get_nu_kin(T,AssignQuantity):

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

