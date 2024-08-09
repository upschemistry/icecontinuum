from pint import UnitRegistry; AssignQuantity = UnitRegistry().Quantity
import numpy as np
import matplotlib.pylab as plt

import sys
sys.path.append('..')
import QLCstuff2 as QLC_nonDim
import f90nml

# Define matplotlib parameters
ticklabelsize = 15
linewidth = 1
fontsize = 15
titlefontsize = 8
markersize = 10

# Read in GI parameters
inputfile = "GI parameters - Reference limit cycle (for testing).nml"
GI=f90nml.read(inputfile)['GI']
nx_crystal = GI['nx_crystal']
L = GI['L']
tau_eq = GI['tau_eq']

# Generate evenly spaced x values
X_QLC = np.linspace(-L,L,nx_crystal)


def generate_reference_solution(runtime = 50, num_steps = 51, verbose = False):
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
    
    inputfile = "GI parameters - Reference limit cycle (for testing).nml"

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

    # Crystal size -- needs to be an even number
    nx_crystal = GI['nx_crystal']

    # Time constant for freezing/thawing
    tau_eq = GI['tau_eq']
    tau_eq_units = GI['tau_eq_units']
    tau_eq = AssignQuantity(tau_eq,tau_eq_units)

    # Integration algorithm (possibilities: RK45, BDF, RK23, DOP853, LSODA, and Radau)
    odemethod = GI['odemethod']

    # Conversions (in case inputs are in other units)
    sigma0.ito('dimensionless')
    h_pr.ito('micrometer')
    D.ito('micrometer^2/microsecond')
    nu_kin.ito('micrometer/second')
    sigmaI_corner.ito('dimensionless')
    c_r.ito('dimensionless')
    tau_eq.ito('microsecond')
    
    # Calculate deltaX
    deltax = (X_QLC[1]-X_QLC[0])

    # Compute sigmaI_QLC and nu_kin_mlyperus
    sigmaI_QLC = sigmaI_corner*(c_r*(X_QLC/L)**2+1-c_r)
    nu_kin_mlyperus = nu_kin/h_pr
    nu_kin_mlyperus.ito('1/microsecond')

    # Specify runtime values
    runtime_units = 'ms'
    runtime = AssignQuantity(runtime,runtime_units)
    runtime.ito('microsecond')

    # Specify the time interval and initial conditions
    tkeep_1Darr = np.linspace(0,runtime,num_steps)
    Ntot_init_1D = np.ones(nx_crystal)
    NQLL_init_1D = QLC_nonDim.getNQLL(Ntot_init_1D,Nstar,Nbar) 

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
        print('tau_eq =',tau_eq)
        print('odemethod =', odemethod)
        print('Spacing of points on the ice surface =', deltax)
        print('runtime =', runtime, "\n")

    print('This is a run from time', tkeep_1Darr[0].to('msec'),'to', tkeep_1Darr[-1].to('msec'))
    print('dt =', tkeep_1Darr[1]-tkeep_1Darr[0])

    # Generate reference solution
    Ntotkeep_1D_nondimensional, NQLLkeep_1D_nondimensional = QLC_nonDim.run_f1d_dimensionless(\
        NQLL_init_1D,Ntot_init_1D,tkeep_1Darr,\
        Nbar, 
        Nstar, 
        sigma0.magnitude, 
        nu_kin_mlyperus.magnitude,
        deltax,
        D.magnitude,
        tau_eq.magnitude, 
        sigmaI_QLC,\
        AssignQuantity,\
        verbose=0, odemethod='RK45')
    
    
    Nicekeep_1D_nondimensional = Ntotkeep_1D_nondimensional-NQLLkeep_1D_nondimensional

    val = np.asarray(a=(Ntotkeep_1D_nondimensional, NQLLkeep_1D_nondimensional, Nicekeep_1D_nondimensional))
    
    return val


def plot_reference_vs_network(reference_solution, network_solution):
    """
    Plots reference solution against PINN solution for Ntot, Nqll, and N-ice.
    reference_solution and network_solution must be of the same shape.
    """
    labels = ["Ntot", "Nqll", "N-ice"]
    for i in range(len(labels)):
        plot_together(reference_solution[i][-1], network_solution[i][-1], X_QLC, labels[i])


def plot_together(expected_list, predicted_list, xlist, label = ""):
    """
    Plots expected output against predicted output.
    """
    plt.figure()
    plt.plot(xlist,expected_list,label="Expected "+label)
    plt.plot(xlist,predicted_list,label="Predicted "+label)
    plt.title(label="Expected vs Predicted "+label)
    plt.xlabel('x (micrometers)')
    plt.ylabel('Number of Ice Layers')
    plt.grid(True)
    plt.legend()

    plt.figure()
    plt.plot(xlist,(expected_list - predicted_list),label=label+" Error")
    plt.xlabel('x (micrometers)')
    plt.ylabel('Deviation (# ice layers)')
    plt.title("Difference between Expected and Predicted "+label)
    plt.grid(True)
    plt.legend()


def plot_alone(ylist, xlist, label = ""):
    plt.figure()
    plt.plot(xlist,ylist,label=label)
    plt.xlabel('x (micrometers)')
    plt.grid(True)
    plt.legend()