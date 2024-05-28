import numpy as np
from copy import copy as cp
import matplotlib.pylab as plt
from scipy.integrate import solve_ivp
from numba import njit, float64, int32, types
from matplotlib import rcParams
from time import time
from scipy.fft import fft, ifft, rfft, irfft, fftfreq
from scipy.interpolate import CubicSpline

def diffusion_term_FT_b_list_integrator(t, bj_list, j2_list, scalar_params):
    D, L = scalar_params

    Dcoefficient1 = D*np.pi**2/(L**2); #print('Dcoefficient1', Dcoefficient1)
    cj_list = bj_list*j2_list
    dy = -Dcoefficient1 * cj_list
    return dy
"""
sol = solve_ivp(\
            pypr_solve_ivp, tinterval, ylast, args=(scalar_params, sigmaI_mag, j_list, j2_list, x_QLC_mag), \
            rtol=1e-12, method=odemethod) 
        ylast = sol.y[:,-1]

def f1d_solve_ivp_FT(t, y, scalar_params, sigmaI, j2_list):

"""