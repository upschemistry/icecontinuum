## attempting to only use my diffusion library (no finding qll thickness iteritavely)
## all my comments begin with ##
## find some way to deal with Fliq?? define in terms of Ntot??

import cProfile
import os
import pstats
from ctypes import py_object
from math import floor
import numpy as np
from matplotlib import pyplot as plt
import time

import diffusion as df
# import diffusionstuff7 as ds
# import diffusionstuff6_1 as ds6

from copy import copy as dup
from scipy.integrate import solve_ivp
from scipy.integrate import odeint
from numba.types import int64,int32
import psutil

#for animations
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation, PillowWriter

#for saving simulations
import pickle

""" THIS IS NEW !!! ---This module implements a Simulation object that can be used to run simulations of the ice continuum;
    as well as functions to save, load, and continue a Simulation; and a function to test performance of functions.
"""

from sim_handling import Simulation as Sim


class SimulationNew(Sim):   
    ## NOTE only run() and getNtot() have been adjusted to work with diffusion

    def __init__(self, model=None, shape=(None,), method= "LSODA", atol= 1e-6, rtol= 1e-6, noisy=False, noise_stddev=0.01, layermax=0, nonstd_init=False, starting_ice=None, startingNtot=None, discretization_halt=True, mem_check=False, mem_threshold = 100E9, name="simulation"):
        """Initialize the Simulation
        Parameters
        ----------
        model (func): integratable differential equation model
        shape (tuple): shape of initial condition

        method (str): integration method
        atol (float): absolute tolerance
        rtol (float): relative tolerance
        noisy (bool): whether to add noise to the initial condition
        noise_stddev (float): standard deviation of noise to add to initial condition
        layermax (int): maximum number of layers in the ice
        """
        
        # model integrator arguments
        self.model = model #integratable differential equation model 
        self.method = method #default to emulate odeint for solve_ivp
        self.atol = atol #default absolute tolerance 
        self.rtol = rtol #default relative tolerance
        self.shape = shape #shape of initial condition

        # run-time arguments
        self.discretization_halt = discretization_halt #whether to halt the simulation when the discretization limit is reached
        self.mem_check = mem_check #whether to write to files to manage memory usage
        self.memory_threshold = mem_threshold #default virutal memory remaining before halting simulation and saving to file before continuing
        self.filename = name + '_save.npy'
        
        # make initial condition an attribute so it can be accessed for initialization in run()
        self.nonstd_init = nonstd_init
        self.starting_ice = starting_ice
        self.startingNtot = startingNtot

        #calculate dimension of initial condition
        if len(shape) == 1 and shape[0] == 1:
            self.dimension = 0
        else:
            self.dimension = len(shape)

        ### Experimental arguments ###
        Nbar = 1.0 #new Nbar from VMD, 260K
        Nstar = .9/(2*np.pi) #~=0.143239
        D = 1.6e-4 #micrometers^2/microsecond # Diffusion coefficient #NOTE: T=260K ??
        nmpermonolayer = 0.3 #thickness of a monolayer of ice #TODO: From MD, 2016 paper?  Sazaki et al. said 0.34 nm per monolayer
        umpersec_over_mlyperus = (nmpermonolayer/1e3*1e6) #conversion of nanometers per monolayer to micron/sec over monolayers/microsecond
        nu_kin = 250 #microns/second #NOTE: T=260K ?? 
        nu_kin_mlyperus = nu_kin/umpersec_over_mlyperus #bilayers? per microsecond # Deposition rate
        
        # Supersaturation
        self.sigma0 = 0.19
        self.sigmaIcorner = 0.25 #-0.10 # Must be bigger than sigma0 to get growth, less than 0 for ablatioq

        ### These are run control parameters ###
        self.noisy_init = noisy
        self.noise_std_dev = noise_stddev
        self.updatingNqll = True #flag for explicit updating Fliq(Ntot) every step 
        # Set up a maximum number of iterations or layers
        self.uselayers = True
        if self.uselayers:
            if layermax == 0:
                #use default layermaxes
                if self.dimension == 0:
                    self.layermax = 4
                elif self.dimension == 1:
                    self.layermax = 15000
                elif self.dimension == 2:
                    self.layermax = 20
            else:
                self.layermax = layermax #use user-defined layermax

        #default countermaxes prevent infinite loops
        if self.dimension == 0:
            self.countermax = 100
        elif self.dimension == 1:
            self.countermax = 10000
        elif self.dimension == 2:
            self.countermax = 10000
        
        #Dimension dependent parameters and time 
        t0 = 0.0 #start at time = 0
        if self.dimension == 0:
            self.deltaT = 1.0040120320801924 #NOTE/TODO: in continuum_model it was using the 1d deltaT for the 0d simulation
         
        discretization = 0.15 #microns per point 
        if self.dimension > 0:
            nx = self.shape[0] #number of points in simulation box            
            # xmax = discretization * nx #consistent discretization of 10 points per micron
            # self.x = np.arange(0, xmax, discretization)

            xmax = 150 # range of x
            self.x = np.linspace(0, xmax, nx)

            deltaX = self.x[1]-self.x[0]
            DoverdeltaX2 = D/deltaX**2 #diffusion coefficient scaled for this time-step and space-step

            # Center_reduction unused by 0d model
            self.center_reduction = 0.25 #in percent #last exp. parameter
            c_r = self.center_reduction/100

            # Time steps
            dtmaxtimefactor = 50 #TODO: what is this?
            dtmax = deltaX**2/D 
            self.deltaT = dtmax/dtmaxtimefactor #factored out of diffusion equation... 
            tmax = self.countermax*self.deltaT #ending time of simulation, used for solve_ivp

        if self.dimension == 2:
            ny = self.shape[1] 
            ymax = discretization * ny
            self.y = np.arange(0, ymax, discretization)

            deltaY = self.y[1]-self.y[0]
            DoverdeltaY2 = D/deltaY**2 #unused           
        # self.tinterval = [t0, tmax] #this is for solve_ivp
        self.tinterval = [t0, self.deltaT] #this is for odeint/ step by step solve ivp integration

        #Save variables not used in model via self.* to an array for saving
        self._extra_vars = {
            "Nbar":Nbar,
            "Nstar":Nstar,
            "D":D,
            "nu_kin":nu_kin,
            "nu_kin_mlyperus":nu_kin_mlyperus,
            "sigma0":self.sigma0,
            "sigmaIcorner":self.sigmaIcorner,
            "t0":t0
        }
        if self.dimension >= 1:
            self._extra_vars["dtmax"] = dtmax
            self._extra_vars["dtmaxtimefactor"] = dtmaxtimefactor
            self._extra_vars["DoverdeltaX2"]= DoverdeltaX2
            self._extra_vars["nx"]= nx
            self._extra_vars["xmax"]= xmax
            self._extra_vars["deltaX"]= deltaX
            self._extra_vars["c_r"]= c_r
        if self.dimension == 2:
            self._extra_vars["DoverdeltaY2"]= DoverdeltaY2
            self._extra_vars["ny"]= ny
            self._extra_vars["ymax"]= ymax
            self._extra_vars["deltaY"]= deltaY           
        self._extra_vars_types = {key:type(value) for key,value in self._extra_vars.items()}
        
        # Initialize the results dictionary
        self._results = {None:None}
        #Intitialize other internal attributes
        self._plot = None
        self._animation = None
        self._rerun = False

        # Initializing model arguments
        if self.dimension == 0:
            self.float_params = {'Nbar':Nbar, 'Nstar':Nstar, 'sigmaIcorner':self.sigmaIcorner, 'sigma0':self.sigma0, 'nu_kin_mlyperus':nu_kin_mlyperus}
        elif self.dimension == 1:
            self.float_params = {'Nbar':Nbar, 'Nstar':Nstar, 'sigma0':self.sigma0, 'nu_kin_mlyperus':nu_kin_mlyperus, 'DoverdeltaX2':DoverdeltaX2}
            self.int_params = {'nx':nx}
        elif self.dimension == 2:
            self.float_params = {'Nbar':Nbar,'Nstar':Nstar, 'sigma0':self.sigma0, 'nu_kin_mlyperus':nu_kin_mlyperus, 'DoverdeltaX2':DoverdeltaX2}
            self.int_params = {'nx':nx,'ny':ny}
        else:
            raise ValueError("Dimension must be 0, 1, or 2")

        #internal attributes
        self._plot = None #matplotlib figure
        self._animation = None #matplotlib animation
        self._results = {None:None} #solve_ivp (-like) dictionary of results
        pass



    def run(self, print_progress=True, print_count_layers=False, halve_time_res=False) -> None:
        """ Runs the simulation and saves the results to the Simulation object. (self.results() to get the results)
        NOTE as of f1d update 6/8/2023, no need for int_params or niter

        Args:
        -----
        print_progress: bool, optional
            Whether to print progress to the console. Default is True.
            
        print_count_layers: bool, optional
            Whether to print the number of layers to the console. Default is False.
        
        halve_time_res: bool, optional
            Whether to halve the time resolution. Default is False.
        """

        if self._results != {None:None}:
            # Already been run: tell plot/animation to update when called again since it is has been run again
            self._rerun = True

        # Discretization halt warning only triggers once
        warningIssued = False

        if halve_time_res:
            # Logic to only save every other time step
            flop = False # Starts as false to not save the second time step, since the first is always saved
        
        # Unpack parameters
        if self.dimension >= 0: # All dimensions have these params
            Nbar = self.float_params['Nbar']
            Nstar = self.float_params['Nstar']
            sigma0 = self.float_params['sigma0']
            nu_kin_mlyperus = self.float_params['nu_kin_mlyperus']

        if self.dimension == 0:     # Dimension specific params
            sigmaIcorner = self.float_params['sigmaIcorner']
            packed_float_params = np.array([Nbar, Nstar, sigmaIcorner, sigma0, nu_kin_mlyperus]) # in the order f1d expects
            model_args = (packed_float_params)
        if self.dimension == 1:
            sigmaI = df.getsigmaI(self.x,np.max(self.x),self.center_reduction,self.sigmaIcorner, method='parabolic')##df.getsigmastep(self.x, np.max(self.x), self.center_reduction, self.sigmaIcorner)
        if self.dimension == 2:
            sigmaI = df.getsigmaI(self.x,np.max(self.x),self.center_reduction,self.sigmaIcorner, method='parabolic')##df.getsigmastep_2d(self.x,self.y, self.center_reduction, self.sigmaIcorner) # supersaturation
            # nx = self.int_params['nx']
            # ny = self.int_params['ny']
            # packed_int_params = np.array(list(map(int64,[nx,ny])))
        if self.dimension >= 1:     # Similar packaging of params for dim 1 and 2
            DoverdeltaX2 = self.float_params['DoverdeltaX2']
            packed_float_params = np.array([Nbar, Nstar, sigma0, nu_kin_mlyperus, DoverdeltaX2]) # in the order f1d expects
            model_args = (packed_float_params, sigmaI)

        if self.nonstd_init:
            # Nonstandard initial conditions
            Nice = self.starting_ice    
            Ntot = self.startingNtot    
            Nqll = Ntot - Nice          
            if self.noisy_init:
                # Initialize with noise
                noise = np.random.normal(0,self.noise_std_dev,self.shape)
                Nice += noise
        else:
            # Lay out the initial system
            if self.dimension == 0:
                Nice = 1
                Nqll = Nbar # Starts as nbar
            else:
                # Dimensions 1 and 2 initializations similar, only difference is in sigma calls
                Nice = np.ones(self.shape)
                Nqll = Nbar - Nstar*np.sin(2*np.pi*(Nice)) # calc dimensionalized Nqll
            if self.noisy_init:
                # Initialize with noise
                noise = np.random.normal(0,self.noise_std_dev,self.shape)
                Nice += noise
            Ntot = Nqll + Nice

        # Only storing Ntot, y0 is now a 1d array
        if self.dimension == 0:
            y0 = np.array(0)
        else:
            y0 = np.array(Ntot)
        ylast = dup(y0)
        tlast = dup(self.tinterval)


        # print('sigma0',sigma0)
        # print('Nbar',Nbar)
        # print('Nstar',Nstar)
        # print('nu_kin_mlyperus',nu_kin_mlyperus)
        # print('DoverdeltaX2',DoverdeltaX2)
        # print('center reduction',self.center_reduction)
        # print('detlat',self.deltaT)


        # Initial conditions for ODE solver goes into keeper dictionary
        self._results['y'] = [y0]
        self._results['t'] = [self.tinterval]

        # Intialize ntot layer calculations
        if self.dimension == 0:
            Ntot0_start = Ntot
            Ntot0 = Ntot
        elif self.dimension == 1:
            Ntot0_start = Ntot[0]
            Ntot0 = Ntot[0]
        elif self.dimension == 2:
            Ntot0_start = Ntot[0,0]
            Ntot0 = Ntot[0,0]
     
        counter = 0
        lastlayer = 0
        lastdiff = 0

        if self.mem_check:
            memcheckcounter = 0
        

        ############
        Ntot_init_1D = np.ones(self.shape)
        # NQLL_init_1D = ds.getNQLL(Ntot_init_1D,Nstar,Nbar)

        # Initialize the keeper arrays
        tkeep_1D = []
        ykeep_1D = []
        tlast = self.tinterval[0]

        # Call the ODE solver
        ylast = Ntot_init_1D
        counter = 0
        layer = 0
        ttot = 0.0
        while True:
            # Integrate up to next time step
            y = odeint(self.model, ylast, self.tinterval, args=(model_args),rtol=1e-12,tfirst=True)
            ylast = list(y[1,:])
            tlast += self.deltaT
            
            # Stuff into keeper arrays
            ykeep_1D.append(ylast)
            tkeep_1D.append(tlast)

            # Make some local copies, with possible updates to NQLL
            ttot += self.deltaT

            # Update counters and see whether to break
            Ntotlast = ylast
            minpoint = min(Ntotlast)
            maxpoint = max(Ntotlast)
            if np.mod(counter,100) == 0:
                print(counter-1, int(Ntotlast[0]), maxpoint-minpoint)
            counter += 1
            
            if Ntotlast[0] > self.layermax-1:
                self._results['y']=ykeep_1D                    
                self._results['t']=tkeep_1D
                break
            if counter > self.countermax-1:
                self._results['y']=ykeep_1D
                self._results['t']=tkeep_1D
                break

        # Call the ODE solver
        # while True:
        #     # Integrate up to next time step 
            
        #     # Check the memory usage
        #     if self.mem_check:
        #         memcheckcounter += 1
        #         if memcheckcounter % 4 == 0: # Only check every 4 steps to save time checking memory
        #             memory_available = psutil.swap_memory().free
        #             if memory_available <= self.memory_threshold:
        #                 # Write or append to file
        #                 ## TODO does this still work??
        #                 super.woa_to_file(self, self.filename)
        #                 print('Memory usage exceeded threshold. Saving to file and halting.')
        #                 return self.filename

            

            # Locally copy previous thicknesses
            # Ntot = ylast
            # if self.updatingNqll:
            #     Nqll = Nbar - Nstar*np.sin(2*np.pi*(Ntot))
            # Nice = Ntot - Nqll

            # Solve
            # if self.method == 'odeint':
            #     solve_ivp_result = solve_ivp(self.model, self.tinterval, np.reshape(ylast,np.prod(np.shape(ylast))), method='RK45', args=model_args, rtol=self.rtol, atol=self.atol)#, t_eval=self.tinterval)
            # else:
            #     y = odeint(self.model, ylast, self.tinterval, args=model_args,rtol=self.rtol,atol=self.atol,tfirst=True)
            #     # solve_ivp_result = solve_ivp(self.model, self.tinterval, np.reshape(ylast,np.prod(np.shape(ylast))), method=self.method, args=model_args, rtol=self.rtol, atol=self.atol)
            # # y = solve_ivp_result.y[:, len(solve_ivp_result.t)-1]

            # # Update the state   
            # # ylast = y
            # ylast = list(y[1,:])
            # tlast += self.deltaT
            # counter += 1

            # # For calculating layers
            # if self.dimension == 0:
            #     Ntot0 = Ntot
            # elif self.dimension == 1:
            #     Ntot0 = Ntot[0]
            # elif self.dimension == 2:
            #     Ntot0 = Ntot[0,0]
            
            # if halve_time_res:
            # # Logic to only save every other time step (including first and last)
            #     if flop:
            #         self._results['y'].append(ylast)
            #         self._results['t'].append(tlast)
            #         flop = False
            #     else:
            #         flop = True # flop is a boolean that flips between true and false
            # else:    
            #     # Stuff into keeper arrays
            #     self._results['y'].append(ylast)
            #     self._results['t'].append(tlast)

            # # Update counters and see whether to break
            # layer = Ntot0-Ntot0_start
            # if (layer-lastlayer) > 0:
            #     minpoint = np.min(Ntot)
            #     maxpoint = np.max(Ntot)
            #     if print_count_layers:
            #         if counter == 1:
            #             # Print what each thing is
            #             print('counter, layer, depth_of_facet_in_layers, delta_depth')
            #         print(counter-1, lastlayer, maxpoint-minpoint, maxpoint-minpoint-lastdiff)
            #     lastdiff = maxpoint-minpoint

            #     # Break if too many steps for discretization
            #     if lastdiff > max(self.shape)//10:
            #         if not warningIssued:
            #             print('Warning: too many steps for discretization after', minpoint, 'layers grown')
            #             warningIssued = True
            #         if self.discretization_halt:
            #             print('Halting due to lack of discretization')
            #             break
            #     lastlayer += 1
                
            # # Test whether we're finished
            # if self.uselayers:
            #     if print_progress:
            #         if self.sigmaIcorner <0:
            #             prog = round(-1* layer/(self.layermax-1)*100, 2)
            #         else:
            #             prog = round(layer/(self.layermax-1)*100, 2)
            #         # Print progress
            #         print("appx progress:" , prog,"%",end="\r")
            #     if self.sigmaIcorner > 0:
            #         if layer > self.layermax-1:
            #             print('breaking because reached max number of layers grown')
            #             break
            #     else:
            #         if layer < -self.layermax:
            #             print('breaking because reached max number of layers ablated')
            #             break
            # else:
            #     if print_progress:
            #         prog = round(counter/(self.countermax)*100, 2)
            #         print("appx progress:" , prog,"%",end="\r")
            #     if counter > self.countermax-1:
            #         print('breaking because reached max number of iterations')
            #         break
        pass

    def results(self) -> dict:
        """ Returns results of simulation (handles running if necessary) """
        if self._results == {None:None}:
            self.run()
        return self._results

    def getNtot(self, step=None) -> np.ndarray:
            """ Returns the array of total ice and QLL thickness at each time step. """

            if step is None:
                return self.results()['y']
            else:
                return self.results()['y'][step]
            

    ###########################################
    ################# TODO ####################
    ###########################################

    ########## FUNCTIONS BELOW SHOULD WORK THE SAME TODO CHECK THAT THEY DO
            ### plot()
            ### animate()
            ### save()
            ### load()
            ### save_plot()
            ### save_animation()
            ### copy_sim()
            ### loadSim()
            ### get_expected_nss_steps()
            ### woa_to_file()
            ### continue_from_file()
            ### continue_from_surface()
            ### multiple_test_avg_time()

    ## TODO CAN I RENAME THIS TO getNqll()?
    def getFliq(self, step=None) -> np.ndarray:
        """ Returns the array of liquid thickness at each time step. """

        # Get Ntot from results
        Ntot = self.results()
        # Unpack needed params
        Nbar = self.float_params['Nbar']
        Nstar = self.float_params['Nstar']
        # Calc Nqll
        Nqll = Nbar - Nstar * np.sin(2*np.pi*Ntot)
        if step is None:
            return np.asarray(Nqll['y'])
        else:
            return Nqll['y'][step]   
        ##Original:: 
#         if step is None:
#             return np.asarray(self.results()['y'])[:, 0]
#         else:
#             return self.results()['y'][step][0]


    def getNice(self, step=None) -> np.ndarray:
        """ Returns the array of ice thickness at each time step. """

        # Get Ntot and Nqll
        if step is None:
            num_steps = len(self.results()['t'])
            Nice = np.empty((num_steps,*self.shape),dtype=object)
            for i in range(num_steps):
                Ntot = self.getNtot(i)
                Nqll = self.getFliq(i)
                Nice[i] = np.subtract(Ntot,Nqll, out=Ntot.copy())
            return Nice
        else:
            Ntot = self.getNtot(step)
            Nqll = self.getFliq(step)
            return np.subtract(Ntot,Nqll, out=Ntot.copy())
        ##Original::
#         if step is None:
#             num_steps = len(self.results()['t'])
#             Nice = np.empty((num_steps, *self.shape), dtype=object)
#             for i in range(num_steps):
#                 Nice[i] = np.subtract(self.getNtot(i), self.getFliq(i), out=self.getNtot(i).copy())
#             return Nice
#         else:
#             return np.subtract(self.getNtot(step), self.getFliq(step), out=self.getNtot(step).copy())

    def steepness(self, step:int, slice:slice):
        """ Returns normalized derivative indicating steps of the ice from at given step in a given range/slice"""

        # Get Ntot and Nqll to calc Nice
        num_steps = len(self.results()['t'])
        Nice = np.empty((num_steps,*self.shape),dtype=object)
        for i in range(num_steps):
            Ntot = self.getNtot(i)
            Nqll = self.getFliq(i)
        Nice = Ntot - Nqll
        # Calc difference in thickness from point to point
        step_density= [Nice[step][slice][i+1]-Nice[step][slice][i] for i in range(slice.stop-1)] 
        # Normalize step density to extreme value in slice
        step_density = [step/np.max(list(map(np.abs ,step))) for step in step_density]
        return step_density
        ##Original::
#         # unpack results
#         Fliq, Ntot = [],[]
#         for step in range(len(self.results()['t'])):
#             next_Fliq, next_Ntot = self._results['y'][step]
#             Fliq.append(next_Fliq)
#             Ntot.append(next_Ntot)    
#         Fliq,Ntot = np.array(Fliq), np.array(Ntot)
#         Nice = Ntot - Fliq

#         step_density= [Nice[step][slice][i+1]-Nice[step][slice][i] for i in range(slice.stop-1)]#difference in thickness from point to point 
#         #step_density = np.mean([Nice[step][slice][i+1]-Nice[step][slice][i] for i in range(slice.stop-1)]) #
#         #normalize step density to extreme value in slice
#         step_density = [step/np.max(list(map(np.abs ,step))) for step in step_density]
#         return step_density

    def get_step_density(self, step:int, slice:slice):
        """ Returns points in slice at which the steepness is at an extreme value """

        # Get Ntot and Nqll to calc Nice
        num_steps = len(self.results()['t'])
        Nice = np.empty((num_steps,*self.shape),dtype=object)
        for i in range(num_steps):
            Ntot = self.getNtot(i)
            Nqll = self.getFliq(i)
        Nice = Ntot - Nqll

        ymid = np.shape(Nice)[1]//2
        # Calc difference in thickness from point to point
        step_density= [Nice[step][slice][i+1][ymid]-Nice[step][slice][i][ymid] for i in range(slice.stop-1)]
        # Calc difference in slope from point to point
        second_deriv = [step_density[i+1]-step_density[i] for i in range(len(step_density)-1)]
        # Find the minimum step to see how close to zero the the second derivative gets
        print(np.min(list(map(np.abs,second_deriv))))
        # Find indices of points where step density is 'zero'
        zeroes_of_step_density = [0]#[i for i in second_deriv if np.abs(i)< 1e-04] 
        for i in second_deriv:
            if np.abs(i)< 1e-04:
                zeroes_of_step_density.append(1)
            else:
                zeroes_of_step_density.append(0)
        zeroes_of_step_density.append(0)#NOTE: two extra zeros to normalize size of zeroes_of_step_density to slice
        return zeroes_of_step_density
        ##Original::
#         # unpack results
#         Fliq, Ntot = [],[]
#         for step in range(len(self.results()['t'])):
#             next_Fliq, next_Ntot = self._results['y'][step]
#             Fliq.append(next_Fliq)
#             Ntot.append(next_Ntot)    
#         Fliq,Ntot = np.array(Fliq), np.array(Ntot)
#         Nice = Ntot - Fliq

#         #print('np.shape(Nice)', np.shape(Nice))
#         #print('np.shape(Nice[step])' , np.shape(Nice[step]))
#         #print('np.shape(Nice[step][slice])',np.shape(Nice[step][slice]))

#         ymid = np.shape(Nice)[1]//2

#         step_density= [Nice[step][slice][i+1][ymid]-Nice[step][slice][i][ymid] for i in range(slice.stop-1)]#difference in thickness from point to point
#         #print('shape of step_density', np.shape(step_density))
#         second_deriv = [step_density[i+1]-step_density[i] for i in range(len(step_density)-1)]#difference in slope from point to point
#         #print('shape of second_deriv', np.shape(second_deriv))
#         print(np.min(list(map(np.abs,second_deriv))))#find the minimum step to see how close to zero the the second derivative gets

#         #print(second_deriv)
#         zeroes_of_step_density = [0]#[i for i in second_deriv if np.abs(i)< 1e-04] #indices of points where step density is 'zero'
#         for i in second_deriv:
#             if np.abs(i)< 1e-04:
#                 zeroes_of_step_density.append(1)
#             else:
#                 zeroes_of_step_density.append(0)
#         zeroes_of_step_density.append(0)#NOTE: two extra zeros to normalize size of zeroes_of_step_density to slice
#         return zeroes_of_step_density
    

#     def normalize_results_to_min(self):
#         Fliq, Ntot = [],[]
#         for step in range(len(self.results()['t'])):
#             next_Fliq, next_Ntot = self._results['y'][step]
#             #normalize results
#             next_Fliq = next_Fliq - np.min(next_Fliq)
#             next_Ntot = next_Ntot - np.min(next_Ntot)
#             Fliq.append(next_Fliq)
#             Ntot.append(next_Ntot)
#         return np.array(Fliq), np.array(Ntot)
        

#     def steady_state_calc(self):
#         # unpack results
#         Fliq, Ntot = self.normalize_results_to_min()
#         flast,ntotlast = 0,0
#         normalizedFliq,normalizedNtot=[],[]
#         for f,n in zip(Fliq,Ntot):
#             f,n = f-flast,n-ntotlast
#             normalizedFliq.append(f)
#             normalizedNtot.append(n)
#             flast,ntotlast = f,n
        
#         #normalizedFliq,normalizedNtot = np.array(Fliq), np.array(Ntot)
#         #normalizedNice = normalizedNtot - normalizedFliq
#         return np.array(normalizedFliq), np.array(normalizedNtot)
    

#     def percent_change_from_last_step(self):
#         # unpack results
#         Fliq, Ntot = self.normalize_results_to_min()
#         flast,ntotlast = Fliq[0],Ntot[0] #step 0
#         normalizedFliq,normalizedNtot=[1],[1] # 100% change at initialization
#         for f,n in zip(Fliq[1:],Ntot[1:]):
#             f,n = f/flast,n/ntotlast #percent change in decimal form
#             normalizedFliq.append(f)
#             normalizedNtot.append(n)
#             flast,ntotlast = f,n
        
#         #normalizedFliq,normalizedNtot = np.array(Fliq), np.array(Ntot)
#         #normalizedNice = normalizedNtot - normalizedFliq
#         return np.array(normalizedFliq), np.array(normalizedNtot)