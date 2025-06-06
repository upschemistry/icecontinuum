import cProfile
import os
import pstats
from ctypes import py_object
from math import floor
import numpy as np
from matplotlib import pyplot as plt
import time
import diffusionstuff7 as ds
from copy import copy as dup
from scipy.integrate import solve_ivp
from numba.types import int64,int32
import psutil

#for animations
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation, PillowWriter

#for saving simulations
import pickle

"""This module implements a Simulation object that can be used to run simulations of the ice continuum;
    as well as functions to save, load, and continue a Simulation; and a function to test performance of functions.
    
    @Author: Max Bloom
        contact: mbloom@pugetsound.edu @mbloom0 on GitHub
"""

class Simulation():
    """Simulation class for integratable differential equation models.
    (specifically made for f0d, f1d, f2d)
    Facilitates running the simulations, saving and loading, and plotting the results.
    
    Attributes:
    ----------
    model (func): integratable differential equation model
    shape (tuple): shape of initial condition
    method (str): integration method
    atol (float): absolute tolerance
    rtol (float): relative tolerance
    
    Internal attributes:
        _plot: matplotlib figure
        _animation: matplotlib animation
        _results: results of simulation
    
    Methods:
    ----------
    run(): runs simulation
    plot(): plots results of simulation
    animate(): animates results of simulation
    results(): returns results of simulation (handles running if necessary)
    save(): saves simulation object to file
    load(): loads simulation object from file
    
    save_plot(): saves plot of simulation object
    save_animation(): saves animation of simulation object
    steepness(): returns normalized derivative, goal of indicating where steps of the ice are at given step in a given range/slice
    steady_state(): removes 

    
    @author: Max Bloom 
    """

    #TODO implement starting from an arbtitrary initial condition ( to test low freq. spatial noise)
        #TODO implement starting from a saved run
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
        Nbar = 1.0 # new Nbar from VMD, 260K
        Nstar = .9/(2*np.pi) # ~=0.14137
        D = 0.02e-2 # micrometers^2/microsecond # Diffusion coefficient #TODO: temperature dependent?
        nmpermonolayer = 0.3 #Thickness of a monolayer of ice #TODO: From MD, 2016 paper?  Sazaki et al. said 0.34 nm per monolayer
        umpersec_over_mlyperus = (nmpermonolayer/1e3*1e6)# Conversion of nanometers per monolayer to micron/sec over monolayers/microsecond
        nu_kin = 49 # microns/second #TODO: citation? 
        deprate = nu_kin/umpersec_over_mlyperus # bilayers? per microsecond # Deposition rate
        # deprate_times_deltaT = deprate * self.deltaT #unused
        # Supersaturation
        self.sigma0 = 0.19
        self.sigmastepmax = 0.20 #-0.10 # Must be bigger than sigma0 to get growth, less than 0 for ablatioq

        ### These are run control parameters ###
        niter = 1 # we do not use iterative calculation
        self.noisy_init = noisy
        self.noise_std_dev = noise_stddev
        # Flag for explicit updating Fliq(Ntot) every step 
        self.updatingFliq = True
        # Set up a maximum number of iterations or layers
        self.uselayers = True
        if self.uselayers:
            if layermax == 0:
                #use default layermaxes
                if self.dimension == 0:
                    self.layermax = 4
                elif self.dimension == 1:
                    self.layermax = 500
                elif self.dimension == 2:
                    self.layermax = 20
            else:
                self.layermax = layermax # use user-defined layermax

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
        
         
        #discretization = 0.05 #microns per point 
        discretization = 0.2 #microns per point 


        if self.dimension > 0:
            nx = self.shape[0] # Number of points in simulation box
            #print(type(nx))
            
            xmax = discretization * nx # consistent discretization of 10 points per micron
            #print(type(xmax))
            self.x = np.arange(0, xmax, discretization)
            #xmax = 50 # range of x
            #self.x = np.linspace(0, xmax, nx)


            deltaX = self.x[1]-self.x[0]
            DoverdeltaX2 = D/deltaX**2 # Diffusion coefficient scaled for this time-step and space-step

            #center_reduction unused by 0d model
            self.center_reduction = 0.25 # In percent #last exp. parameter
            c_r = self.center_reduction/100

            # Time steps
            dtmaxtimefactor = 50 #TODO: what is this?
            dtmax = deltaX**2/D 
            self.deltaT = dtmax/dtmaxtimefactor #factored out of diffusion equation... 
            tmax = self.countermax*self.deltaT #ending time of simulation, used for solve_ivp

        if self.dimension == 2:
            #ny = nx
            ny = self.shape[1] 
            ymax = discretization * ny
            #ymax = round(xmax * ny/nx)
            self.y = np.arange(0, ymax, discretization)
            #self.y = np.linspace(0, ymax, ny)

            deltaY = self.y[1]-self.y[0]
            DoverdeltaY2 = D/deltaY**2 #unused           
        #self.tinterval = [t0, tmax] #this is for solve_ivp
        self.tinterval = [t0, self.deltaT] #this is for odeint/ step by step solve ivp integration

        #Save variables not used in model via self.* to an array for saving
        self._extra_vars = {
            "Nbar":Nbar,
            "Nstar":Nstar,
            "D":D,
            "nu_kin":nu_kin,
            "deprate":deprate,
            #"deprate_times_deltaT":deprate_times_deltaT,#unused
            "sigma0":self.sigma0,
            "sigmastepmax":self.sigmastepmax,
            "niter":niter,
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
            self.float_params = {'Nbar':Nbar, 'Nstar':Nstar, 'sigmastepmax':self.sigmastepmax, 'sigma0':self.sigma0, 'deprate':deprate}
            self.int_params = {'niter':niter}
        elif self.dimension == 1:
            self.float_params = {'Nbar':Nbar, 'Nstar':Nstar, 'sigma0':self.sigma0, 'deprate':deprate, 'DoverdeltaX2':DoverdeltaX2}
            self.int_params = {'niter':niter, 'nx':nx}
        elif self.dimension == 2:
            self.float_params = {'Nbar':Nbar,'Nstar':Nstar, 'sigma0':self.sigma0, 'deprate':deprate, 'DoverdeltaX2':DoverdeltaX2}
            self.int_params = {'niter':niter,'nx':nx,'ny':ny}
        else:
            raise ValueError("Dimension must be 0, 1, or 2")

        #internal attributes
        self._plot = None #matplotlib figure
        self._animation = None #matplotlib animation
        self._results = {None:None} #solve_ivp (-like) dictionary of results
        pass

    def run(self, print_progress=True, print_count_layers=False, halve_time_res=False) -> None:
        """Runs the simulation and saves the results to the Simulation object. (self.results() to get the results)
        
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
            #already been run: tell plot/animation to update when called again since it is has been run again
            self._rerun = True

        #discretization halt warning only triggers once
        warningIssued = False

        if halve_time_res:
            #logic to only save every other time step
            flop = False #starts as false to not save the second time step, since the first is always saved
        
        #unpack parameters
        if self.dimension >= 0:
            Nbar = self.float_params['Nbar']
            Nstar = self.float_params['Nstar']
            sigma0 = self.float_params['sigma0']
            deprate = self.float_params['deprate']

            niter = self.int_params['niter']
        if self.dimension == 0:
            sigmastepmax = self.float_params['sigmastepmax']
            packed_float_params = np.array([Nbar, Nstar, sigmastepmax, sigma0, deprate])#in the order f1d expects
        if self.dimension >= 1:
            DoverdeltaX2 = self.float_params['DoverdeltaX2']
            nx = self.int_params['nx']

            # Bundle parameters for ODE solver
            packed_float_params = np.array([Nbar, Nstar, sigma0, deprate, DoverdeltaX2])
            packed_int_params = np.array(list(map(int32,[niter,nx])))#f1d expects int32s
    
            #NOTE: niter removed as of diffusionstuff8
        if self.dimension == 2:
            #DoverdeltaY2 = self.float_params['DoverdeltaY2'] #unused
            ny = self.int_params['ny']
            
            #packed_float_params = np.array([Nbar, Nstar, sigma0, deprate, DoverdeltaX2])
            packed_int_params = np.array(list(map(int64,[niter,nx,ny]))) # sigmastep math in f2d in diffusionstuff7 requires int64
        
        if self.nonstd_init:
            #nonstandard initial conditions
            Nice = self.starting_ice
            Ntot = self.startingNtot
            Fliq = Ntot - Nice
            if self.noisy_init:
                # Initialize with noise
                noise = np.random.normal(0,self.noise_std_dev,self.shape)
                Nice += noise
            if self.dimension == 0:
                #sigma = sigmastepmax
                model_args = (packed_float_params,niter)
                nliq_func = ds.getNliq # function to get to update fliq
            elif self.dimension == 1:
                sigma = ds.getsigmastep(self.x, np.max(self.x), self.center_reduction, self.sigmastepmax)
                model_args = (packed_float_params,packed_int_params,sigma)
                nliq_func = ds.getNliq_array # function to get to update fliq
            elif self.dimension == 2:
                sigma = ds.getsigmastep_2d(self.x,self.y, self.center_reduction, self.sigmastepmax) # supersaturation
                model_args = (packed_float_params,packed_int_params,sigma) 
                nliq_func = ds.getNliq_2d_array # function to get to update fliq
        else:
            # Lay out the initial system
            if self.dimension == 0:
                Nice = 1
            else:
                Nice = np.ones(self.shape)
            if self.noisy_init:
                # Initialize with noise
                noise = np.random.normal(0,self.noise_std_dev,self.shape)
                Nice += noise
            if self.dimension == 0:
                Fliq = Nbar #ds.getNliq(Nice,Nstar,Nbar,niter) fliq updates to this but starts as nbar
                #sigma = sigmastepmax
                model_args = (packed_float_params,niter)
                nliq_func = ds.getNliq # function to get to update fliq
            elif self.dimension == 1:
                Fliq = ds.getNliq_array(Nice,Nstar,Nbar,niter)
                sigma = ds.getsigmastep(self.x, np.max(self.x), self.center_reduction, self.sigmastepmax)
                model_args = (packed_float_params,packed_int_params,sigma)
                nliq_func = ds.getNliq_array # function to get to update fliq
            elif self.dimension == 2:
                Fliq = ds.getNliq_2d_array(Nice,Nstar,Nbar,niter) # Initialize as a pre-equilibrated layer of liquid over ice
                sigma = ds.getsigmastep_2d(self.x,self.y, self.center_reduction, self.sigmastepmax) # supersaturation
                model_args = (packed_float_params,packed_int_params,sigma) 
                nliq_func = ds.getNliq_2d_array # function to get to update fliq
            Ntot = Fliq + Nice

        if self.dimension == 0:
            y0 = np.array([Fliq, 0.0])#starts at zero ice to get see more layers form at beginning
        else:
            y0 = np.array([Fliq,Ntot])
        ylast = dup(y0)
        tlast = dup(self.tinterval[0])

        # Initial conditions for ODE solver goes into keeper dictionary
        self._results['y'] = [y0]
        self._results['t'] = [self.tinterval[0]]

        # intialize ntot layer calculations
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
        # Call the ODE solver
        while True:
            # Integrate up to next time step 
            
            # Check the memory usage
            if self.mem_check:
                memcheckcounter += 1
                if memcheckcounter % 4 == 0:# only check every 4 steps to save time checking memory
                    memory_available = psutil.swap_memory().free
                    if memory_available <= self.memory_threshold:
                        #write or append to file
                        woa_to_file(self, self.filename)
                        print('Memory usage exceeded threshold. Saving to file and halting.')
                        return self.filename

            # Make some local copies, with possible updates to Fliq
            Fliq, Ntot = ylast
            if self.updatingFliq:
                Fliq = nliq_func(Ntot,Nstar,Nbar,int32(niter)) # This updates to remove any drift
                ylast[0] = Fliq
            Nice = Ntot - Fliq

            #print(self.model, self.tinterval, np.reshape(ylast,np.prod(np.shape(ylast))), self.method,model_args, self.rtol, self.atol)
            if self.method == 'odeint':
                solve_ivp_result = solve_ivp(self.model, self.tinterval, np.reshape(ylast,np.prod(np.shape(ylast))), method='RK45', args=model_args, rtol=self.rtol, atol=self.atol)#, t_eval=self.tinterval)
                y = solve_ivp_result.y[:, len(solve_ivp_result.t)-1]#y[:,-1] : get last timestep that solve_ivp returns
            else:
                #old version: solve_ivp_result = solve_ivp(self.model, self.tinterval, np.reshape(ylast,np.prod(np.shape(ylast))), method=self.method, args=model_args, t_eval=self.tinterval, rtol=self.rtol, atol=self.atol)
                solve_ivp_result = solve_ivp(self.model, self.tinterval, np.reshape(ylast,np.prod(np.shape(ylast))), method=self.method, args=model_args, rtol=self.rtol, atol=self.atol)
        
            y = solve_ivp_result.y[:, len(solve_ivp_result.t)-1]#y[:,-1] : get last timestep that solve_ivp returns
            
            # Update the state                 
            if self.dimension == 0:
                ylast = y
            elif self.dimension == 1:
                ylast = np.reshape(y,(2,self.shape[0])) #used to by ylast = y[1] for odeint
            elif self.dimension == 2:
                ylast = np.reshape(y,(2,self.shape[0],self.shape[1])) 
            tlast += self.deltaT
            counter += 1
            #### where fliq was updated normally####

            #for calculating layers
            if self.dimension == 0:
                Ntot0 = Ntot
            elif self.dimension == 1:
                Ntot0 = Ntot[0]
            elif self.dimension == 2:
                Ntot0 = Ntot[0,0]

            if halve_time_res:
            #logic to only save every other time step (including first and last)
                if flop:
                    self._results['y'].append(ylast)
                    self._results['t'].append(tlast)
                    flop = False
                else:
                    flop = True #flop is a boolean that flips between true and false
            else:    
                # Stuff into keeper arrays
                self._results['y'].append(ylast)
                self._results['t'].append(tlast)

            # Update counters and see whether to break
            layer = Ntot0-Ntot0_start
            if (layer-lastlayer) > 0:
                minpoint = np.min(Nice)
                maxpoint = np.max(Nice)
                if print_count_layers:
                    if counter == 1:
                        #print what each thing is
                        print('counter, layer, depth_of_facet_in_layers, delta_depth')
                    print(counter-1, lastlayer, maxpoint-minpoint, maxpoint-minpoint-lastdiff)
                lastdiff = maxpoint-minpoint

                #break if too many steps for discretization
                if lastdiff > max(self.shape)//10:
                    if not warningIssued:
                        print('Warning: too many steps for discretization after', minpoint, 'layers grown')
                        warningIssued = True
                    if self.discretization_halt:
                        print('Halting due to lack of discretization')
                        break
                
                lastlayer += 1
                
            # Test whether we're finished
            if self.uselayers:
                if print_progress:
                    if self.sigmastepmax <0:
                        prog = round(-1* layer/(self.layermax-1)*100, 2)
                    else:
                        prog = round(layer/(self.layermax-1)*100, 2)
                    print("appx progress:" , prog,"%",end="\r")
                if self.sigmastepmax > 0:
                    if layer > self.layermax-1:
                        print('breaking because reached max number of layers grown')
                        break
                else:
                    if layer < -self.layermax:
                        print('breaking because reached max number of layers ablated')
                        break
            else:
                if print_progress:
                    prog = round(counter/(self.countermax)*100, 2)
                    print("appx progress:" , prog,"%",end="\r")
                if counter > self.countermax-1:
                    print('breaking because reached max number of iterations')
                    break
        
        #solve_ivp does not support terminating after a given number of layers
        #self._results = solve_ivp(self.model, self.t_span, self.y0, t_eval=self.t_eval, method=self.method, atol=self.atol, rtol=self.rtol, args=self._args)
        pass
    
    def results(self) -> dict:
        """ Returns results of simulation (handles running if necessary) """
        if self._results == {None:None}:
            self.run()
        return self._results

    def getFliq(self, step=None) -> np.ndarray:
        """
        Returns the array of liquid thickness at each time step.
        """
        if step is None:
            return np.asarray(self.results()['y'])[:, 0]
        else:
            return self.results()['y'][step][0]
        
    def getNtot(self, step=None) -> np.ndarray:
        """
        Returns the array of total ice and QLL thickness at each time step.
        """
        if step is None:
            return np.asarray(self.results()['y'])[:, 1]
        else:
            return self.results()['y'][step][1]

    def getNice(self, step=None) -> np.ndarray:
        """
        Returns the array of ice thickness at each time step.
        """
        if step is None:
            num_steps = len(self.results()['t'])
            Nice = np.empty((num_steps, *self.shape), dtype=object)
            for i in range(num_steps):
                Nice[i] = np.subtract(self.getNtot(i), self.getFliq(i), out=self.getNtot(i).copy())
            return Nice
        else:
            return np.subtract(self.getNtot(step), self.getFliq(step), out=self.getNtot(step).copy())

    def plot(self, completion=1, figurename='', ice=True, tot=False, liq=False, surface=True, contour=False):# -> matplotlib_figure: ## , method = 'surface'): #TODO: test plotting
        """ Plot the results of the simulation.
        
        Args:
        ----------
        figurename: str
            Name of the figure to save.
        ice: bool
            Whether to plot the ice.
        tot: bool
            Whether to plot the liquid and the ice combined.
        liq: bool
            Whether to plot the liquid (without the ice).
        surface: bool
            Plot (the 2d model) as a surface?
        contour: bool
            Plot (the 2d model) as a contour diagram?

        
        """
        """ plot results of simulation, returns matplotlib figure """
        #if self._plot == None or self._rerun:
        #create plot of results
        step = floor((len(self.results()['t'])-1)*completion)
                    
        # Plot the results
        self._plot = plt.figure(figurename)
        if self.dimension == 0:
            # Plot results
            plt.xlabel(r't ($\mu s$)')
            plt.grid('on')
            if ice: 
                Nice = self.getNice()
                plt.plot(self.t, Nice)
                plt.ylabel(r'$N_{ice} $')
            if tot:
                Ntot = self.getNtot()
                plt.plot(self.t, Ntot)
                plt.ylabel(r'$N_{ice} + $N_{QLL} $')
            if liq:
                Fliq = self.getFliq()
                plt.plot(self.t, Fliq)
                plt.ylabel(r'$N_{QLL} $')
            plt.legend()
            plt.xlabel('Time')
            plt.ylabel('Layers of ice')
        elif self.dimension == 1:
            ax = plt.axis()
            if ice:
                plt.plot(self.x, self.getNice(step), 'k', label='ice')
            if tot:
                plt.plot(self.x, self.getNtot(step), 'b', label='ice+QLL')
                #plt.plot(x-xmid, Fliq+Nice-minpoint, 'b', label='ice+liquid', lw=linewidth)
            if liq:
                plt.plot(self.x, self.getFliq(step), 'g', label='QLL')
            plt.legend()
            plt.xlabel(r'x ($\mu m$)')
            plt.ylabel('Layers of ice')
        elif self.dimension == 2:
            #access coordinate arrays for plotting
            ys, xs = np.meshgrid(self.y, self.x)
            
            #print(xs.shape, ys.shape, Nice.shape)
            ax = plt.axes(projection='3d')
            if surface:
                if ice:
                    ax.plot_surface(X=xs, Y=ys, Z=self.getNice(step), cmap='viridis')#, vmin=0, vmax=200)
                if tot:
                    ax.plot_surface(X=xs, Y=ys, Z=self.getNtot(step), cmap='YlGnBu_r')#, vmin=0, vmax=200)
                if liq:
                    ax.plot_surface(X=xs, Y=ys, Z=self.getFliq(step), cmap='YlGnBu_r')
            elif contour: #elif method == 'contour':
                levels = np.arange(-6,12,0.25)
                if ice:
                    ax.contour(xs,ys, self.getNice(step), extent=(0, 2, 0, 2), cmap='YlGnBu_r', vmin=0, vmax=200, zorder=1, levels=levels)
                if tot:
                    ax.contour(xs,ys, self.getNtot(step), extent=(0, 2, 0, 2), cmap='YlGnBu_r', vmin=0, vmax=200, zorder=1, levels=levels)
                if liq:
                    ax.contour(xs,ys, self.getFliq(step), extent=(0, 2, 0, 2), cmap='YlGnBu_r', vmin=0, vmax=200, zorder=1, levels=levels)
            
            ax.set_xlabel(r'x ($\mu m$)')
            ax.set_ylabel(r'y ($\mu m$)')
            ax.set_zlabel('Layers of ice')
            # else:
            #     print('Error: dimension not supported')
            #     return None
        plt.show()
        #return self._plot 
        pass
    
    def animate(self, proportionalSpeed=True, ice=True, tot=False, liq=False, surface=True,
                 crossSection=False, ret=False, speed=1, focus_on_growth=True):
        """ Animate the results of the simulation.

        Args:
        ----------
        proportionalSpeed: bool
            Whether to use a proportional speed for the animation.
        
        ice: bool
            Whether to plot the ice.
        
        tot: bool
            Whether to plot the liquid and the ice combined.
        
        liq: bool   
            Whether to plot the liquid (without the ice).
        
        surface: bool   
            Plot (the 2d model) as a surface?
        
        crossSection: bool
            Plot a cross section (of the 2d model)?
        
        ret: bool
            Return the animation, or just pass.
        """
        if self._animation == None:
            #create animation of results
            num_steps = len(self.results()['t'])
            #shape of results is (num_steps, 2, nx, ny)
            Nice = self.getNice()
            if tot:
                Ntot = self.getNtot()
            
            #shape of fliq, ntot and nice should be (num_steps, nx, ny)
            fig = plt.figure()
            #global update_fig#neccesary for saving the animation- but we cant save the animation object anyway so unnecessary
            if self.dimension == 1:
                #2d animation of 1d model over time
                ax = plt.axes()
                ax.set_xlabel(r'x ($\mu m$)')
                ax.set_ylabel('Layers of ice')
                #plt.xlim(min(self.x-0.5), max(self.x+0.5))
                
                def update_fig(num):
                    ax.clear()
                    #ax.draw()
                    #ax.set_ylim(min(Nice[num]-0.5), max(Ntot[num]+0.5))
                    if ice:
                        #plt.plot(self.x, Nice[num], 'k', label='ice')
                        ax.plot(self.x, Nice[num], 'k', label='ice')
                    if tot:
                        #plt.plot(self.x, Ntot[num], 'b', label='ice+QLL')
                        ax.plot(self.x, Ntot[num], 'b', label='ice+QLL')
                    if liq:
                        #plt.plot(self.x, Fliq[num], 'g', label='QLL')
                        ax.plot(self.x, Ntot[num], 'b', label='ice+QLL')
                        #ax.set_ylim(min(Fliq[num]-0.5), max(Ntot[num]+0.5))
                    ax.set_xlim(min(self.x-0.5), max(self.x+0.5))
                    ax.set_ylim(np.min(Nice[num]-0.5), np.max(Nice[num]+1.5))
                    pass
                plt.legend()
            elif self.dimension == 2: 
                #access coordinate arrays for plotting
                #np.meshgrid returns y before x
                ys, xs = np.meshgrid(self.y, self.x)

                #3d animation of the results
                #print(xs.shape, ys.shape, Nice.shape)
                ax = plt.axes(projection='3d')

                #set up axis labels and limits
                ax.set_xlabel(r'$x (\mu m$)')#,fontsize=fontsize)
                ax.set_ylabel(r'$y (\mu m$)')#,fontsize=fontsize)
                ax.set_zlabel(r'$ice \ layers$')#,fontsize=fontsize)
                ax.set_xlim(0, max(self.x))
                ax.set_ylim(0, max(self.y))
                if not focus_on_growth:
                    ax.set_zlim3d(np.min(Nice)-0.5, np.max(Nice)+1.5) # show full range of ice layers grown (0 to layermax)

                #labels
                def update_fig(num):
                    ax.clear() # remove last iteration of plot 
                    #limits
                    if focus_on_growth: #stay zoomed on new growth
                        ax.set_zlim3d(np.min(Nice[num]-0.5), np.max(Nice[num]+1.5)) 
                    
                    #ax.set_aspect('equal') doesnt work for 3d, breaking animation for some reason
                    if surface:
                        if ice:
                            ax.plot_surface(X=xs, Y=ys, Z=Nice[num], cmap='viridis')#, vmin=0, vmax=200)#plot the surface of the ice 
                        if tot:
                            ax.plot_surface(X=xs, Y=ys, Z=Ntot[num], cmap='YlGnBu_r')#, vmin=0, vmax=200)#plot the surface of the QLL
                        #cross section
                        # xmid = round(np.shape(Nice)[0]/2)
                        # if ice:
                        #     print('xs:',xs,'ys:',ys)
                        #     ax.plot_surface(X=xs[xmid:], Y=ys[xmid:], Z=Nice[num][xmid:][:],cmap ='viridis')# cmap='viridis')#, vmin=0, vmax=200) #plot half of the surface of the ice

                        # if tot:
                        #     ax.plot_surface(X=xs[xmid:], Y=ys[xmid:], Z=Ntot[num][xmid:][:], cmap='cividis')#, vmin=0, vmax=200) #plot half the surface of the QLL
                    else:
                        if ice:
                            ax.wireframe(X=xs, Y=ys, Z=Nice[num], cmap='viridis')#, vmin=0, vmax=200) #plot the surface of the ice 
                        if tot:
                            ax.wireframe(X=xs, Y=ys, Z=Ntot[num], cmap='YlGnBu_r')#, vmin=0, vmax=200)#plot the surface of the QLL
                    pass
            
            #create animation
            if proportionalSpeed:#TODO: scale interval to make length of gif/mp4 be 10 seconds, scaling speed of animation by factor proportional to length of simulation
                intrvl = int(50*30/self.layermax)+1#targeting speeds similar to 50ms interval at 30 layers, if more layers it will speed it up to keep the animation at the same visual speed
            else:
                intrvl = 5 #5ms interval

            if speed != 1: # increase animation replay speed by "speed" factor
                intrvl = intrvl // speed
            

            #self._animation = animation.FuncAnimation(self._anim_fig, update_fig, num_steps, interval=intrvl, blit=False, cache_frame_data=False, repeat = True)
            anim = FuncAnimation(fig, update_fig, num_steps, interval=intrvl, blit=False, cache_frame_data=False, repeat = True) #pickle does not like saving animation, also it is a a lot of data to save
        plt.show()
        if ret:
            return anim#self._animation
        else:
            pass

    def save(self, id = []) -> None:
        """ Saves Simulation object to a pickle file
        
        args:
            id: list of strings to append to filename: what makes this simulation unique
        """
        # Saving these results to file
        #if Nice[0] > 100000: #what is this nesh?
        #    Nice -= 100000
        id = ''.join('_'+i for i in id)
        filename = self.model.__name__+'_simulation'+id+'.pkl'
        with open(filename, 'wb') as f:
            print("saving to", f)
            pickle.dump(self, f)
        pass

    def load(self, filename: str) -> py_object:
        """ Loads and initializes Simulation object from pickle file  """
        with open(filename, 'rb') as f:
            self = pickle.load(f)
        return self
    
    def save_plot(self, filename) -> None:
        """ saves plot of simulation object """         
        #Save the results as an image
        filename = str(self.dimension) + 'd_model_results_'+str(self.layermax)+'_layers'
        self.plot().savefig(filename+'.png', dpi=300)
        pass

    def save_animation(self, filename:str, filetype:str) -> None:
        """ saves animation of simulation object """
        try:
            if filetype == 'mp4':
                Writer = animation.writers['ffmpeg']
                writer = Writer(fps=15, bitrate=1800)
            elif filetype == 'gif':
                writer = PillowWriter(fps=60,bitrate=1800)#animation.writers['imagemagick']
            else:
                print('filetype not supported')
                return
        except Exception as e:
            print(e)
            print('Error creating animation file writer')
            return

        try:
            self.animate(ret=True).save(filename+'.'+filetype, writer=writer)
        except Exception as e:
            print(e)
            print('Error saving animation')
            return
        pass

    def steepness(self, step:int, slice:slice):
        """ Returns normalized derivative indicating steps of the ice from at given step in a given range/slice"""
        # unpack results
        Fliq, Ntot = [],[]
        for step in range(len(self.results()['t'])):
            next_Fliq, next_Ntot = self._results['y'][step]
            Fliq.append(next_Fliq)
            Ntot.append(next_Ntot)    
        Fliq,Ntot = np.array(Fliq), np.array(Ntot)
        Nice = Ntot - Fliq

        step_density= [Nice[step][slice][i+1]-Nice[step][slice][i] for i in range(slice.stop-1)]#difference in thickness from point to point 
        #step_density = np.mean([Nice[step][slice][i+1]-Nice[step][slice][i] for i in range(slice.stop-1)]) #
        #normalize step density to extreme value in slice
        step_density = [step/np.max(list(map(np.abs ,step))) for step in step_density]
        return step_density

    def get_step_density(self, step:int, slice:slice):
        """ Returns points in slice at which teh steepness is at an extreme value """
        # unpack results
        Fliq, Ntot = [],[]
        for step in range(len(self.results()['t'])):
            next_Fliq, next_Ntot = self._results['y'][step]
            Fliq.append(next_Fliq)
            Ntot.append(next_Ntot)    
        Fliq,Ntot = np.array(Fliq), np.array(Ntot)
        Nice = Ntot - Fliq

        #print('np.shape(Nice)', np.shape(Nice))
        #print('np.shape(Nice[step])' , np.shape(Nice[step]))
        #print('np.shape(Nice[step][slice])',np.shape(Nice[step][slice]))

        ymid = np.shape(Nice)[1]//2

        step_density= [Nice[step][slice][i+1][ymid]-Nice[step][slice][i][ymid] for i in range(slice.stop-1)]#difference in thickness from point to point
        #print('shape of step_density', np.shape(step_density))
        second_deriv = [step_density[i+1]-step_density[i] for i in range(len(step_density)-1)]#difference in slope from point to point
        #print('shape of second_deriv', np.shape(second_deriv))
        print(np.min(list(map(np.abs,second_deriv))))#find the minimum step to see how close to zero the the second derivative gets

        #print(second_deriv)
        zeroes_of_step_density = [0]#[i for i in second_deriv if np.abs(i)< 1e-04] #indices of points where step density is 'zero'
        for i in second_deriv:
            if np.abs(i)< 1e-04:
                zeroes_of_step_density.append(1)
            else:
                zeroes_of_step_density.append(0)
        zeroes_of_step_density.append(0)#NOTE: two extra zeros to normalize size of zeroes_of_step_density to slice
        return zeroes_of_step_density
    
    def normalize_results_to_min(self):
        Fliq, Ntot = [],[]
        for step in range(len(self.results()['t'])):
            next_Fliq, next_Ntot = self._results['y'][step]
            #normalize results
            next_Fliq = next_Fliq - np.min(next_Fliq)
            next_Ntot = next_Ntot - np.min(next_Ntot)
            Fliq.append(next_Fliq)
            Ntot.append(next_Ntot)
        return np.array(Fliq), np.array(Ntot)
        
    def steady_state_calc(self):
        # unpack results
        Fliq, Ntot = self.normalize_results_to_min()
        flast,ntotlast = 0,0
        normalizedFliq,normalizedNtot=[],[]
        for f,n in zip(Fliq,Ntot):
            f,n = f-flast,n-ntotlast
            normalizedFliq.append(f)
            normalizedNtot.append(n)
            flast,ntotlast = f,n
        
        #normalizedFliq,normalizedNtot = np.array(Fliq), np.array(Ntot)
        #normalizedNice = normalizedNtot - normalizedFliq
        return np.array(normalizedFliq), np.array(normalizedNtot)
    
    def percent_change_from_last_step(self):
        # unpack results
        Fliq, Ntot = self.normalize_results_to_min()
        flast,ntotlast = Fliq[0],Ntot[0] #step 0
        normalizedFliq,normalizedNtot=[1],[1] # 100% change at initialization
        for f,n in zip(Fliq[1:],Ntot[1:]):
            f,n = f/flast,n/ntotlast #percent change in decimal form
            normalizedFliq.append(f)
            normalizedNtot.append(n)
            flast,ntotlast = f,n
        
        #normalizedFliq,normalizedNtot = np.array(Fliq), np.array(Ntot)
        #normalizedNice = normalizedNtot - normalizedFliq
        return np.array(normalizedFliq), np.array(normalizedNtot)

    def get_expected_nss_steps(self):
        #Calculating number of expected steps to reach steady state
        L = np.max(self.x)/2; print(L) # micrometers
        c_r = self.center_reduction / 100; print(c_r) # dimensionless
        nu_kin_ml = self._extra_vars['deprate']; print(nu_kin_ml) # monolayers per microsecond
        sigma_I = self.sigmastepmax; print(sigma_I) # dimensionless
        #print(self._extra_vars['D']) # D is in micrometers^2/microsecond
        M = np.array([.0027, .0025])
        B = np.array([2.9, 1.59])
        beta = np.array([0.65, 0.65])
        xfactor = nu_kin_ml*L**2*c_r**beta*sigma_I/self._extra_vars['D']
        NSS = M*xfactor + B
        print('Nss predicted')
        print('sinusoid:', NSS[0])
        print('parabolic:', NSS[1])
        return NSS


def loadSim(filename: str) -> py_object:
        """ Loads and initializes Simulation object from pickle file  """
        with open(filename, 'rb') as f:
            self = pickle.load(f)
        return self

"""
Performance testing functions for the ice model.
"""
def multiple_test_avg_time(func, args=None, n_tests = 50):
    """
    Test a function n_tests times and return the average time taken for the function,
    as well as the profile statistics.
    """
    times = []
    for i in range(n_tests):
        # Create a Profile object
        pr = cProfile.Profile()

        # Run the function and create the profile data
        start = np.float64(time.time())
        if args == None:
            pr.runcall(func)
        else:
            pr.runcall(func, *args)
        times.append(time.time()-start)

        # Create a Stats object
        stats = pstats.Stats(pr)

    avg_time_taken = float(np.mean(times))
    print("Time to run "+str(func.__name__)+" on average for "+ str(n_tests) +" tests: ", avg_time_taken, "seconds")
    return stats, avg_time_taken


# idea for a function but ?????
# def runSimulations(params_array) -> dict:
#     """
#     Run the simulations and return the solve_ivp results dictionary.
#     """
#     results_array = []
#     for params in params_array:
        
#         sim = Simulation(model, shape, method=method, atol=atol, rtol=rtol)
#         sim.run()
#         results_array[params] = sim.results()
#     return results_array

#functions to handle simulations too large to fit in memory

def copy_sim(simulation: Simulation):
    #copy simulation params
    new_sim = Simulation(simulation.model, simulation.shape, method=simulation.method, rtol=simulation.rtol)
    new_sim.layermax = simulation.layermax
    new_sim.float_params['DoverdeltaX2'] = simulation.float_params['DoverdeltaX2']
    new_sim.sigma0 = simulation.sigma0
    new_sim.sigmastepmax = simulation.sigmastepmax
    new_sim.center_reduction = simulation.center_reduction
    new_sim.noisy_init = simulation.noisy_init
    
    new_sim.nonstd_init = True
    new_sim.uselayers = simulation.uselayers
    new_sim.countermax = simulation.countermax
    new_sim.discretization_halt = simulation.discretization_halt
    new_sim.mem_check = simulation.mem_check
    new_sim.filename = simulation.filename
    
    return new_sim

# def handle_mem_intense_sim(simulation: Simulation, filename: str):
#     """ returns Ntot if it can, otherwise writes to file when memory runs out, then continues simulation from where it left off."""
    
#     if os.path.exists(filename):
#         mode = 'a'  # Append to the file if it already exists
#         start_from_last_step = True
#     else:
#         mode = 'w'  # Create a new file and write to it if it doesn't exist
#         start_from_last_step = False

#     try:
#         if start_from_last_step:  
#             #continue simulation
#             new_sim = copy_sim(simulation)

#             #load previous simulation last step from memory
#             last_step = np.load(filename, mmap_mode='r', allow_pickle=True)[-1]

#             new_sim.startingNtot = last_step[1]
#             new_sim.starting_ice = last_step[1]-last_step[0]
#             #run new simulation
#             handle_mem_intense_sim(new_sim, filename)
#         else:
#             return simulation.getNtot() #i.e. test_2d_asym.getNtot()

#     except MemoryError:
#         print("Ran out of memory when trying to get Ntot. Writing results to file, and continuing simulation instead.")
#         #print('shape of array being saved is: ', test_2d_asym.results()['y'].shape)

#         #ys = np.array(simulation.results()['y'])
#         #ys = simulation.results()['y'] #this is a list of arrays, so it is not a numpy array
#         ys = np.asarray(simulation.results()['y']) #create view of list as numpy array

#         #write to npy file
#         if os.path.exists(filename):
#             # Append the new data to the existing data along the first axis and then
#             # Save the combined data to the file, overwriting the old data
#             with open(filename, 'ab') as f:
#                 #I am using allow pickle because the data is a list, but I don't want to
#                 # use extra memory before saving by converting to an array first.
#                 # When it is loaded it will be an array as desired.
#                 #np.save(filename, ys, allow_pickle=True) 
#                 np.save(filename, ys)
#         else:
#             #np.save(filename, ys, allow_pickle=True)
#             np.save(filename, ys)

#         #continue simulation
#         new_sim = copy_sim(simulation)

#         del simulation._results #remove results from memory

#         #run new simulation
#         handle_mem_intense_sim(new_sim, filename)

#write or append the simulation results to a file
def woa_to_file(simulation, filename):
    if os.path.exists(filename):
        results_existing = np.load(filename, mmap_mode='r')#, allow_pickle=True)
        #results_existing = np.load(filename)#, allow_pickle=True)

        #results_existing = np.load(filename, allow_pickle=True)
        #with open(filename, 'wb') as f:
        #copy to temporary different file to use disk space instead of memory, and allow memmapping of the exisitng results file
        np.save('temp_'+filename, np.concatenate((results_existing, simulation.results()['y']), axis=0))
        
        # replace the exisitng file with the new file
        os.replace('temp_'+filename,filename)#note: overwrites existing results with existing results + new results
        pass
        #mode = 'ab'  # Append to the file if it already exists
    #else:
    #    mode = 'wb'  # Create a new file and write to it if it doesn't exist
    mode = 'wb'  # Create a new file and write to it if it doesn't exist

    with open(filename, mode) as f:
        #I am using allow pickle because the data is a list, but I don't want to
        # use extra memory before saving by converting to an array first.
        # When it is loaded it will be an array as desired.
        #np.save(f, simulation.results()['y'], allow_pickle=True)
        np.save(f, simulation.results()['y'])
    pass

def continue_from_file(simulation, filename):
    #if path exists
    if os.path.exists(filename):
        #with open(filename,'rb') as f:
            #last_step = np.load(f,mmap_mode='r',allow_pickle=True)[-1]
        last_step = np.load(filename,mmap_mode='r')[-1]
        #last_step = np.load(filename, mmap_mode='r', allow_pickle=True)[-1]
        #last_step = np.load(filename, allow_pickle=True)[-1]
    else:
        print('path does not exist')

    return continue_from_surface(simulation, last_step)

def continue_from_surface(simulation,last_step):
    new_sim = copy_sim(simulation)
    new_sim.startingNtot = last_step[1]
    new_sim.starting_ice = last_step[1]-last_step[0]
    
    return new_sim