from ctypes import py_object
from functools import cache
from math import floor
import numpy as np
from matplotlib import pyplot as plt
import time
import diffusionstuff8 as ds
from copy import copy as dup
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from numba.types import int64,int32

#for animations
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation, PillowWriter

#for saving simulations
import pickle

"""This module describes a Simulation object that can be used to run simulations of the ice continuum;
    as well as a function to test performance of functions.
    
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
    save(): saves simulation object to pickle file
    save_profile(): saves profile of simulation object to pickle file for
                    reproducability (does not save results)
    load(): loads simulation object from file
    
    save_plot(): saves plot of simulation object
    save_animation(): saves animation of simulation object
    steepness(): returns normalized derivative, goal of indicating where steps of the ice are at given step in a given range/slice
    steady_state(): removes 

    
    @author: Max Bloom 
    """

    #TODO implement starting from an arbtitrary initial condition ( to test low freq. spatial noise)
        #TODO implement starting from a saved run
    def __init__(self, model=None, shape=(None,), method= "LSODA", atol= 1e-6, rtol= 1e-6, noisy=False, noise_stddev=0.01, layermax=0, nonstd_init=False, starting_ice=None, startingNtot=None, discretization_halt=True):
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

        # make initial condition an attribute so it can be accessed for initialization in run()
        self.nonstd_init = nonstd_init
        self.starting_ice = starting_ice
        self.startingNtot = startingNtot

        #calculate dimension of initial condition
        if len(shape) == 1 and shape[0] == 1:#if 1D and only one element
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
        deprate = nu_kin/umpersec_over_mlyperus # monolayers per microsecond # Deposition rate
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

        if self.dimension > 0:
            nx = self.shape[0] # Number of points in simulation box
            #xmax = 50_000 # range of x #TODO: testing 1000x for nanometers instead of microns - larger dx = no numerical instability?
            #xmax = 250 #5x larger #NOTE: this didnt fix the numerical instability
            xmax = 50 #default (in microns)
            self.x = np.linspace(0, xmax, nx)

            deltaX = self.x[1]-self.x[0]
            DoverdeltaX2 = D/deltaX**2 # Diffusion coefficient scaled for this time-step and space-step

            #center_reduction unused by 0d model
            self.center_reduction = 0.25 # In percent #last exp. parameter
            c_r = self.center_reduction/100

            # Time steps
            dtmaxtimefactor = 100 #50 #TODO: meaning?
            dtmax = deltaX**2/D 
            self.deltaT = dtmax/dtmaxtimefactor #factored out of diffusion equation... 
            tmax = self.countermax*self.deltaT #ending time of simulation, used for solve_ivp (?)
        if self.dimension == 2:
            #ny = nx
            ny = self.shape[1] 
            ymax = round(xmax * ny/nx)
            self.y = np.linspace(0, ymax, ny)

            deltaY = self.y[1]-self.y[0]
            DoverdeltaY2 = D/deltaY**2 #unused           
        #self.tinterval = [t0, tmax] #this is for solve_ivp
        self.tinterval = [t0, self.deltaT] #this is for odeint

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
        
        #unpack parameters, based on dimension
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
            #packed_int_params = np.array(list(map(int32,[niter,nx])))#f1d expects int32s
            packed_int_params = np.array([int32(nx)])#f1d expects int32s
            #NOTE: niter removed as of diffusionstuff8
        if self.dimension == 2:
            #DoverdeltaY2 = self.float_params['DoverdeltaY2'] #unused
            ny = self.int_params['ny']
            
            #packed_float_params = np.array([Nbar, Nstar, sigma0, deprate, DoverdeltaX2])
            #packed_int_params = np.array(list(map(int64,[niter,nx,ny]))) # sigmastep math in f2d in diffusionstuff7 requires int64
            packed_int_params = np.array(list(map(int64,[nx,ny])))
        
    #nonstd_init=False, starting_ice=None, startingNtot=None

        if self.nonstd_init:
            #nonstandard initial conditions
            Ntot = self.startingNtot
            if self.dimension == 1:
                qllfunc = lambda x : ds.get_qll_1d(x,self.float_params['Nbar'],self.float_params['Nstar'])
            elif self.dimension == 2:
                qllfunc = lambda x : ds.get_qll_2d(x,self.float_params['Nbar'],self.float_params['Nstar'])

            Fliq = qllfunc(Ntot)
            Nice = Ntot - Fliq

        else:
            # Lay out the initial system
            Nice = np.ones(self.shape)
            Fliq = np.zeros(self.shape)
            Ntot = Fliq + Nice
            if self.noisy_init:
                # Initialize with noise
                noise = np.random.normal(0,self.noise_std_dev,self.shape)
                Nice += noise
        if self.dimension == 0:
            #Fliq = Nbar # ds.getNliq(Nice,Nstar,Nbar,niter) fliq updates to this but starts as nbar
            #sigma = sigmastepmax
            model_args = (packed_float_params,niter)
            #nliq_func = ds.getNliq # function to get to update fliq
        elif self.dimension == 1:
            #Fliq = ds.getNliq_array(Nice,Nstar,Nbar,niter)
            sigma = ds.getsigmastep(self.x, np.max(self.x), self.center_reduction, self.sigmastepmax)
            #model_args = (packed_float_params,packed_int_params,sigma)
            model_args = (packed_float_params,sigma)
            #nliq_func = ds.getNliq_array # function to get to update fliq
            get_Fliq_func = ds.get_qll_1d
        elif self.dimension == 2:
            #NOTE: DEPRECATED #Fliq = ds.getNliq_2d_array(Nice,Nstar,Nbar,niter) # Initialize as a pre-equilibrated layer of liquid over ice
            sigma = ds.getsigmastep_2d(self.x,self.y, self.center_reduction, self.sigmastepmax) # supersaturation
            model_args = (packed_float_params,packed_int_params,sigma) 
            #nliq_func = ds.getNliq_2d_array # function to get to update fliq
            get_Fliq_func = ds.get_qll_2d
            

        # if self.dimension == 0:
        #     y0 = np.array([Fliq, 0.0])#starts at zero ice to get see more layers form at beginning
        # else:
        #     y0 = np.array([Fliq,Ntot]) #NOTE: fliq deprecated as of diffusionstuff8
        y0 = Ntot
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

        if self.method == 'odeint':
            method = 'RK45'
            #y = odeint(self.model, self.tinterval, np.reshape(ylast,np.prod(np.shape(ylast))), args=model_args, rtol=self.rtol, atol=self.atol, tfirst=True)
        else:
            method = self.method
        # Call the ODE solver
        while True:
            #print("shape of ylast: ", np.shape(ylast))
            #print("np.prod(np.shape(ylast)): ", np.prod(np.shape(ylast)))
            # Integrate up to next time step 
            solve_ivp_result = solve_ivp(self.model, self.tinterval, np.reshape(ylast,np.prod(np.shape(ylast))), method=method, args=model_args, t_eval=self.tinterval, rtol=self.rtol, atol=self.atol)
            y = solve_ivp_result.y[:, len(solve_ivp_result.t)-1]#y[:,-1] : get last timestep that solve_ivp returns
            #print("shape of ylast: ", np.shape(ylast))
            #print("np.prod(np.shape(ylast)): ", np.prod(np.shape(ylast)))
            #print('ylast: ', ylast)
            
            # Update the state                 
            if self.dimension == 0:
                ylast = y
            elif self.dimension == 1:
                #ylast = np.reshape(y,(2,self.shape[0])) #used to by ylast = y[1] for odeint
                ylast = y
            elif self.dimension == 2:
                ylast = np.reshape(y,(self.shape[0],self.shape[1])) 
            tlast += self.deltaT
            counter += 1
            
            # Make some local copies
            Ntot = ylast
            
            #Calculate fliq from Ntot
            Fliq = get_Fliq_func(Ntot,Nbar,Nstar)

            #if self.updatingFliq:
            #    Fliq = nliq_func(Ntot,Nstar,Nbar,int32(niter)) # This updates to remove any drift
            #    ylast[0] = Fliq
            Nice = Ntot - Fliq

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
                        prog = round((-1*layer/(self.layermax-1))*100, 2)
                    else:
                        prog = round((layer/(self.layermax-1))*100, 2)
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
                prog = round((counter/self.countermax)*100, 2)
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

    @cache # since frequently combined with subsequent calls
    def getNtot(self, step=None) -> np.ndarray:
        """ Returns Ntot at a given step (or array of all steps if unspecified) """
        if step == None:
            Ntot = []
            for step in range(len(self.results()['t'])):
                next_Ntot = self.results()['y'][step]
                Ntot.append(next_Ntot)
            return np.array(Ntot)
        else:   
            return self.results()['y'][step]#[1]

    def getFliq(self, step=None) -> np.ndarray:
        """ Returns the liquid fraction at a given step (or array of all steps if unspecified) """
        #pre-fill qll_func call with Nbar and Nstar
        #if self.dimension == 0: #todo
        if self.dimension == 1:
            qllfunc = lambda x : ds.get_qll_1d(x,self.float_params['Nbar'],self.float_params['Nstar'])
        elif self.dimension == 2:
            qllfunc = lambda x : ds.get_qll_2d(x,self.float_params['Nbar'],self.float_params['Nstar'])
        
        if step == None:
            Fliq = []
            for x in self.getNtot():
                next_Fliq = qllfunc(x)
                Fliq.append(next_Fliq)
            return np.array(Fliq)
        else:
            return qllfunc(self.results()['y'][step])
    
    def getNice(self, step=None) -> np.ndarray:
        """ Returns the ice fraction at a given step (or array of all steps if unspecified) """
        if step == None:
            return self.getNtot() - self.getFliq()
        else:
            return self.getNtot(step) - self.getFliq(step)

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
            #shape of results is (num_steps, nx, ny)
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
                    ax.set_xlabel(r'x ($\mu m$)')
                    ax.set_ylabel('Layers of ice')
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
                    ax.set_xlabel(r'$x (\mu m$)')#,fontsize=fontsize)
                    ax.set_ylabel(r'$y (\mu m$)')#,fontsize=fontsize)
                    ax.set_zlabel(r'$ice \ layers$')#,fontsize=fontsize)
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

    def save_profile(self, id = [], filename = None) -> None:
        """ Saves profile of Simulation object to a pickle file for reproducability
        
        args:
            id: list of strings to append to filename: what makes this simulation unique
            filename: name of file to save to, if omitted uses id to make filename
        """
        id = ''.join('_'+i for i in id)
        if filename is None:
            filename = self.model.__name__+'_simulation'+id+'.pkl'

        with open(filename, 'wb') as f:
            print("saving to", f)
            pickle.dump(self, f)
        pass

    def _load(self, filename: str) -> py_object:
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
            return pickle.load(f) #self object is loaded

"""
Performance testing functions for the ice model.
"""
#Meta testing parameters
def multiple_test_avg_time(func, args=None, n_tests = 50):
    """
    Test a function n_tests times and return the average time taken for the function.
    """
    times = []
    for i in range(n_tests):
        start = np.float64(time.time())
        if args == None:
            func()
        else:
            func(*args)
        times.append(time.time()-start)

    avg_time_taken = float(np.mean(times))
    print("Time to run "+str(func.__name__)+" on average for "+ str(n_tests) +" tests: ", avg_time_taken, "seconds")
    return avg_time_taken


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