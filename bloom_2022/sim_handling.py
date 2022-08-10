# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
import time
import diffusionstuff7 as ds
from copy import copy as dup
from scipy.integrate import odeint
#from scipy.integrate import solve_ivp
from numba import int64

#for animations
import matplotlib.animation as animation

#for saving simulations
import pickle

"""This module describes a Simulation object that can be used to run simulations of the ice continuum;
    as well as a function to test performance of functions.
    
    @Author: Max Bloom
        contact: mbloom@pugetsound.edu @mbloom1 on GitHub
"""

class Simulation():
    """Simulation class for integratable differential equation models.
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
    
    @author: Max Bloom 
    """

    def __init__(self, model, shape, method= "LSODA", atol= 1e-6, rtol= 1e-6, noisy=False, noise_stddev=0.01, layermax=10):
        """Initialize the Simulation
        Parameters
        ----------
        model (func): integratable differential equation model
        shape (tuple): shape of initial condition

        method (str): integration method
        atol (float): absolute tolerance
        rtol (float): relative tolerance
        """
        #solve_ivp arguments
        self.model = model #integratable differential equation model 
        self.method = method #default to emulate odeint
        self.atol = atol #default absolute tolerance
        self.rtol = rtol #default relative tolerance
        self.shape = shape #shape of initial condition
        self.dimension = len(shape) #dimension of initial condition

        # These are run control parameters
        """ Fliq is the QLL (shape: nx)
            Ntot is the the combined values of the ice layers (Nice), combined with the QLL layer (Nliq or Fliq or NQLL variously referred to as)
                Ntot is shaped (2, nx)
        """
        self.noisy_init = noisy
        self.noise_std_dev = noise_stddev
        # Flag for explicit updating Fliq(Ntot) every step 
        self.updatingFliq = True
        # Set up a maximum number of iterations or layers
        self.uselayers = True
        
        if self.uselayers:
            self.layermax_0D = 4
            self.layermax_1D = 50
            self.layermax_2D = 20
        #else:
        self.countermax_0D = 100
        self.countermax_1D = 15000
        self.countermax_2D = 1000#15000
        
        #NOTE: the variables below are not part of self so are not saved to file by Simulation.save() (?)
        niter = 1
        #Setting up the 2D system
        #nx = 500 # Number of points in simulation box
        nx = self.shape[0] # Number of points in simulation box
        xmax = 50 # range of x
        self.x = np.linspace(0, xmax, nx)
        deltaX = self.x[1]-self.x[0]

        #ny = nx
        ny = self.shape[1] 
        ymax = round(xmax * ny/nx)
        self.y = np.linspace(0, ymax, ny)
        deltaY = self.y[1]-self.y[0]
        
        Nbar = 1.0 # new Nbar from VMD, 260K
        Nstar = .9/(2*np.pi)
        
        #Time and diffusion parameters
        # Just conversions
        nmpermonolayer = 0.3
        umpersec_over_mlyperus = (nmpermonolayer/1e3*1e6)
        # Diffusion coefficient
        D = 0.02e-2 # micrometers^2/microsecond

        # Time steps
        dtmaxtimefactor = 10
        dtmaxtimefactor = 50
        dtmax = deltaX**2/D
        self.deltaT = dtmax/dtmaxtimefactor
        tmax = self.countermax_2D*self.deltaT
        t0 = 0.0


        # Deposition rate
        nu_kin = 49 # microns/second
        deprate = nu_kin/umpersec_over_mlyperus # monolayers per microsecond
        deprate_times_deltaT = deprate * self.deltaT
        # Supersaturation
        sigma0 = 0.19
        sigmastepmax = 0.20 #-0.10 # Must be bigger than sigma0 to get growth, less than 0 for ablation
        center_reduction = 1.0 #0.25 # In percent
        c_r = center_reduction/100
        # Diffusion coefficient scaled for this time-step and space-step
        DoverdeltaX2 = D/deltaX**2
        DoverdeltaY2 = D/deltaY**2 #unused

        #Save variables not used in model via self.* to an array for saving
        self._extra_vars = {
            "Nbar":Nbar,
            "Nstar":Nstar,
            "D":D,
            "dtmax":dtmax,
            "dtmaxtimefactor":dtmaxtimefactor,
            "nu_kin":nu_kin,
            "deprate":deprate,
            "deprate_times_deltaT":deprate_times_deltaT,
            "sigma0":sigma0,
            "sigmastepmax":sigmastepmax,
            "c_r":c_r,
            "DoverdeltaX2":DoverdeltaX2,
            "DoverdeltaY2":DoverdeltaY2,
            "nx":nx,
            "ny":ny,
            "xmax":xmax,
            "ymax":ymax,
            "deltaX":deltaX,
            "deltaY":deltaY,
            "niter":niter,
            "t0":t0
        }
        self._extra_vars_types = {key:type(value) for key,value in self._extra_vars.items()}
        
        # Initialize the results dictionary
        self._results = {None:None}
        #Intitialize other internal attributes
        self._plot = None
        self._animation = None

        #self.tinterval = [t0, tmax] #this is for solve_ivp
        self.tinterval = [t0, self.deltaT] #this is for odeint

        self.float_params = {'Nbar':Nbar,'Nstar':Nstar, 'sigma0':sigma0, 'deprate':deprate, 'DoverdeltaX2':DoverdeltaX2, 'center_reduction':center_reduction, 'sigmastepmax':sigmastepmax}
        self.int_params = {'niter':niter,'nx':nx,'ny':ny}

        #internal attributes
        self._plot = None #matplotlib figure
        self._animation = None #matplotlib animation
        self._results = {None:None} #solve_ivp (-like) dictionary of results
        pass

    def run(self) -> None:
        # try:#try to run simulation and package results
            # This is the 2-d run using odeint
            # Bundle parameters for ODE solver
            #float_params = np.array([Nbar, Nstar, sigma0, deprate, DoverdeltaX2])
            #int_params = np.array(list(map(int64,[niter,nx,ny]))) # functions in ds7 require int64
            #unpack parameters
            Nbar, Nstar, sigma0, deprate, DoverdeltaX2, center_reduction, sigmastepmax = self.float_params.values()
            packed_float_params = np.array([Nbar, Nstar, sigma0, deprate, DoverdeltaX2])
            niter, nx, ny = self.int_params.values()
            packed_int_params = np.array(list(map(int64,self.int_params.values()))) # sigmastep math in f2d in diffusionstuff7 requires int64

            # Lay out the system
            Nice = np.ones(self.shape)
            Fliq = ds.getNliq_2d_array(Nice,Nstar,Nbar,niter) # Initialize as a pre-equilibrated layer of liquid over ice

            sigmastep_2d = ds.getsigmastep_2d(self.x,self.y,center_reduction,sigmastepmax)
            
            if self.noisy_init:
                # Initialize with noise
                noise = np.random.normal(0,self.noise_std_dev,self.shape)
                Nice += noise
            Ntot = Fliq + Nice
            # nmid = int(nx/2)
            # nquart = int(nx/4)
            # xmid = max(x)/2
            # xmax = x[nx-1]

            y0 = np.array([Fliq,Ntot])
            ylast = dup(y0)
            tlast = dup(self.tinterval[0])

            # Initial conditions for ODE solver goes into keeper dictionary
            self._results['y'] = [y0]
            self._results['t'] = [self.tinterval[0]]

            # Call the ODE solver
            Ntot0_start = Ntot[0,0]
            Ntot0 = Ntot[0,0]
            updatingFliq = True#False
            counter = 0
            lastlayer = 0
            lastdiff = 0
            while True:
                # Integrate up to next time step
                y = odeint(self.model, np.reshape(ylast,np.prod(np.shape(ylast))), self.tinterval, args=(packed_float_params,packed_int_params,sigmastep_2d), rtol=1e-12)
                
                # Update the state                  #NOTE: prod(shape(ylast)) is like (2*nx*ny)
                ylast = np.reshape(y[1],(2,nx,ny))
                tlast += self.deltaT
                counter += 1
                
                # Make some local copies, with possible updates to Fliq
                Fliq, Ntot = ylast
                if updatingFliq:
                    Fliq = ds.getNliq_2d_array(Ntot,Nstar,Nbar,niter) # This updates to remove any drift
                    ylast[0] = Fliq
                Nice = Ntot - Fliq

                #for calculating layers
                Ntot0 = Ntot[0,0]

                # Stuff into keeper arrays for making graphics
                self._results['y'].append(ylast)
                self._results['t'].append(tlast)

                # Update counters and see whether to break
                layer = Ntot0-Ntot0_start
                if (layer-lastlayer) > 0:
                    minpoint = np.min(Nice)
                    maxpoint = np.max(Nice)
                    print(counter-1, lastlayer, maxpoint-minpoint, maxpoint-minpoint-lastdiff)
                    lastdiff = maxpoint-minpoint
                    lastlayer += 1
                    
                # Test whether we're finished
                if self.uselayers:
                    print("appx progress:" , round((layer/(self.layermax_2D-1))*100, 2),"%",end="\r")
                    if sigmastepmax > 0:
                        if layer > self.layermax_2D-1:
                            print('breaking because reached max number of layers grown')
                            break
                    else:
                        if layer < -self.layermax_2D:
                            print('breaking because reached max number of layers ablated')
                            break
                else:
                    if counter > self.countermax_2D-1:
                        print('breaking because reached max number of iterations')
                        break
            
            #solve_ivp does not support terminating after a given number of layers
            #self._results = solve_ivp(self.model, self.t_span, self.y0, t_eval=self.t_eval, method=self.method, atol=self.atol, rtol=self.rtol, args=self._args)

            #TODO: save parameters that the model needs
            # Nice=Nicekeep, Fliq=Fliqkeep,
            #                     x=x, t=tkeep, 
            #                     Nbar=Nbar, Nstar=Nstar,
            #                     sigma0=sigma0, c_r=c_r, D=D, L=L, 
            #                     nu_kin=nu_kin, nu_kin_ml=nu_kin_ml, 
            #                     sigmastepmax=sigmastepmax, sigmastepstyle=sigmastepstyle,
            #                     ykeep_0Darr=ykeep_0Darr,
            #                     tkeep_0D=tkeep_0D,  
            #                     dtmaxtimefactor = dtmaxtimefactor,
            #                     deltaT = deltaT
        # except Exception as e:
        #     print(e)
        #     print('Error in simulation')
            pass

    def plot(self, completion=1, figurename='', ice=True,tot=False,liq=False, surface=True, contour=False):# -> matplotlib_figure: ## , method = 'surface'): #TODO: test plotting
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
        if self._plot == None:
            #create plot of results
            num_steps = len(self.results()['t'])
            step = int((num_steps-1)*completion)

            Fliq, Ntot = [],[]
            for i in range(num_steps):
                next_Fliq, next_Ntot = self._results['y'][i]
                Fliq.append(next_Fliq)
                Ntot.append(next_Ntot)    
            Fliq,Ntot = np.array(Fliq), np.array(Ntot)
            Nice = Ntot - Fliq

            #access coordinate arrays for plotting
            xs, ys = np.meshgrid(self.x, self.y)

            # Plot the results
            self._plot = plt.figure(figurename)
            if self.dimension == 0:#NOTE: shape is stil 1d for the zero d model- this is a TODO
                if ice: 
                    plt.plot(self.t, Nice, label='ice')
                if tot:
                    plt.plot(self.t, Ntot, label='total')
                if liq:
                    plt.plot(self.t, Fliq, label='liquid')
                plt.legend()
                plt.xlabel('Time')
                plt.ylabel('Layers of ice')
            elif self.dimension == 1:
                if ice:
                    plt.plot(self.x, Nice)
                if tot:
                    plt.plot(self.x, Ntot)
                if liq:
                    plt.plot(self.x, Fliq)
                plt.xlabel('x')
                plt.ylabel('Layers of ice')
            elif self.dimension == 2:
                ax = plt.axes(projection='3d')
                if surface:
                    if ice:
                        ax.plot_surface(X=xs, Y=ys, Z=Nice[step], cmap='viridis')#, vmin=0, vmax=200)
                    if tot:
                        ax.plot_surface(X=xs, Y=ys, Z=Ntot[step], cmap='YlGnBu_r')#, vmin=0, vmax=200)
                    if liq:
                        ax.plot_surface(X=xs, Y=ys, Z=Fliq[step], cmap='YlGnBu_r')
                elif contour: #elif method == 'contour':
                    levels = np.arange(-6,12,0.25)
                    if ice:
                        ax.contour(xs,ys, Nice[step], extent=(0, 2, 0, 2), cmap='YlGnBu_r', vmin=0, vmax=200, zorder=1, levels=levels)
                    if tot:
                        ax.contour(xs,ys, Ntot[step], extent=(0, 2, 0, 2), cmap='YlGnBu_r', vmin=0, vmax=200, zorder=1, levels=levels)
                    if liq:
                        ax.contour(xs,ys, Fliq[step], extent=(0, 2, 0, 2), cmap='YlGnBu_r', vmin=0, vmax=200, zorder=1, levels=levels)
            else:
                print('Error: dimension not supported')
                return None
        plt.show()
        #return self._plot 
        pass
    
    def animate(self, proportionalSpeed=True, ice=True, tot=False, liq=False, surface=True, crossSection=False):
        #TODO: does not graph in 3d, also does not save animation as instance attribute
        if self._animation == None:
            #create animation of results
            num_steps = len(self.results()['t'])
            #shape of results is (num_steps, 2, nx, ny)
            Fliq, Ntot = [],[]
            for step in range(num_steps):
                next_Fliq, next_Ntot = self._results['y'][step]
                Fliq.append(next_Fliq)
                Ntot.append(next_Ntot)    
            Fliq,Ntot = np.array(Fliq), np.array(Ntot)
            Nice = Ntot - Fliq
            #shape of fliq, ntot and nice should be (num_steps, nx, ny)

            #access coordinate arrays for plotting
            xs, ys = np.meshgrid(self.x, self.y)
            
            #3d animation of the results
            self._anim_fig = plt.figure()
            ax = plt.axes(projection='3d')
            def update_surface(num):
                ax.clear() # remove last iteration of plot 
                #labels
                ax.set_xlabel(r'$x (\mu m$)')#,fontsize=fontsize)
                ax.set_ylabel(r'$y (\mu m$)')#,fontsize=fontsize)
                ax.set_zlabel(r'$ice \ layers$')#,fontsize=fontsize)
                #limits
                ax.set_zlim3d(-self.layermax_2D, self.layermax_2D)
                ax.set_ylim(0, max(self.y))
                ax.set_xlim(0, max(self.x))

                if surface:
                    plot_func = ax.plot_surface
                #elif contour: #elif method == 'contour':
                #    plot_func = ax.contour
                else:
                    plot_func = ax.wireframe

                if crossSection:
                    xmid = round(np.shape(Nice)[0]/2)
                    if ice:
                        plot_func(X=xs[xmid:], Y=ys[xmid:], Z=Nice[num][xmid:][:],cmap ='viridis')# cmap='viridis')#, vmin=0, vmax=200) #plot half of the surface of the ice
                    if tot:
                        plot_func(X=xs[xmid:], Y=ys[xmid:], Z=Ntot[num][xmid:][:], cmap='cividis')#, vmin=0, vmax=200) #plot half the surface of the QLL
                else:
                    if ice:
                        plot_func(X=xs, Y=ys, Z=Nice[num], cmap='viridis')#, vmin=0, vmax=200) #plot the surface of the ice 
                    if tot:
                        plot_func(X=xs, Y=ys, Z=Ntot[num], cmap='YlGnBu_r')#, vmin=0, vmax=200)#plot the surface of the QLL
                pass

            self._animation = animation.FuncAnimation(self._anim_fig, update_surface, num_steps, interval=100, blit=False, cache_frame_data=False, repeat = True)
            plt.show()
            #if proportionalSpeed:#TODO: scale interval to make length of gif/mp4 be 10 seconds, scaling speed of animation by factor proportional to length of simulation
                #interval = 

        #return self._animation
        pass

    def results(self) -> dict:
        """ Returns results of simulation (handles running if necessary) """
        if self._results == {None:None}:
            self.run()
        return self._results

    def save(self, _id = []) -> None:
        """ Saves Simulation object to a pickle file
        
        args:
            _id: list of strings to append to filename: what makes this simulation unique
        """
        # Saving these results to file
        #if Nice[0] > 100000: #what is this nesh?
        #    Nice -= 100000
        _id = ''.join('_'+i for i in _id)
        filename = self.model.__name__+'_simulation'+_id+'.pkl'
        with open(filename, 'wb') as f:
            print("saving to", f)
            pickle.dump(self, f)
        pass

    def load(self, filename: str) -> None:
        """ Loads and initializes Simulation object from pickle file  """
        with open(filename, 'rb') as f:
            self = pickle.load(f)
        pass
    
    def save_plot(self, filename) -> None:
        """ saves plot of simulation object """
        if self._plot == None:
            self.plot()
        #Save the results as an image
        filename = '3d_model_results_'+str(self.layermax_2D)+'_layers'
        self._plot.savefig(filename+'.png', dpi=300)
        pass

    def save_animation(self, filename:str, filetype:str) -> None:
        """ saves animation of simulation object """
        if self._animation == None:
            self.animate()

        try:
            if filetype == 'mp4':
                Writer = animation.writers['ffmpeg']
                writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
            elif filetype == 'gif':
                Writer = animation.writers['imagemagick']
                writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
            else:
                print('filetype not supported')
                return
        except Exception as e:
            print(e)
            print('Error creating animation file writer')
            return
        try:
            self._animation.save(filename+'.'+filetype, writer=writer)
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

"""
Performance testing functions for the ice model.
"""
#Meta testing parameters
def multiple_test_avg_time(func, args, n_tests = 50):
    """
    Test a function n_tests times and return the average time taken for the function.
    """
    times = []
    for i in range(n_tests):
        start = np.float64(time())
        func(*args)
        times.append(time()-start)

    retval = float(np.mean(times))
    print("Time to run "+str(func.__name__)+" on average for "+ str(n_tests) +" tests: ", retval, "seconds")
    return retval

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