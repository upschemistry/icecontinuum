
"""
Created on Mon Jun 19 2023

@author: ella
"""


from os import dup
import numpy as np
import psutil
import scipy as sp
import diffusionstuff10 as ds
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp



class runSim():
    def __init__(self,model=None,shape=(None,),method="LSODA",sigmaImethod="parabolic",mem_check=False,mem_threshold=100e9):
        self.model = model
        self.method = method
        self.shape = shape

        self.mem_check = mem_check
        self.memory_threshold = mem_threshold

        ##TODO only for 1d atm
        self.dimension = 1

        ### Experimental args ###
        Nbar = 1
        Nstar = 0.9/(2*np.pi)
        D = 1.6e-4
        nmpermonolayer = 0.3
        umpersec_over_mlyperus = (nmpermonolayer/1e3*1e6)
        nu_kin = 250
        nu_kin_mlyperus = nu_kin/umpersec_over_mlyperus
        # Supersaturation
        sigma0 = 0.19
        sigmaIcorner = 0.22
        center_reduction = 0.25
        c_r = center_reduction/100
        #########################

        ##TODO only for 1d atm
        k = np.arange(0,nx)

        xmax = 150
        nx = self.shape[0]
        self.x = np.arange(0,xmax,nx)
        self.deltaX = self.x[1] - self.x[0]

        t_init = 0.0
        dtmaxtimefactor = 2
        dtmax = self.deltaX**2/D
        self.deltaT = dtmax/dtmaxtimefactor
        # self.tinterval = [t_init, t_init+self.deltaT]
        self.tinterval = np.arange(t_init,200000,1000)

        if sigmaImethod=='sinusoid':
            sigmaI = ds.getsigmaI(self.x,xmax,center_reduction,sigmaIcorner,method='sinusoid')
        elif sigmaImethod=='parabolic':
            sigmaI = ds.getsigmaI(self.x,xmax,center_reduction,sigmaIcorner,method='parabolic')
        else:
            print('bad choice')
            return

        self._extra_vars = {
            "Nbar" : Nbar,
            "Nstar" : Nstar,
            "D" : D, #D/nx
            "nu_kin" : nu_kin,
            "nu_kin_mlyperus" : nu_kin_mlyperus,
            "sigma0" : sigma0,
            "sigmaI" : sigmaI,
            "sigmaIcorner" : sigmaIcorner,
            "t_init" : t_init,
            "nx" : nx,
            "xmax" : xmax,
            "c_r" : c_r,
            "k" : k
        }

        self._results = {None:None}
        self._rerun = False
        pass

    def run(self,print_progress=True,print_count_layers=False) -> None:
        if self._results != {None:None}:
            self._rerun = True

        warningIssued = False

        # parameters
        Nbar = self._extra_vars["Nbar"]
        Nstar = self._extra_vars["Nstar"]
        # D = self._extra_vars["D"]
        # nu_kin = self._extra_vars["nu_kin"]
        # nu_kin_mlyperus = self._extra_vars["nu_kin_mlyperus"]
        # sigma0 = self._extra_vars["sigma0"]
        # sigmaI = self._extra_vars["sigmaI"]
        # sigmaIcorner = self._extra_vars["sigmaIcorner"]
        t_init = self._extra_vars["t_init"]
        nx = self._extra_vars["nx"]
        # xmax = self._extra_vars["xmax"]
        # c_r = self._extra_vars["c_r"]
        # k = self._extra_vars["k"]

        #############
        Nice = np.ones(self.shape)
        Nqll = ds.getNQLL(Nice,Nstar,Nbar)
        Ntot = Nqll + Nice

        # nICE = fftnorm(Nice)
        nQLL = fftnorm(Nqll)
        nTOT = fftnorm(Ntot)

        n0 = np.concatenate((nQLL,nTOT))
        
        if self.model != None:
            def myRHS(t,y):
                return self.model(t,y,self._extra_vars)
        else:
            print("woah stop there please")
            return
        
        nlast = dup(n0)
        tlast = t_init
        self._results['y'] = [n0]
        self._results['t'] = [self.tinterval[0]]

        counter = 0
        layer = 0
        ttot = 0

        if self.mem_check:
            memcheckcounter = 0
        while True:
            if self.mem_check:
                memcheckcounter += 1
                if memcheckcounter % 4 == 0:# only check every 4 steps to save time checking memory
                    memory_available = psutil.swap_memory().free
                    if memory_available <= self.memory_threshold:
                        #write or append to file
                        print('Memory usage exceeded threshold. Halting.')
                        return

            sol = solve_ivp(myRHS,self.tinterval,nlast)
            nlast = sol.y[:,-1]
            tlast += self.deltaT

            self._results['y'].append(nlast)
            self._results['t'].append(tlast)

            ttot += self.deltaT

            nTOTlast, nQLLlast = np.reshape(nlast,(2,nx))
            minpoint = min(nTOTlast)
            maxpoint = max(nTOTlast)
            print(counter-1, int(nTOTlast[0]),maxpoint-minpoint)
            counter += 1

            if self.uselayers:  #TODO
                if nTOTlast[0] > self.layermax-1:#TODO
                    break
            else:
                if counter > self.countermax-1:#TODO
                    break
        pass
    pass

def fftnorm(u_full):
    """Computes normalized FFT (such that FFT and IFFT are symmetrically normalized)
    ### rfft excludes redundant outputs; complex conjugates are left out so every bin only has the positive frequencies
    ### norm='forward' arg normalizes transform by 1/n, inverse transform is unscaled
    ### TODO: qs
    ###     does this mean the inverse transform is natrually scaled back to map to original untransformed scale??
    ###     is this better than (eg MZKdV.py) mult by 1/N??

    ### NOTE: rfft RETURNS A SHORTENED ARRAY (about haf the size)

    Parameters
    ----------
    u_full : 1D Numpy Array (N,)
        The vector whose discrete FFT is to be computed

    Returns
    -------
    normalizedFFT : 1D Numpy Array (N,)
        The transformed version of that vector
    """

    normalizedFFT = np.fft.rfft(u_full,norm = "forward")
    return normalizedFFT

def ifftnorm(u_full):
    """Computes normalized IFFT (such that FFT and IFFT are symmetrically normalized)
    ### irfft is inverse of rfft, norm is forward for symmetric normalizations

    Parameters
    ----------
    u_full : 1D Numpy Array (N,)
        The vector whose discrete IFFT is to be computed

    Returns
    -------
    normalizedIFFT : 1D Numpy Array (N,)
        The transformed version of that vector
    """
    
    normalizedIFFT = np.fft.irfft(u_full, norm = "forward")
    return normalizedIFFT

def convolution(nTOTk,nu_kin_mlyperus,sigmaM,Nstar):
    """Computes Fourier transform of the nonlinear term in the QLL PDE
    
    - 2 pi N^* nuKin sigmaM cos(Ntot)
    
    Computed in real space and then converted back
    to Fourier space.

    ### sigmaM is eqn 6 in paper, ~~delta kinda~~ but entirely dependent on Nqll
    ###

    Parameters
    ----------
    nT : 1D Numpy Array (N,)
        Total water layers, in k space
        
    nu_kin_mlyperus : TBD
        Deposition rate in monolayers per microsecond
        
    sigmaM : TBD
        Microscopic supersaturation, dependent on position through m dependence on Nqll, in real space
        
    Nstar : TBD
        Parameterizes variation about the mean (simply a best fit parameter??)

    Returns
    -------
    convo : 1D Numpy Array (N,)
        Fourier transform of the nonlinear term
    """
    
    # compute double sum in real space, then apply scalar multiplier
    convo = - 2 * np.pi * Nstar * nu_kin_mlyperus * fftnorm(sigmaM * np.cos(2*np.pi * ifftnorm(nTOTk)))
    return convo

def nTotRHS(nQLLk,nu_kin_mlyperus,sigmaM,k,D):
    """Computes RHS of the ODE for the positive modes of Ntot
    
    ##TODO: why factor of 2pi?? how are we normalizing here??
    dnk/dt = -k^2 D nkQLL + 2 pi FFT(sigma_m) nu_kin
    
    
    Parameters
    ----------
    nQLL : 1D Numpy Array (N,)
        Positive modes of state vector for quasi-liquid layers
        
    nu_kin_mlyperus : TBD
        Deposition rate in monolayers per microsecond
        
    sigmaM : TBD
        Microscopic supersaturation, dependent on position through m dependence on Nqll, in real space
        
    k : 1D Numpy Array (N,)
        Vector of (nonredundant??) wavenumbers
        
    D : float
        Diffusion coefficient

    Returns
    -------
    dnTot : 1D Numpy Array (N,)
        Rate of change of positive modes of nTot
    """

    dnTot = -k**2 * D * nQLLk + nu_kin_mlyperus * fftnorm(sigmaM)
    return dnTot

def nQLLRHS(nTOTk,nQLLk,nu_kin_mlyperus,sigmaM,k,D,Nstar):
    """Computes RHS of the ODE for the positive modes of Ntot
    
    ##TODO: i dont see where the 2pi in dn0 comes from??
    dn0/dt = 2 * pi * sigma_m * nu_kin
    dnk/dt = -k^2 D nkQLL
    
    
    Parameters
    ----------
    nTot : 1D Numpy Array (N,)
        Positive modes of state vector for total layers
    
    nQLL : 1D Numpy Array (N,)
        Positive modes of state vector for quasi-liquid layers
        
    nu_kin_mlyperus : TBD
        Deposition rate in monolayers per microsecond
        
    sigmaM : TBD
        Microscopic supersaturation, dependent on position through m dependence on Nqll, in real space
        
    k : 1D Numpy Array (N,)
        Vector of wavenumbers
        
    D : float
        Diffusion coefficient
        
    Nstar : float
        TBD

    Returns
    -------
    dnQLL : 1D Numpy Array (N,)
        Rate of change of positive modes of nTot
    """
    
    ## convolution computed: 
    ## -2*np.pi*Nstar*nu_kin_mlyperus * fftnorm(sigmaM * np.cos(2*np.pi * ifftnorm(nTot)))
    convo = convolution(nTOTk,nu_kin_mlyperus,sigmaM,Nstar)
    dnQLL = -k**2 * D * nQLLk + convo
    return dnQLL