# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 17:32:14 2015

@author: Jonathan
"""

import numpy as np
from matplotlib import pyplot as plt
#from matplotlib import animation
from matplotlib import mlab as mlab
import copy
import time
import diffusionstuff as ds

def main():
    sec1 = time.time()
    # Parameters related to equilibrium
    Nbar = 1.0 # new Nbar from VMD, 260K
    Nstar = .9/(2*np.pi)
    phi = 0.0
    Nmono = 1.0
    Niceoffset = ds.getNiceoffset(Nbar, Nstar, Nmono, phi)

    # Want units in terms of micrometers and microseconds
    n = 1000 # Number of points in simulation box
    t = 3000000 #still arbitrary timesteps #Total Simulation Time
    deltaT = 1.0 #microseconds
    deltaX = 1.5 #micrometers
    Fliqstart = 1.5 #monolayers 
    current_time = 0
    D = 2e-2 # micrometers^2/microseconds
    #print "Dprime:", Dprime
    
    deprate = .02 # Monolayers per microsecond (Hopefully, depends on how the scale translates in the x direction now that x and deltaX are smaller)
    deprate_micronspersecond = deprate*1e6 * 0.37e-3
    print deprate_micronspersecond 
    depratePFac = 0.999
    #alphaTerr = 1.0 #Not in use yet
    
    # Dependent Parameters
    x = np.arange(0, n, deltaX) # used arange over linspace because it is more beautiful and easier to manipulate
    #print "x:", x
    boxpoints = len(x)
    Fliq = np.zeros(boxpoints) + Fliqstart
    Nice = np.zeros(boxpoints)
    Dprime = D*deltaT/deltaX**2 # New diffperdt #Recast into 20 microns
    print "Dprime:", Dprime
    xmid = max(x)/2
    
    depratep = deprate*deltaT * depratePFac
    c = (deprate*deltaT - depratep)/xmid**2
    depratesurface = (x-xmid)**2*c+depratep
    
    
    
    #Pre equilibration
    for i in range(boxpoints):
        deltaN = ds.getdeltaN(Nice[i],Fliq[i],Nbar,Nstar,Nmono,phi)
        Fliq[i] += deltaN
        Nice[i] -= deltaN
    
    #plt.figure(5)
    #plt.clf()
    #plt.plot(x, Nice, 'k', x, Fliq+Nice, 'b', x, Fliq, 'g')    
    
    Dprimesurface = np.zeros(boxpoints) + Dprime
    Dprimeedge = 0.99 * Dprime
    deprateedge = deprate* 0.1*deltaT #Additional Deposition at the edge
    
    
        
    dtmax = deltaX**2/D
    print 'max dt is:', dtmax, 'current dt is:', deltaT
    deltaN = np.zeros(np.shape(Fliq))
    
    for i in range(int(t/deltaT)):
        intvalues = (Nice-Niceoffset).astype(int)
        differences = np.diff(intvalues)
        stepsUp = mlab.find(differences > 0)
        stepsDown = mlab.find(differences < 0) + 1 # +1 so we are on lower side of the edge
        steps = np.append(stepsDown, stepsUp)
        
        dprimecopy = copy.copy(Dprimesurface)
        dprimecopy[steps] = Dprimeedge
        
        Fliq = ds.diffuse(Fliq, dprimecopy)
        
        Fliq = Fliq + depratesurface
        #Fliq[steps] = Fliq[steps] + deprateedge
        
        for j in range(boxpoints):
            deltaN[j] = ds.getdeltaN(Nice[j], Fliq[j], Nbar, Nstar, Nmono, phi)
        Fliq += deltaN
        Nice -= deltaN
    minpoint = min(Nice)
    Nice -= minpoint
    
    plt.figure(4)
    plt.clf()
    plt.plot(x, Nice, 'k', x, Fliq+Nice, 'b', x, Fliq, 'g', x[steps], Fliq[steps], 'o')
    sec2 = time.time()
    print "Time taken:", int((sec2-sec1)/60), "min", (sec2-sec1)%60, "secs"

#    saveit = True    
#    if saveit:
#        np.savez_compressed('FallData/D02-S02-E99-RE10-PFAC999-600k.npz', Nice=Nice, Niceoffset=Niceoffset,
#                            Diffofx=Diffofx, diffperdtEdgeAdjust=diffperdtEdgeAdjust,
#                            Fliq=Fliq, rainperdtTerr=rainperdtTerr, Nbar=Nbar,
#                            rainperdtTerrAdjust=rainperdtTerrAdjust, Nstar=Nstar,
#                            Nmono=Nmono, phi=phi, x=x, t=t)

main()

'''
def main():
    sec1 = time.time()    
    startFromScratch = True
    if startFromScratch:    
        # Parameters related to equilibrium
        Nbar = 0.5 #increase to 1.0 to see what happens
        Nstar = .9/(2*np.pi)
        phi = 0.0
        Nmono = 1.0
        Niceoffset = ds.getNiceoffset(Nbar, Nstar, Nmono, phi)
        
        
        n = 1000 # size of simulation box
        t = 2000 # Total simulation time
        current_time = 0
        Fliqstart = 1.5
        diffperdt = 0.035
        supersat = 0.02
        supersatpfac = .99
        alphaTerr = 1.0
        
        #dependent parameters
        supersatp = supersat*supersatpfac
        x = np.linspace(0,n-1, n)
        Fliq = np.zeros(n)+Fliqstart
        Nice = np.zeros(n)
        xmid = max(x)/2
        c = (supersat-supersatp)/xmid**2
        rainperdt = (x-xmid)**2*c+supersatp
        rainperdtTerr = rainperdt*alphaTerr    
        
        #Pre-equilibration
        for i in range(n):
            deltaN = ds.getdeltaN(Nice[i],Fliq[i], Nbar, Nstar, Nmono, phi)
            Fliq[i] += deltaN
            Nice[i] -= deltaN
    
        Diffofx = np.zeros(n) + diffperdt
        diffperdtEdgeAdjust = diffperdt * 0.8
        rainperdtTerrAdjust = rainperdtTerr * 0.10
    else: 
        #IF WE HAVE A SAVED REFERENCE STATE:
        #Comment out the code above and use this to load in all
        #relevant values for the simulation.     
        t = 10
        data = np.load('EulerMethod/TempStorage/D0325-S02-E70-500k.npz')
        Nice = data['Nice']
        Niceoffset = data['Niceoffset']
        Diffofx = data['Diffofx']
        diffperdtEdgeAdjust = data['diffperdtEdgeAdjust']
        Fliq = data['Fliq']
        rainperdtTerr = data['rainperdtTerr']
        Nbar = data['Nbar']
        rainperdtTerrAdjust = data['rainperdtTerrAdjust']
        Nstar = data['Nstar']
        Nmono = data['Nmono']
        phi = data['phi']
        x = data['x']
        current_time = data['t']
        n = len(x)
        
    dx = 1
    D = np.mean(Diffofx)
    dtmax = dx**2/(D)
    dt = 1.0
    print 'max dt is:', dtmax, 'current dt is:', dt
    deltaN = np.zeros(np.shape(Fliq))

    for i in range(t):
        
        #look for edges
        intValues = (Nice-Niceoffset).astype(int)        
        differences = np.diff(intValues)
        stepsUp = mlab.find(differences > 0)
        stepsDown = mlab.find(differences < 0)+1 #+1 so we are on lower side of edge
        steps = np.append(stepsDown, stepsUp)
        # adjust diffusion coefficient at edges
        Dofx = copy.copy(Diffofx)        
        Dofx[steps] = diffperdtEdgeAdjust
        
        #diffuse liquid across surface
        Fliq = ds.diffuse(Fliq, Dofx*dt)
        
        #incoming vapor
        Fliq = Fliq + rainperdtTerr*dt
        Fliq[steps] = Fliq[steps] + rainperdtTerrAdjust[steps]*dt
        
        #equilibrate
        for j in range(n):
            deltaN[j] = ds.getdeltaN(Nice[j], Fliq[j], Nbar, Nstar, Nmono, phi)
        # making correction    
        Fliq += deltaN
        Nice -= deltaN
            
    #minpoint = min(Nice)
    #Nice -= minpoint
            
    plt.figure(2)        
    plt.clf()
    plt.plot(x, Nice, 'k', x, Fliq+Nice, 'b', x, Fliq, 'g', x[steps], Fliq[steps], 'o') 
    
    # tell me how long the simulation took
    sec2 = time.time()
    print "Time taken:", int((sec2-sec1)/60), "min", (sec2-sec1)%60, "secs"
    
    saveit = False
    t = t+current_time
    print "# of steps is now:", t
    if saveit:
        np.savez_compressed('EulerMethod/ForAnalysis/D03-S02-E60-RE10-500k.npz', Nice=Nice, Niceoffset=Niceoffset,
                            Diffofx=Diffofx, diffperdtEdgeAdjust=diffperdtEdgeAdjust,
                            Fliq=Fliq, rainperdtTerr=rainperdtTerr, Nbar=Nbar,
                            rainperdtTerrAdjust=rainperdtTerrAdjust, Nstar=Nstar,
                            Nmono=Nmono, phi=phi, x=x, t=t)


    """
    import cProfile
    cProfile.run('main()', 'pstats')

    from pstats import Stats
    p = Stats('pstats')
    p.strip_dirs().sort_stats('cumtime').print_stats(15)
    """                            
    
    """                  
                     
    # for confirming existence of steady state                    
    extra_t = 50000
    for i in range(extra_t):
        
        #look for edges
        intValues = (Nice-Niceoffset).astype(int)        
        differences = np.diff(intValues)
        stepsUp = mlab.find(differences > 0)
        stepsDown = mlab.find(differences < 0)+1 #+1 so we are on lower side of edge
        steps = np.append(stepsDown, stepsUp)
        # adjust diffusion coefficient at edges
        Dofx = copy.copy(Diffofx)        
        Dofx[steps] = diffperdtEdgeAdjust
        
        #diffuse liquid across surface
        Fliq = ds.diffuse(Fliq, Dofx*dt)
        
        #incoming vapor
        Fliq = Fliq + rainperdtTerr*dt
        Fliq[steps] = Fliq[steps] + rainperdtTerrAdjust[steps]*dt
        
        #equilibrate
        for j in range(n):
            deltaN[j] = ds.getdeltaN(Nice[j], Fliq[j], Nbar, Nstar, Nmono, phi)
        Fliq += deltaN
        Nice -= deltaN
            
    minpoint = min(Nice)
    Nice -= minpoint
            
    plt.figure(10)        
    plt.clf()
    plt.plot(x, Nice, 'k', x, Fliq+Nice, 'b', x, Fliq, 'g', x[steps], Fliq[steps], 'o') 
    
    # tell me how long the simulation took
    sec3 = time.time()
    print "Additional Time taken:", int((sec3-sec2)/60), "min", (sec3-sec2)%60, "secs"
    
    #uncomment the following to make a file with all important values saved    
    t = t+extra_t
    print "# of steps is now:", t
    if saveit:
        np.savez_compressed('EulerMethod/ForAnalysis/D03-S02-E60-RE10-600k.npz', Nice=Nice, Niceoffset=Niceoffset,
                            Diffofx=Diffofx, diffperdtEdgeAdjust=diffperdtEdgeAdjust,
                            Fliq=Fliq, rainperdtTerr=rainperdtTerr, Nbar=Nbar,
                            rainperdtTerrAdjust=rainperdtTerrAdjust, Nstar=Nstar,
                            Nmono=Nmono, phi=phi, x=x, t=t)
                
    """

'''



