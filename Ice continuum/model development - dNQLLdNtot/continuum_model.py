# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 20:42:43 2016

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
    t = 10000 #still arbitrary timesteps #Total Simulation Time
    deltaT = 2.0 #microseconds
    deltaX = 1.0 #micrometers
    Fliqstart = 1.5 #monolayers 
    current_time = 0
    D = 2e-2 # micrometers^2/microseconds
    #print "Dprime:", Dprime
    
    deprate = .2 # Monolayers per microsecond (Hopefully, depends on how the scale translates in the x direction now that x and deltaX are smaller)
    deprate_micronspersecond = deprate*1e6 * 0.37e-3
    #print deprate_micronspersecond 
    depratePFac = 0.99
    #alphaTerr = 1.0 #Not in use yet
    
    # Dependent Parameters
    x = np.arange(0, n, deltaX) # used arange over linspace because it is more beautiful and easier to manipulate
    #print "x:", x
    boxpoints = len(x)
    Fliq = np.zeros(boxpoints) + Fliqstart
    Nice = np.zeros(boxpoints)
    Dprime = D*deltaT/deltaX**2 # New diffperdt #Recast into 20 microns
    #print "Dprime:", Dprime
    xmid = max(x)/2
    
    depratep = deprate*deltaT * depratePFac
    c = (deprate*deltaT - depratep)/xmid**2
    depratesurface = (x-xmid)**2*c+depratep
    
    plt.clf()
    sigma0 = 0.1
    sigmapfac = 0.983
    sigmastepmax = 0.2
    sigmastepmin = sigmastepmax * sigmapfac
    redux = sigmastepmax - sigmastepmin
    sigmastep = redux*(x-xmid)**2/xmid**2 + sigmastepmin
    plt.plot(x, sigmastep)
    
    
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
    #print 'max dt is:', dtmax, 'current dt is:', deltaT
    deltaN = np.zeros(np.shape(Fliq))
    sigavg0 = []
    sigavg250 = []
    sigavg500 = []    
    
    for i in range(int(t/deltaT)):
        #intvalues = (Nice-Niceoffset).astype(int)
        #differences = np.diff(intvalues)
        #stepsUp = mlab.find(differences > 0)
        #stepsDown = mlab.find(differences < 0) + 1 # +1 so we are on lower side of the edge
        #steps = np.append(stepsDown, stepsUp)
        
        dprimecopy = copy.copy(Dprimesurface)
        #dprimecopy[steps] = Dprimeedge
        
        delta = (Fliq - (Nbar - Nstar))/(2*Nstar)
        sigD = (sigmastep - delta * sigma0)/(1+delta*sigma0)        
        depsurf = deprate * sigD * deltaT
        sigavg0.append(sigD[0])
        sigavg250.append(sigD[250])
        sigavg500.append(sigD[500])
        Fliq += depsurf
        
        Fliq = ds.diffuse(Fliq, dprimecopy)
        
        
        
        #Fliq = Fliq + (sigma - depratesurface * (maxF-Fliq)/maxF) # What is sigma?  # Square maxF and Fliq?
        
        for j in range(boxpoints):
            deltaN[j] = ds.getdeltaN(Nice[j], Fliq[j], Nbar, Nstar, Nmono, phi)
        Fliq += deltaN
        Nice -= deltaN
    minpoint = min(Nice)
    print("Height of Ice", minpoint)
    Nice -= minpoint
    
    print("0:", np.mean(sigavg0))    
    print("250:", np.mean(sigavg250))    
    print("500:", np.mean(sigavg500))    
    
    plt.figure(4)
    plt.clf()
    plt.plot(x, Nice, 'k', x, Fliq+Nice, 'b', x, Fliq, 'g')#, x[steps], Fliq[steps], 'o')
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