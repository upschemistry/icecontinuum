# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 09:16:30 2015

@author: Jonathan
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib import mlab as mlab
import diffusionstuff as ds
import copy

#from pylab import *
dpi = 100
#import time



"""
#This diffuse function works if the diffusion coefficient is constant
#If diffusion is constant, use this code because it is slightly faster
#than the other diffuse        
def oldDiffuse(x, diff):
    y = copy.copy(x) #doesn't diffuse properly if we say y = x
    l = len(x)
    for i in range(1,l-1):
        y[i] = x[i] + diff[i+1] * x[i+1] - diff[i] * 2*x[i] + diff[i-1] * x[i-1]
    # Boundary Conditions 
    y[0] = x[0] + diff[1] * x[1] - diff[0] * x[0]
    y[l-1] = x[l-1] + diff[l-2] * x[l-2]- diff[l-1] * x[l-1]
    return y

"""
        
def nextStep():
    data = np.load('FallData/D02-S02-E95-RE10-PFAC999-500k.npz') # load in steady state
    #acquire data from file    
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
    initTime = data['t']
    cnt = 0
    n = len(x)    
    
    while cnt < 1100: #interval to animate
    
        cnt += 1
        # find edges        
        intValues = (Nice-Niceoffset).astype(int)        
        differences = np.diff(intValues)
        stepsUp = mlab.find(differences > 0)
        stepsDown = mlab.find(differences < 0)+1 #+1 so we are on lower side of edge
        steps = np.append(stepsDown, stepsUp)
        
        # adjust diffusion coefficient at edges
        Dofx = copy.copy(Diffofx)        
        Dofx[steps] -= diffperdtEdgeAdjust
            
        #diffuse liquid across surface
        Fliq = ds.diffuse(Fliq, Dofx)
        
        #incoming vapor
        Fliq = Fliq + rainperdtTerr
        Fliq[steps] = Fliq[steps] + rainperdtTerrAdjust[steps]
        
        #equilibrate
        for i in range(n):
            deltaN = ds.getdeltaN(Nice[i], Fliq[i], Nbar, Nstar, Nmono, phi)
            Fliq[i] += deltaN
            Nice[i] -= deltaN
        if cnt%1==0:
            yield x, Fliq, Nice, steps, cnt, initTime# send info for animation before looping again


fig = plt.figure()
ax = plt.axes(xlim=(0,999), ylim = (0,30))
#line, = ax.plot([],[]) #way to get single plot
#lines = ax.plot( *([[],[]]*4) ) #quick way to get multiple plots
#Set up the plots we want with characteristics we want
line1 = ax.plot([],[])[0]
line2 = ax.plot([],[])[0]
line3 = ax.plot([],[])[0]
line4 = ax.plot([],[], 'o')[0]
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
xdata, liqdata, icedata, stepdata = [], [], [], []

"""
def ani_frame():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    im = ax.imshow(rand(300,300), cmap = 'gray', interpolation = 'nearest')
    im.set_clim([0,1])
    fig.set_size_inches([5,5])
    tight_layout()
"""  
  
def animate(data):
    x,y,z,steps,count,initial= data #get data from nextStep()

    xdata = copy.copy(x)
    liqdata = copy.copy(y) #could be replace, just a thought
    icedata = copy.copy(z)
    #icedata = icedata - min(icedata)
    
    stepdata = copy.copy(steps)
    time_text.set_text('time = ' + str(count+initial))
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, 3.0 + icedata[0])
    ax.figure.canvas.draw()
    line1.set_data([xdata,liqdata])
    line2.set_data([xdata,icedata])
    line3.set_data([xdata,icedata+liqdata])
    line4.set_data([xdata[stepdata],liqdata[stepdata]])

anim = animation.FuncAnimation(fig,animate, nextStep)
###writer = animation.writers['ffmpeg'](fps = 30)

#for saving video file
#anim.save('variousimages/neshpresentation/FirstStep4.mp4', writer = 'mencoder', dpi = dpi)
                            
#plt.show()