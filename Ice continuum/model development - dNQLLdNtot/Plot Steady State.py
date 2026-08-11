# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 14:21:25 2015

@author: chemistry
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import mlab as mlab


data = np.load('FallData/D02-S02-E99-RE10-PFAC99-500k.npz')
Nice = data['Nice']
Niceoffset = data['Niceoffset']
Fliq = data['Fliq']
x = data['x']
#Diffofx = data['Diffofx']
current_time = data['t']
n = len(x)

#look for edges
intValues = (Nice-Niceoffset).astype(int)        
differences = np.diff(intValues)
stepsUp = mlab.find(differences > 0)
stepsDown = mlab.find(differences < 0)+1 #+1 so we are on lower side of edge
steps = np.append(stepsDown, stepsUp)

plt.figure(10)     
plt.clf()
plt.plot(x, Nice, 'k', x, Fliq+Nice, 'b', x, Fliq, 'g', x[steps], Fliq[steps], 'o') 
"""
dx = 1
D = np.mean(Diffofx)
dtmax = dx**2/(2*D)
dt = 1
print 'max dt is:', dtmax, 'current dt is:', dt


data = np.load('EulerMethod/D075-S02-E40-RE10-600k.npz')
Nice = data['Nice']
Niceoffset = data['Niceoffset']
Fliq = data['Fliq']
x = data['x']
current_time = data['t']
n = len(x)

#look for edges
intValues = (Nice-Niceoffset).astype(int)        
differences = np.diff(intValues)
stepsUp = mlab.find(differences > 0)
stepsDown = mlab.find(differences < 0)+1 #+1 so we are on lower side of edge
steps = np.append(stepsDown, stepsUp)

plt.figure(16)     
plt.clf()
plt.plot(x, Nice, 'k', x, Fliq+Nice, 'b', x, Fliq, 'g', x[steps], Fliq[steps], 'o') 


#plt.figure(3)
#plt.clf()
#plt.plot(x, Fliq2-Fliq)

"""