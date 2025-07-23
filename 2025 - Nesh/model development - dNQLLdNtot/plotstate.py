
# coding: utf-8

# In[1]:

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams


# In[6]:

# Load a trajectory
lastfile = 'NeshData/continuum_model6 8-3-2016[1][2].npz'
print "loading", lastfile
npzfile = np.load(lastfile)
Fliq = npzfile['Fliq']
Nice = npzfile['Nice']
Nbar = npzfile['Nbar']
Nstar = npzfile['Nstar']
x = npzfile['x']
boxpoints = len(x)
nx = len(x)
deltaX = x[1]-x[0]
Ntot = Fliq + Nice
nmid = int(nx/2)
nquart = int(nx/4)
xmid = max(x)/2
xmax = x[nx-1]
minpoint = min(Nice)


# Graphics parameters
ticklabelsize = 15
fontsize = 20
linewidth = 2


# Open a plot window
plt.figure(1) # Can also specify size: plt.figure(1,figsize=(8,4))
plt.plot(         x-xmid, Nice-minpoint, 'k',          x-xmid, Fliq+Nice-minpoint, 'b',          x-xmid, Fliq, 'g', lw=linewidth)
plt.xlabel(r'x ($\mu m$)',fontsize=fontsize)
plt.ylabel('Layers',fontsize=fontsize)
rcParams['xtick.labelsize'] = ticklabelsize 
rcParams['ytick.labelsize'] = ticklabelsize
plt.grid('on')
plt.xlim([-25,25])


# In[ ]:




# In[ ]:



