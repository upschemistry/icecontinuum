# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 12:18:25 2015

@author: nesh
"""

import numpy as np
import pylab as plt
data = np.loadtxt('NLLdata.txt') # This was saved from excel as "tab-delimited text"
Nice = data[:,0]/240
Nicemod = np.mod(Nice,1)
NLL = data[:,2]/240


Niceforshow = np.linspace(0,1)
Nstar = 0.10
Nbar = 1.03
phase = .48
NLLforshow = Nbar + Nstar*np.sin((Niceforshow+phase)*2*np.pi)
plt.plot(Nicemod,NLL,'o', Niceforshow,NLLforshow)