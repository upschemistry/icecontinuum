#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 17:43:50 2023

@author: nesh
"""

# Resources
import numpy as np
from matplotlib import pyplot as plt

# Parameters of the system
l = 25
npts = 500
midpt = np.round(npts/2).astype(int); print(midpt)
x = np.linspace(-l,l,npts)

# Let's look at a steady-state solution
N_l = 11
k = N_l/l**2
f_ss = k*x**2
plt.figure()
plt.plot(x,f_ss)

