# -*- coding: utf-8 -*-
"""
Spyder Editor


"""
import numpy as np
import pylab as plt


data1 = np.loadtxt('20ns_top.txt') # data saved from excel as tab-delimited text
data2 = np.loadtxt('20ns_bottom.txt') # data saved from excel as tab-delimited text

plt.scatter(data1[:,0],data1[:,1],c = 'b', s = 10, marker = 'x')
plt.scatter(data2[:,0],data2[:,1],c = 'r', s = 10, marker = 'x')

plt.xlabel("Frame number (frame = 20ps)")
plt.ylabel("NLL")
plt.ylim([0,375])
plt.xlim([0,1050])



plt.show()