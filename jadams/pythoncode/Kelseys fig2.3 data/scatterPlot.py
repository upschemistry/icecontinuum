# -*- coding: utf-8 -*-
"""
Spyder Editor


"""
import numpy as np
import pylab as plt


data1 = np.loadtxt('20ns_top.txt') # data saved from excel as tab-delimited text
data2 = np.loadtxt('20ns_bottom.txt') # data saved from excel as tab-delimited text

time = []

for i in range(1001):
    time.append(data1[i,0] *20*0.001)  #changes from number of frames to number of nanoseconds


plt.scatter(time,data1[:,1],c = 'b', s = 10, marker = 'x')
plt.scatter(time,data2[:,1],c = 'r', s = 10, marker = 'x')


plt.xlabel("Time (ns)")
plt.ylabel("NLL")
plt.ylim([0,375])
plt.xlim([0,20])



plt.show()