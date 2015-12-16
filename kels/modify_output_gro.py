# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 15:57:04 2015

@author: chemistry
"""
import heapq
import numpy 


from numpy import *

F = open('confout_vchange10.gro', 'r') #input
A = open('V_changed13_25molec_v8.5.gro', 'w') #output
XO = []
YO = [] #array of Y position values for oxygen atoms
ZO = [] #array of Z position values for oxygen atoms

V_XO = []
V_YO =[]#array of Y velocities for oxygen atoms
V_ZO = []#array of Z velocities for oxygen atoms


XHW1 = []
YHW1 = [] #Y positions of HW1
ZHW1 = [] #Z positions of HW1

V_XHW1 = []
V_YHW1 = [] #array of Y velocities for HW1 atoms
V_ZHW1 = []#array of Z velocities for HW1 atoms
XHW2 = []
YHW2 = []  
ZHW2 = []
V_XHW2 = []
V_YHW2 = []
V_ZHW2 = []#array of Z velocities for HW2 atoms

V_YLP1 = []#array of Y velocities for LP1 atoms
V_YLP2 = []#array of Y velocities for LP2 atoms
V_YEP = []#array of Y velocities for EP atoms

V_ZLP1 = []#array of Z velocities for LP1 atoms
V_ZLP2 = []#array of Z velocities for LP2 atoms
V_ZEP = []#array of Z velocities for EP atoms

wholething = []
#array for each column of .gro file
col_0 = []
col_1 = []
col_2 = []
col_3 = []
col_4 = []
col_5 = []
col_6 = []
col_7 = []
col_8 = []
col_9 = []

gro_array=[]


with open ("confout_vchange10.gro", "r") as myfile:
    data=myfile.readlines()

    
for i in range(2, len(data)-1):
    string = str(data[i])
    
    col_0.append(string[0:8])

    col_1.append(string[8:20])
    
    col_3.append(string[20:28])
    
    col_4.append(string[28:36]) #contains y  positions
    
    col_5.append(string[36:44])
    
    col_6.append(string[44:52])
    
    col_7.append(string[52:60]) #contains y velocities
    
    col_8.append(string[60:68])

#add the Y positions to array
for i in range(len(col_4)):
    if i%6==0:
        yO =float(col_4[i])
        YO.append(yO)

#add the y velocities for each atom to its array
for i in range(len(col_6)):
    if i%6==0:
        vyO = float(col_7[i])
        V_YO.append(vyO)
    if i%6==1:
        vyhw1 = float(col_7[i])
        V_YHW1.append(vyhw1)
    if i%6==2:
        vyhw2 = float(col_7[i])
        V_YHW2.append(vyhw2)
    if i%6==3:
        vylp1 = float(col_7[i])
        V_YLP1.append(vylp1)
    if i%6==4:
        vylp2 = float(col_7[i])
        V_YLP2.append(vylp2)
    if i%6==5:
        vyep = float(col_7[i])
        V_YEP.append(vyep)

n= 25 #number of molecules to change
vchange_factor = 8.5
yhigh = [] #array of indices of highest y postion values from YO
yhigh.append(heapq.nlargest(n,range(len(YO)),YO.__getitem__)) #find indices of n largest values

cobs = (mean(array(V_YO)**2))**.5 #observed rms speed in y direction
c = round(cobs*vchange_factor, 4) #constant to add to y velocities
#change y velocities
for i in range(n):
    k = yhigh[0][i]
    V_YO[k]= V_YO[k] + c
    V_YHW1[k] = V_YHW1[k] + c
    V_YHW2[k] = V_YHW2[k] + c
    V_YLP1[k] = V_YLP1[k] + c
    V_YLP2[k] = V_YLP2[k] + c
    V_YEP[k] = V_YEP[k] + c

#array of the indices that need to be changed in .gro (will need to add 2 to this later)
change = []
for i in range(n):
    change.append((yhigh[0][i])*6  )
    change.append((yhigh[0][i])*6 + 1)
    change.append((yhigh[0][i])*6 +2)
    change.append((yhigh[0][i])*6 +3)
    change.append((yhigh[0][i])*6 +4)
    change.append((yhigh[0][i])*6 +5)

#change each y velocity for the indices in change[]
for q in change:
    string = data[q+2]
    if q%6 == 0: #if it is an O atom
        add = str(V_YO[(q)/6]) #new y velocity 
        if add[0:1]!= '-': #fix spacing for non-negative numbers
            add = ' ' + add
        if len(add)!= 7: #add a zero if last decimal place is 0
            add = add + '0'
        changed = string[0:53] + add + string[60:69]
        data[q+2] = changed 
    if q%6 == 1: #if it is an HW1 atom
        add = str(V_YHW1[(q-1)/6]) #new y velocity 
        if add[0:1]!= '-':
            add = ' ' + add
        if len(add)!= 7:
            add = add + '0'
        changed = string[0:53] + add + string[60:69]
        data[q+2] = changed 
    if q%6 == 2: #if it is an HW2 atom
        add = str(V_YHW2[(q-2)/6]) #new y velocity 
        if add[0:1]!= '-':
            add = ' ' + add
        if len(add)!= 7:
            add = add + '0'
        changed = string[0:53] + add + string[60:69]
        data[q+2] = changed 
    if q%6 == 3: #if it is an LP1
        add = str(V_YLP1[(q-3)/6]) #new y velocity 
        if add[0:1]!= '-':
            add = ' ' + add
        if len(add)!= 7:
            add = add + '0'
        changed = string[0:53] + add + string[60:69]
        data[q+2] = changed 
    if q%6 == 4: #if it is an LP2
        add = str(V_YLP2[(q-4)/6]) #new y velocity 
        if add[0:1]!= '-':
            add = ' ' + add
        if len(add)!= 7:
            add = add + '0'
        changed = string[0:53] + add + string[60:69]
        data[q+2] = changed 
    if q%6 == 5: #if it is an EP
        add = str(V_YEP[(q-5)/6]) #new y velocity 
        if add[0:1]!= '-':
            add = ' ' + add
        if len(add)!= 7:
            add = add + '0'
        changed = string[0:53] + add + string[60:69]
        data[q+2] = changed 

    
#write the new .gro to output file    
for i in range(len(data)):
    A.write(data[i])   
        
        

