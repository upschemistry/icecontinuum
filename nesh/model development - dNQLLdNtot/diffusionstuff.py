# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 15:01:47 2015

@author: nesh, jonathan
"""
import numpy as np
import copy

def diffuse(y_old, diff):
    y = copy.copy(y_old) #doesn't diffuse properly if we say y = y_old
    l = len(y_old)
    for i in range(1,l-1):
        y[i] = y_old[i] + ((diff[i+1]-diff[i-1])/2)*((y_old[i+1]-y_old[i-1])/2) + diff[i]*(y_old[i+1]-2*y_old[i]+y_old[i-1])
    # Boundary Conditions 
    y[0] = y_old[0] + ((diff[1]-diff[0])/2)*((y_old[1]-y_old[0])/2)+diff[0]*(y_old[1]-2*y_old[0]+y_old[0]) #assuming second derivative for x[0] is essentially the same as  it is for x[1]
    y[l-1] = y_old[l-1] + ((diff[l-2]-diff[l-1])/2)*((y_old[l-2]-y_old[l-1])/2)+diff[l-1]*(y_old[l-2]-2*y_old[l-1]+y_old[l-1])
    return y
    
def diffuseconstantD(y_old, diff):
    y = copy.copy(y_old) #doesn't diffuse properly if we say y = y_old
    l = len(y_old)
    for i in range(1,l-1):
        y[i] = y_old[i] + diff[i]*(y_old[i+1]-2*y_old[i]+y_old[i-1])
    
    # These are the correct boundary Conditions 
#    y[0] = y_old[0] + diff[0]*(y_old[1]-2*y_old[0]+y_old[0]) 
#    y[l-1] = y_old[l-1] + diff[l-1]*(y_old[l-2]-2*y_old[l-1]+y_old[l-1])

    #   This is a debugging substitute
#    y[0] = y_old[0]  
#    y[l-1] = y_old[l-1] 

    #   This is a debugging substitute
    y[0] = y_old[0] + diff[0]*(y_old[1]-2*y_old[0]+y_old[l-1]) 
    y[l-1] = y_old[l-1] + diff[l-1]*(y_old[0]-2*y_old[l-1]+y_old[l-2])
 
    
    # More debugging
    #print y_old[0:5]
    
    return y

def diffuseconstantDP(y_old, diff):
    l = len(y_old)
    dy = np.zeros(np.shape(y_old))
    for i in range(1,l-1):
        dy[i] = diff[i]*(y_old[i+1]-2*y_old[i]+y_old[i-1])
    
    # These are the correct boundary Conditions 
#    y[0] = y_old[0] + diff[0]*(y_old[1]-2*y_old[0]+y_old[0]) 
#    y[l-1] = y_old[l-1] + diff[l-1]*(y_old[l-2]-2*y_old[l-1]+y_old[l-1])

    #   This is a debugging substitute
#    y[0] = y_old[0]  
#    y[l-1] = y_old[l-1] 

    #   This applies cyclic boundary conditions
    dy[0] = diff[0]*(y_old[1]-2*y_old[0]+y_old[l-1]) 
    dy[l-1] = diff[l-1]*(y_old[0]-2*y_old[l-1]+y_old[l-2])
     
    return dy
    
def diffuseFast(Fliq_old, NIce_old, Ddt, term0, steps, Nbar, Nstar, Nmono, phi):
    # Rain    
    
    # Diffusion
    term1 = np.zeros(np.shape(Fliq_old))
    l = len(Fliq_old)
    for i in range(1,l-1):
        term1[i] = ((Ddt[i+1]-Ddt[i-1])/2)*((Fliq_old[i+1]-Fliq_old[i-1])/2) + Ddt[i]*(Fliq_old[i+1]-2*Fliq_old[i]+Fliq_old[i-1])
    # Boundary Conditions 
    term1[0] = ((Ddt[1]-Ddt[0])/2)*((Fliq_old[1]-Fliq_old[0])/2)+Ddt[0]*(Fliq_old[1]-2*Fliq_old[0]+Fliq_old[0]) #assuming second derivative for x[0] is essentially the same as  it is for x[1]
    term1[l-1] = ((Ddt[l-2]-Ddt[l-1])/2)*((Fliq_old[l-2]-Fliq_old[l-1])/2)+Ddt[l-1]*(Fliq_old[l-2]-2*Fliq_old[l-1]+Fliq_old[l-1])
    
    # Perturbation due to rain and diffusion    
    term2 = term1 + term0
    
    # Conversion to ice
    F1Prime = getFliqPrime(NIce_old,Nbar,Nstar,Nmono,phi)
    dFliqdt = term2*F1Prime/(1+F1Prime)
    dNIcedt = term2 - dFliqdt
    Fliq = Fliq_old + dFliqdt
    NIce = NIce_old + dNIcedt
    return Fliq, NIce

    
def getNFliq(NIce,Nbar,Nstar,Nmono,phi):
    return Nbar+Nstar*np.sin(NIce/Nmono*2*np.pi-phi)

def getFliqPrime(NIce,Nbar,Nstar,Nmono,phi):
    return Nstar*np.cos(NIce/Nmono*2*np.pi-phi)*(1/Nmono*2*np.pi)

def getdeltaN(NIcep,NFliqp,Nbar,Nstar,Nmono,phi):
    deltaN = 0.0
    for i in range(10):
        deltaN = Nbar+Nstar*np.sin((NIcep-deltaN)/Nmono*2*np.pi-phi)-NFliqp
        #deltaN = getNFliq(NIcep-deltaN,Nbar,Nstar,Nmono,phi)-NFliqp
    return deltaN

def fqll_next(fqll_last,Ntot,Nstar,Nbar):
    fstar = Nstar/Nbar
    return 1 + fstar*np.sin(2*np.pi*(Ntot-Nbar*fqll_last))

def getNliq(Ntot,Nstar,Nbar,n):
    fqll_last = 1.0
    for i in range(n):
        fqll_last = fqll_next(fqll_last,Ntot,Nstar,Nbar)
    return fqll_last*Nbar

def getNiceoffset(Nbar=None, Nstar=None, Nmono=None, phi=None):
    # to see the plots, use the getNiceoffset that is commented out
    #get the response curve
    Nicetest = np.linspace(0,1)
    Fliqtest = getNFliq(Nicetest,Nbar,Nstar,Nmono,phi)
    Imin = np.argmin(Fliqtest)
    return Nicetest[Imin]

