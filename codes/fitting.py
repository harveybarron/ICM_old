# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 17:46:02 2021

@author: Chandraniva
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import scipy.optimize as opt

f = fits.open('map2048_MILCA_Coma_20deg_G.fits')

data = f[1].data    #data is extracted

x,y = np.unravel_index(data.argmax(), data.shape)       #index of max y value is stored

print("centre of the cluster at", x,y)  # max y is assumed to be centre of coma

r_i = 1   #initial r taken to be 1 to avoid division by zero error 
r_f = 65    #final r taken to be of length 65
size = r_f - r_i 

#arrays to store log10 values of y's and r's where r is the distance from centre of coma
ys_log = np.zeros(size)
rs_log = np.zeros(size)

#arrays to store values of y's and r's
ys = np.zeros(size)
rs = np.zeros(size)

#r runs from initial to final r 
for r in range (r_i,r_f,1):
    dtheta = 1/(100*(r))    #step size is chosen according to r. Factor of 100 optimized the running time of code on my system. For better accuracy, choose a larger factor.
    #angle theta, count and sum_y are initialised for each r 
    theta = 0
    count = 0       #later used to calculate average y
    sum_y = 0       #used to store the sum of y's for a given radius
    while theta <= 2*np.pi:   #theta runs from 0 to 2pi
        i = int(round(x+r*np.cos(theta)))   #row index calculated for given theta from centre of coma
        j = int(round(y+r*np.sin(theta)))   #column index calculated for given theta from centre of coma
        sum_y += data[i][j]         #sum of y's is calculated
        count += 1                  #keeps count of number of pixels used to calculate sum
        theta += dtheta             #theta is updated after each step
    
    avg_y = sum_y/count       #avg y calculated
    
    #index of ararys runs from 0 to r_f-r_i, which is the size
    ys[r-r_i] = (avg_y)     
    rs[r-r_i] = (r)
    
    #log of the values are also separately stored
    ys_log[r-r_i] = np.log10(avg_y)
    rs_log[r-r_i] = np.log10(r)
    print ("-", end = "")   #cause my system is slow :)
    
#beta model function defined
def f(x,a,b,c):
    return a/(1+(x/b)**2)**c

constants = opt.curve_fit(f, rs, ys)    #optimize module is used to curve fit 
#optimized constants a,b and c are stored
a_fit = constants[0][0]
b_fit = constants[0][1]
c_fit = constants[0][2]

#arrays to store the beta model fucntion and it's log
beta_fit = np.zeros(size)
beta_fit_log = np.zeros(size)

#the Beta model function is stored 
for r in range (r_i,r_f,1):
    beta_fit[r-r_i] = f(r, a_fit, b_fit, c_fit )
    beta_fit_log[r-r_i] = np.log10(f(r, a_fit, b_fit, c_fit ))

print("\n")    
print(a_fit, b_fit, c_fit)

#x and y labels are set accordingly
x_ticks = ['100', '1000']
y_ticks = ['10','1','0.1']

t11 = [np.log10(2),np.log10(22)]
t12 = [-5,-6,-7]

plt.xticks(ticks=t11, labels=x_ticks, size='small')
plt.yticks(ticks=t12, labels=y_ticks, size='small')

#average y profile along with the beta model is plotted as a function of distance from centre of coma    
plt.plot(rs_log, ys_log, label = 'Milca y-profile')
plt.plot( rs_log, beta_fit_log, label = 'Beta-model fit')
plt.xlabel("Distance from centre of cluster (kpc)" ,fontsize=11)
plt.ylabel("Average y profile (10^-6)" ,fontsize=11)
plt.title("Avg y profile in MILCA map", fontsize=13)
plt.legend()
plt.savefig('Figure 2 with beta fit.svg', dpi = 1200)
plt.show()





















        