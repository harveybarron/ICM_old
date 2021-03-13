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

data = f[1].data

x,y = np.unravel_index(data.argmax(), data.shape) 

print(x,y)  # max y is assumed to be centre of coma

r_i = 1   #initial r
r_f = 65
size = r_f - r_i 

ys_log = np.zeros(size)
rs_log = np.zeros(size)

ys = np.zeros(size)
rs = np.zeros(size)

for r in range (r_i,r_f,1):
    dtheta = 1/(100*(r))
    theta = 0
    count = 0
    sum_y = 0
    while theta <= 2*np.pi:
        i = int(round(x+r*np.cos(theta)))
        j = int(round(y+r*np.sin(theta)))
        sum_y += data[i][j]
        count += 1
        theta += dtheta
    
    avg_y = sum_y/count
    
    ys[r-r_i] = (avg_y)
    rs[r-r_i] = (r)
    
    ys_log[r-r_i] = np.log10(avg_y)
    rs_log[r-r_i] = np.log10(r)
    print ("-", end = "")
    

def f(x,a,b,c):
    return a/(1+(x/b)**2)**c

constants = opt.curve_fit(f, rs, ys)
a_fit = constants[0][0]
b_fit = constants[0][1]
c_fit = constants[0][2]

beta_fit = np.zeros(size)
beta_fit_log = np.zeros(size)

for r in range (r_i,r_f,1):
    beta_fit[r-r_i] = f(r, a_fit, b_fit, c_fit )
    beta_fit_log[r-r_i] = np.log10(f(r, a_fit, b_fit, c_fit ))

print("\n")    
print(a_fit, b_fit, c_fit)

    
plt.plot(rs_log, ys_log, label = 'Milca y-profile')
plt.plot( rs_log, beta_fit_log, label = 'Beta-model fit')
plt.xlabel("Log10 of distance from centre of cluster in terms of matrix size" ,fontsize=11)
plt.ylabel("Log10 of average y profile" ,fontsize=11)
plt.title("Avg y profile in MILCA map", fontsize=13)
plt.legend()
plt.savefig('Figure 2 with beta fit.svg', dpi = 1200)
plt.show()





















        