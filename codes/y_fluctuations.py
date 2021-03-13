# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 20:30:32 2021

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

r_i = 1   
r_f = 65
size = r_f - r_i 

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
    
    print ("-", end = "")

a = int(np.sqrt(2)*size-2)

y_fluc = np.zeros((a,a))
normalised_y_fluc = np.zeros((a,a))

for i in range (0,a,1):
    for j in range (0,a,1):
        r = int(np.sqrt((int(a/2) - i)**2+(int(a/2)-j)**2))
        y_fluc[i][j] = data[x-int(a/2)+i][y+int(a/2)-j] - ys[r]
        normalised_y_fluc[i][j] =  (data[x-int(a/2)+i][y+int(a/2)-j] - ys[r])/ys[r]

print("\n")
 
from matplotlib.colors import TwoSlopeNorm

norm = TwoSlopeNorm(vmin=y_fluc.min(), vcenter=0, vmax=y_fluc.max())
pc = plt.pcolormesh(y_fluc, norm=norm, cmap="seismic")     
plt.imshow(y_fluc, cmap = 'seismic')
plt.colorbar(pc)
plt.title("y fluctuations in MILCA")
plt.savefig("figure 3.svg", dpi = 1200)
plt.show()

norm = TwoSlopeNorm(vmin=normalised_y_fluc.min(), vcenter=0, vmax=normalised_y_fluc.max())
pc = plt.pcolormesh( normalised_y_fluc, norm=norm, cmap="seismic") 
plt.imshow(normalised_y_fluc, cmap = 'seismic')
plt.colorbar(pc)
plt.title("normalised y fluctuations in MILCA")
plt.savefig("figure 4.svg", dpi = 1200)
plt.show()



















