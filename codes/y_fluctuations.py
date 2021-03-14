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
size = r_f  

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
    
    ys[0] = data[x][y]
    ys[r] = (avg_y)
    
    rs[0] = 0
    rs[r] = r
    
    print ("-", end = "")

a = 70   #a = int(np.sqrt(2)*size-1)
print("length of image in terms of matrix length = ",a)

y_fluc = np.zeros((a,a))
normalised_y_fluc = np.zeros((a,a))

for i in range (0,a,1):
    for j in range (0,a,1):
        r = int(np.sqrt(((a/2) - i)**2+((a/2)-j)**2))
        y_fluc[i][j] = data[x-int(a/2)+i][y+int(a/2)-j] - ys[r]
        normalised_y_fluc[i][j] =  (data[x-int(a/2)+i][y+int(a/2)-j] - ys[r])/ys[r]

print("\n")
 
from matplotlib.colors import TwoSlopeNorm

x_ticks = ['-2', '-1.5','-1','-0.5','0','0.5','1', '1.5','2']
y_ticks = ['2', '1.5','1','0.5','0','-0.5','-1', '-1.5','-2']

t11 = [0,8.75,17.5,26.25,35,43.75,52.5,61.25,69]

plt.xticks(ticks=t11, labels=x_ticks, size='small')
plt.yticks(ticks=t11, labels=y_ticks, size='small')

norm = TwoSlopeNorm(vmin=y_fluc.min(), vcenter=0, vmax=y_fluc.max())
pc = plt.pcolormesh(y_fluc, norm=norm, cmap="seismic")     
plt.imshow(y_fluc, cmap = 'seismic')
plt.colorbar(pc)
plt.xlabel("degrees")
plt.ylabel("degrees")
plt.title("y fluctuations in MILCA")
plt.savefig("figure 3.svg", dpi = 1200)
plt.show()

x_ticks = ['-2', '-1.5','-1','-0.5','0','0.5','1', '1.5','2']
y_ticks = ['2', '1.5','1','0.5','0','-0.5','-1', '-1.5','-2']

t11 = [0,8.75,17.5,26.25,35,43.75,52.5,61.25,69]

plt.xticks(ticks=t11, labels=x_ticks, size='small')
plt.yticks(ticks=t11, labels=y_ticks, size='small')

plt.xticks(ticks=t11, labels=x_ticks, size='small')
plt.yticks(ticks=t11, labels=y_ticks, size='small')
 
norm = TwoSlopeNorm(vmin=normalised_y_fluc.min(), vcenter=0, vmax=normalised_y_fluc.max())
pc = plt.pcolormesh( normalised_y_fluc, norm=norm, cmap="seismic") 
plt.imshow(normalised_y_fluc, cmap = 'seismic')
plt.colorbar(pc)
plt.xlabel("degrees")
plt.ylabel("degrees")
plt.title("normalised y fluctuations in MILCA")
plt.savefig("figure 4.svg", dpi = 1200)
plt.show()




















