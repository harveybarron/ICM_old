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

data = f[1].data    #data is extracted

x,y = np.unravel_index(data.argmax(), data.shape)       #index of max y value is stored

print("centre of the cluster at", x,y)  # max y is assumed to be centre of coma

r_i = 1   #initial r taken to be 1 to avoid division by zero error 
r_f = 65    #final r taken to be of length 65
size = r_f  

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
    ys[0] = data[x][y]
    ys[r] = (avg_y)
    
    rs[0] = 0
    rs[r] = r
    
    print ("-", end = "")   #cause my system is slow :)
    
#a is the length of the square corresponding to 4 degrees 
a=70 #a = int(np.sqrt(2)*size-1)
print("\nlength of image in terms of matrix length = ",a)

#arrays used to store y fluctuations and normalised y fluctuations
y_fluc = np.zeros((a,a))
normalised_y_fluc = np.zeros((a,a))

for i in range (0,a,1):
    for j in range (0,a,1):
        r = int(np.sqrt(((a/2) - i)**2+((a/2)-j)**2))   #distance of i,jth element calculated from centre of square
        y_fluc[i][j] = data[x-int(a/2)+i][y+int(a/2)-j] - ys[r]     #fluctuation of  i,jth element calculated by subtracting the average profile at that distant from the y value at that pixel
        normalised_y_fluc[i][j] =  (data[x-int(a/2)+i][y+int(a/2)-j] - ys[r])/ys[r]    #normalised by dividing the fluctuation with the average profile

print("\n")
 
from matplotlib.colors import TwoSlopeNorm     #Needed in plotting diverging colormaps

#x and y labels at approriate intervals are set
x_ticks = ['-1', '-0.75','-0.5','-0.25','0','0.25','0.5', '0.75','1']
y_ticks = ['-1', '-0.75','-0.5','-0.25','0','0.25','0.5', '0.75','1']

t11 = [0,8.75,17.5,26.25,35,43.75,52.5,61.25,69]

plt.xticks(ticks=t11, labels=x_ticks, size='small')
plt.yticks(ticks=t11, labels=y_ticks, size='small')

#the min, max and centre of the colormap is defined and y fluctuations are plotted
norm = TwoSlopeNorm(vmin=y_fluc.min(), vcenter=0, vmax=y_fluc.max())
pc = plt.pcolormesh(y_fluc, norm=norm, cmap="seismic")     
plt.imshow(y_fluc, cmap = 'seismic')
plt.colorbar(pc)
plt.xlabel("degrees")
plt.ylabel("degrees")
plt.title("y fluctuations in MILCA")
plt.savefig("figure 3.svg", dpi = 1200)
plt.show()

#x and y labels at approriate intervals are set
x_ticks = ['-1', '-0.75','-0.5','-0.25','0','0.25','0.5', '0.75','1']
y_ticks = ['-1', '-0.75','-0.5','-0.25','0','0.25','0.5', '0.75','1']

t11 = [0,8.75,17.5,26.25,35,43.75,52.5,61.25,69]

plt.xticks(ticks=t11, labels=x_ticks, size='small')
plt.yticks(ticks=t11, labels=y_ticks, size='small')

plt.xticks(ticks=t11, labels=x_ticks, size='small')
plt.yticks(ticks=t11, labels=y_ticks, size='small')
 
#the min, max and centre of the colormap is defined and y fluctuations are plotted
norm = TwoSlopeNorm(vmin=normalised_y_fluc.min(), vcenter=0, vmax=normalised_y_fluc.max())
pc = plt.pcolormesh( normalised_y_fluc, norm=norm, cmap="seismic") 
plt.imshow(normalised_y_fluc, cmap = 'seismic')
plt.colorbar(pc)
plt.xlabel("degrees")
plt.ylabel("degrees")
plt.title("normalised y fluctuations in MILCA")
plt.savefig("figure 4.svg", dpi = 1200)
plt.show()




















