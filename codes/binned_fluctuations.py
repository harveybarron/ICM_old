# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 01:54:43 2021

@author: Chandraniva
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import scipy.optimize as opt

f = fits.open('map2048_MILCA_Coma_20deg_G.fits')

data = f[1].data    #data is extracted

x=np.arange(700) # where npix is the number of pixels in the x direction, ie NAXIS1 from the header
y=np.arange(700) # as above, NAXIS2=NAXIS1

X, Y = np.meshgrid(x, y) # takes your vectors and converts them to a grid

maxval = np.unravel_index(np.argmax(data), data.shape) # get the coordinates of the maximum

R=np.sqrt((X-maxval[0])**2+(Y-maxval[1])**2) # calculate distance in pixel coordinates from the centre – note this should not be rounded

pixsize = 1.7177432059

conv =  27.052 #kpc

r_i = 0   #initial r taken to be 1 to avoid division by zero error 
r_f = 112    #final r taken to be of length 65
step_size = 3 #arcmin

if (r_f-r_i)%step_size==0:
    size = int((r_f - r_i) / step_size +1)
else:
    size = int((r_f - r_i) / step_size +2 )


#arrays to store values of y's and r's
ys = np.zeros(size)
rs = np.zeros(size)

i=r_i
t=1

while i<r_f:

    in_ann=np.nonzero((R*pixsize>i) & (R*pixsize<(i+step_size)))
    av=np.mean(data[in_ann])
    
    
    ys[t] = av
    rs[t] = i+step_size/2
    
    t+=1
    i+=step_size
    
def f(x,a,b,c):
    return a/(1+(x/b)**2)**c

constants = opt.curve_fit(f, rs[1:], ys[1:])    #optimize module is used to curve fit 

a_fit = constants[0][0]

ys[0]=a_fit
rs[0]=0 
    
a=121 #arcmin

plt.plot(rs,ys)
plt.show()

print("\nlength of image in arcmin = ",a)


x_ind = np.nonzero(((x-maxval[0])*pixsize>=-(a/2)) & (((x-maxval[0])*pixsize<=(a/2))))
y_ind = np.nonzero(((y-maxval[1])*pixsize>=-(a/2)) & (((y-maxval[1])*pixsize<=(a/2))))

#arrays used to store y fluctuations and normalised y fluctuations
y_fluc = np.zeros((len(x_ind[0]),len(y_ind[0])))
normalised_y_fluc = np.zeros((len(x_ind[0]),len(y_ind[0])))


t1,t2 = 0,0

for i in y_ind[0]:
    for j in x_ind[0]:
        radius = np.sqrt((i-maxval[0])**2 + (j-maxval[1])**2)*pixsize
        y_radius = np.interp(radius, rs, ys)
        y_fluc[t1][t2] = data[i][j] - y_radius
        normalised_y_fluc[t1][t2] = (data[i][j] - y_radius)/y_radius

        t2 += 1
    t2 = 0
    t1 += 1


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