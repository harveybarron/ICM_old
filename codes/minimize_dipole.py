# -*- coding: utf-8 -*-
"""
Created on Tue May 25 14:00:12 2021

@author: Chandraniva
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import scipy.optimize as opt
from Computing_ys_in_annuli import get_2Dys_in_annuli

pixsize = 1.7177432059
conv =  27.052 #kpc
    
f = fits.open('map2048_MILCA_Coma_20deg_G.fits')
data = f[1].data

def get_dipoleMetric(x):    
    rs,ys,step_size,maxval = get_2Dys_in_annuli(maxval = x)
    
    ys_new = np.zeros(len(ys)+1)
    rs_new = np.zeros(len(rs)+1)
    
    rs_new[1:] = rs
    ys_new[1:] = ys

    x=np.arange(700) # where npix is the number of pixels in the x direction, ie NAXIS1 from the header
    y=np.arange(700) # as above, NAXIS2=NAXIS1

    ys_new[0]= ys[0]
     
    image_length=121*2 #arcmin
    
    x_ind = np.nonzero(((x-maxval[0])*pixsize>=-(image_length/2)) & (((x-maxval[0])*pixsize<=(image_length/2))))
    y_ind = np.nonzero(((y-maxval[1])*pixsize>=-(image_length/2)) & (((y-maxval[1])*pixsize<=(image_length/2))))
    
    #arrays used to store y fluctuations and normalised y fluctuations
    y_fluc = np.zeros((len(x_ind[0]),len(y_ind[0])))
    normalised_y_fluc = np.zeros((len(x_ind[0]),len(y_ind[0])))
    
    t1,t2 = 0,0
    for i in x_ind[0]:
        for j in y_ind[0]:
            #radius = np.sqrt((i-maxval[0])**2 + (j-maxval[1])**2)*pixsize   
            
            f = x[2]
            theta = x[3]
            r_ellipse = np.sqrt((f*(np.cos(theta))**2 + 1/f*(np.sin(theta))**2)*(i - maxval[0])**2  \
                        + (f*(np.sin(theta))**2 + 1/f*(np.cos(theta))**2)*(j - maxval[1])**2  \
                        + 2*(np.cos(theta))*(np.sin(theta))*(f-1/f)*(i - maxval[0])*(j - maxval[1]))*pixsize
            
            y_radius = np.interp(r_ellipse, rs_new, ys_new)
            y_fluc[t1][t2] = data[i][j] - y_radius
            normalised_y_fluc[t1][t2] = (data[i][j] - y_radius)/y_radius
            t2 += 1
        t2 = 0
        t1 += 1
    
    mid = int(len(y_fluc)/2.)
    dist = int((len(y_fluc)/2.)/4)
    y_centre = y_fluc[mid-int(dist):mid+int(dist),mid-dist:mid+int(dist)]
    y_centre = y_centre.reshape(-1)
    idx_neg = np.where(y_centre<0)[0]
    idx_pos = np.where(y_centre>0)[0]
    dipole_metric = sum(abs(y_centre[i]) for i in idx_neg)+ \
                    sum(abs(y_centre[i]) for i in idx_pos)
    return dipole_metric

"""
xi,xf = 345,355.01
yi,yf = 345,355.01
steps_x = 1
steps_y = 1
z1 = int(((xf-xi)/steps_x)+1)
z2 = int((yf-yi)/steps_y+1)
metrics = np.ones((z1,z2))
xc,yc = xi,yi
print(metrics)
itr1, itr2 = 0,0
while xc<=xf:
    while yc<=yf:
        metrics[itr1,itr2] = get_dipoleMetric([xc,yc,1.,0])
        itr2 += 1
        yc+=steps_y
    yc = yi
    itr2 = 0
    itr1+=1
    xc+=steps_x
    print(".",end="")
    
print(metrics)

"""
fi,ff= 0.4,3.01
theta_i,theta_f = 0, np.pi
steps1 = 0.2
steps2 = 0.5
z1 = int(((ff-fi)/steps1)+1)
z2 = int((theta_f-theta_i)/steps2+1)

metrics = np.ones((z1,z2))

print(metrics)

fs,thetas = fi, theta_i

itr1, itr2 = 0,0

while fs<ff:
    while thetas<theta_f:
        metrics[itr1,itr2] = get_dipoleMetric([352.8,349.6,fs,thetas])
        itr2 += 1
        thetas+=steps2
    thetas = theta_i
    itr2 = 0
    itr1+=1
    fs+=steps1
    print(".",end="")
    
print(metrics)

minval = np.unravel_index(np.argmin(metrics), metrics.shape) 


f_opt = minval[0]*steps1+fi
theta_opt = minval[1]*steps2+theta_i
print(metrics[minval[0],minval[1]],"\t Optimum (f, theta) = " , f_opt,theta_opt)
"""
y_opt = minval[0]*steps_x+xi 
x_opt = minval[1]*steps_y+yi
print(metrics[minval[0],minval[1]],"\t Optimum (y,x) = " , y_opt,x_opt)
"""  
plt.imshow(metrics, cmap = 'Greys')
plt.annotate('minima',(minval[1],minval[0]),fontsize=14, color = 'red', alpha = 1)
plt.colorbar()
plt.xlabel("Theta")
plt.ylabel("f")
plt.title("Dipole Metric")
#plt.savefig("Dipole Metric.png", dpi = 400)
plt.show()





























