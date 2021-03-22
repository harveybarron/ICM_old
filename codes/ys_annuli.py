# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 23:48:48 2021

@author: Chandraniva
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import scipy.optimize as opt

pixsize = 1.7177432059
    
conv =  27.052 #kpc


def y_2d():
    f = fits.open('map2048_MILCA_Coma_20deg_G.fits')
    
    data = f[1].data    #data is extracted
    
    x=np.arange(700) # where npix is the number of pixels in the x direction, ie NAXIS1 from the header
    y=np.arange(700) # as above, NAXIS2=NAXIS1
    
    X, Y = np.meshgrid(x, y) # takes your vectors and converts them to a grid
    
    maxval = np.unravel_index(np.argmax(data), data.shape) # get the coordinates of the maximum
    
    R=np.sqrt((X-maxval[0])**2+(Y-maxval[1])**2) # calculate distance in pixel coordinates from the centre – note this should not be rounded
    

    
    r_i = 0   #initial r taken to be 1 to avoid division by zero error 
    r_f = 112    #final r taken to be of length 65
    step_size = 3 #arcmin
    
    if (r_f-r_i)%step_size==0:
        size = int((r_f - r_i) / step_size )
    else:
        size = int((r_f - r_i) / step_size +1 )
    
    
    #arrays to store values of y's and r's
    ys = np.zeros(size)
    rs = np.zeros(size)
    
    i=r_i
    t=0
    
    while i<r_f:
    
        in_ann=np.nonzero((R*pixsize>i) & (R*pixsize<(i+step_size)))
        av=np.mean(data[in_ann])
        
        ys[t] = av
        rs[t] = i+step_size/2
        
        t+=1
        i+=step_size
    
    return rs, ys, step_size, r_i, r_f, size
     
    
