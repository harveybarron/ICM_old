"""
This script consists of a single function which extracts data from the fits file
and caclculates average 2D ys in bins of 3 aarcmin from 0 to 112 kpc.
Warning: This script requires the fits files 'map2048_MILCA_Coma_20deg_G.fits' to work
"""

import numpy as np
from astropy.io import fits

pixsize = 1.7177432059  
conv =  27.052 #kpc

def get_2Dys_in_annuli():
    f = fits.open('map2048_MILCA_Coma_20deg_G.fits')   
    data = f[1].data    #data is extracted
    
    x=np.arange(700) # where npix is the number of pixels in the x direction, ie NAXIS1 from the header
    y=np.arange(700) # as above, NAXIS2=NAXIS1    
    X, Y = np.meshgrid(x, y) # takes your vectors and converts them to a grid
    
    maxval = np.unravel_index(np.argmax(data), data.shape) # get the coordinates of the maximum
    
    R=np.sqrt((X-maxval[0])**2+(Y-maxval[1])**2) # calculate distance in pixel coordinates from the centre – note this should not be rounded
        
    r_initial = 0   #initial r taken to be 1 to avoid division by zero error 
    r_final = 112    #final r taken to be of length 
    step_size = 3 #arcmin
    
    if (r_final-r_initial)%step_size==0:
        size = int((r_final - r_initial) / step_size )
    else:
        size = int((r_final - r_initial)/step_size +1)
    
    #arrays to store values of y's and r's
    ys = np.zeros(size)
    rs = np.zeros(size)
    
    i=r_initial
    t=0
    
    while i<r_final:
    
        in_ann=np.nonzero((R*pixsize>i) & (R*pixsize<(i+step_size)))
        av=np.mean(data[in_ann])
        
        ys[t] = av
        rs[t] = i+step_size/2
        
        t+=1
        i+=step_size
    
    return rs, ys, step_size
     
    