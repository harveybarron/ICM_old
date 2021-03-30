#ys_annuli

#the script calculates the average y profile in the selected region of coma cluster.

#the function returns the following arguments:
#1. The value of rs in arcmin
#2. average y profile of the coma cluster in the confined annuli\
#3. step size i.e., bin size
#4. initial value of radii of the cluster 
#5. final value of radii of the cluster, so that the difference between initial and final values will give width of annuli
#6. size is the no. of annuli the user would like to divide the cluster into

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import scipy.optimize as opt

pixsize = 1.7177432059
    
conv =  27.052 #kpc

#function to calculate average y profile of coma clusters in different y maps using beta-fit model to the MILCA map.
def y_2d(): 
    #reading and extracting the data
    f = fits.open('map2048_MILCA_Coma_20deg_G.fits')
    
    data = f[1].data    
    
    #defining number of pixels i.e, 700 in this case on both the axes-x and y, then converting to a grid 
    x=np.arange(700) 
    y=np.arange(700) 
    X, Y = np.meshgrid(x, y) 
    
    #for maximum coordinates
    maxval = np.unravel_index(np.argmax(data), data.shape) 
    
    # calculate distance in pixel coordinates from the centre 
    R=np.sqrt((X-maxval[0])**2+(Y-maxval[1])**2)  

    #taking initial and final r values for the region of coma cluster taken
    r_i = 0  
    r_f = 112  
    step_size = 3 
    
    if (r_f-r_i)%step_size==0:
        size = int((r_f - r_i) / step_size )
    else:
        size = int((r_f - r_i) / step_size +1 )
    
    
    #defining arrays
    ys = np.zeros(size)
    rs = np.zeros(size)
    
    i=r_i
    t=0
    
    #running a loop to get average y profile from initial to the next step(so we are dividing the cluster into different annulis by going further and increasing the width of each annuli) and going on.
    while i<r_f:
    
        in_ann=np.nonzero((R*pixsize>i) & (R*pixsize<(i+step_size)))
        av=np.mean(data[in_ann])
        
        ys[t] = av
        rs[t] = i+step_size/2
        
        t+=1
        i+=step_size
    
    return rs, ys, step_size, r_i, r_f, size 
     
    
