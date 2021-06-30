# -*- coding: utf-8 -*-
"""
Created on Wed May 26 20:15:29 2021

@author: Chandraniva
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sklearn.datasets as dt
from astropy.io import fits
from Computing_ys_in_annuli import get_2Dys_in_annuli

pixsize = 1.7177432059
conv =  27.052 #kpc
f = fits.open('map2048_MILCA_Coma_20deg_G.fits')
data = f[1].data

def gradient_descent(max_iterations,threshold,w_init,
                     obj_func,learning_rate):
    
    w = w_init
    print(grad(w))
    w_history = w
    f_history = obj_func(w)
    delta_w = [0,0,0,0]
    i = 0
    diff = 1.0e10
    while  i<max_iterations and diff>threshold:
        delta_w[0] = -learning_rate*grad(w)[0] 
        delta_w[1] = -learning_rate*grad(w)[1] 
        delta_w[2] = -0.01*learning_rate*grad(w)[2] 
        delta_w[3] = -0.01*learning_rate*grad(w)[3] 

        print(delta_w)
        w = [w[i]+delta_w[i] for i in range(4)]
        print(w)
        
        # store the history of w and f
        w_history = np.vstack((w_history,w))
        f_history = np.vstack((f_history,obj_func(w)))
        
        # update iteration number and diff between successive values
        # of objective function
        i+=1
        diff = np.absolute(f_history[-1]-f_history[-2])
    
    return w_history,f_history

def grad(w):
    eps = 1e-3
    return [(fun([w[0]+eps,w[1],w[2],w[3]])-fun([w[0]-eps,w[1],w[2],w[3]]))/(2*eps), \
            (fun([w[0],w[1]+eps,w[2],w[3]])-fun([w[0],w[1]-eps,w[2],w[3]]))/(2*eps), \
            (fun([w[0],w[1],w[2]+eps,w[3]])-fun([w[0],w[1],w[2]-eps,w[3]]))/(2*eps), \
            (fun([w[0],w[1],w[2],w[3]+eps])-fun([w[0],w[1],w[2],w[3]-eps]))/(2*eps)] 
        


def fun(w):    

    rs,ys,step_size,maxval = get_2Dys_in_annuli(maxval=w)
    
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
    dipole_metric = sum(abs(y_centre[i]) for i in idx_neg) +\
                    sum(abs(y_centre[i]) for i in idx_pos)
    return dipole_metric



w_init = [352.32799422, 350.37014835, 0.89435012, -0.64212693]
w_history,f_history = gradient_descent(50,1e-7,w_init,fun,1e2)

print(w_history)
print(f_history)

































