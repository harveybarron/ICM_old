# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 16:22:28 2021

@author: Chandraniva
"""

from scipy.integrate import quad
import numpy as np
from astropy.io import fits
from numpy import linalg
import scipy.optimize as opt
import matplotlib.pyplot as plt
import scipy.signal

def gaussian(sigma,mu,a,b):
 
    k = 1 / (sigma * np.sqrt(2*np.pi))
    s = -1.0 / (2 * sigma * sigma)
    def f(x):
        return k * np.exp(s * (x - mu)*(x - mu))

    return (quad(f, a, b)[0])


def gauss(x,sigma,mu):
    k = 1 / (sigma * np.sqrt(2*np.pi))
    s = -1.0 / (2 * sigma * sigma)
    return k * np.exp(s * (x - mu)*(x - mu))
    

sigma = 10/(2*np.sqrt(2*np.log(2)))


from ys_annuli import y_2d

rs,ys,step_size,r_i,r_f,size = y_2d()

length = len(ys)

R_PROJ = np.zeros((length,length))
R_PSF = np.zeros((length,length))

psf = np.zeros(size)

def V_int(a,b,c,d):
    return (4/3*((b*b-c*c)**(1.5)-(b*b-d*d)**(1.5)+(a*a-d*d)**(1.5)-(a*a-c*c)**(1.5)))

for i in range (length):
    for j in range (length):

        R_i1 = (rs[i]-step_size/2.)
        R_i = (rs[i]+step_size/2.)
        R_j1 = (rs[j]-step_size/2.)
        R_j = (rs[j]+step_size/2.)
        
        R_PSF[i][j] = gauss(rs[i],sigma,rs[j])
        psf[i] += gauss(rs[i],sigma,rs[j])
        
        if j == i:
            R_PROJ[i][j] = 4/3*np.sqrt((R_i)**2-(R_i1)**2)
        elif j>i:
            R_PROJ[i][j] = V_int(R_j1,R_j,R_i1,R_i)/((R_i)**2-(R_i1)**2) 
        elif j<i:
            R_PROJ[i][j] = 0
        

R_PSF_inv = linalg.inv(R_PSF)
R_PROJ_inv = linalg.inv(R_PROJ)

i=r_i
t=0


while i<r_f:
    R_i1 = (rs[t]-step_size/2.)
    R_i = (rs[t]+step_size/2.)
    #psf[t] = gauss(i+step_size/2.,sigma,0)/(R_i**2-R_i1**2)
    t+=1
    i+=step_size    
    
    
ys_deconv = np.real(scipy.signal.deconvolve(ys, psf)[1])

ys_3d = np.matmul(R_PROJ_inv, ys_deconv)

pixsize = 1.7177432059
    
conv =  27.052 #kpc
    
#beta model function defined
def f(x,a,b,c):
    return a/(1+(x/b)**2)**c

constants = opt.curve_fit(f, rs[:], ys_3d[:])    #optimize module is used to curve fit 
#optimized constants a,b and c are stored
a_fit = constants[0][0]
b_fit = constants[0][1]
c_fit = constants[0][2]

#arrays to store the beta model fucntion and it's log
beta_fit = np.zeros(size)

i = r_i
t=0

while i<r_f:
    beta_fit[t] = f(i, a_fit, b_fit, c_fit )
    t += 1
    i += step_size
    
print("\n")    
print("a = "+str(a_fit)+"  b = " +str( b_fit) + "  c = " +str(c_fit)+ "\n")


rs_log = np.log10(rs)
ys_log = np.log10(ys_3d)
beta_fit_log = np.log10(beta_fit)

#x and y labels are set accordingly
x_ticks = ['100', '1000']
y_ticks = ['10','1','0.1','0.01']


#t11 = [np.log10(100/conv),np.log10(1000/conv)]
t11 = [100/conv,1000/conv]
#t12 = [-5,-6,-7,-8]

plt.xticks(ticks=t11, labels=x_ticks, size='small')
#plt.yticks(ticks=t12, labels=y_ticks, size='small')

plt.plot(rs, ys_3d,'o' ,label = 'Milca 3D y-profile')
#plt.plot(rs_log, beta_fit_log, label = "Beta fit with beta = %1.3f"%c_fit)
plt.xlabel("Distance from centre of cluster (kpc)" ,fontsize=11)
plt.ylabel("Average 3D y profile (10^-6)" ,fontsize=11)
plt.title("Avg 3D y profile after deconvolution", fontsize=13)
plt.legend()
plt.savefig("3d_y with PSF (linear scale).svg", dpi = 1200)
plt.show()


plt.xticks(ticks=t11, labels=x_ticks, size='small')
#plt.yticks(ticks=t12, labels=y_ticks, size='small')

plt.plot(rs, ys,'o' ,label = 'Milca 2D y-profile')
plt.xlabel("Distance from centre of cluster (kpc)" ,fontsize=11)
plt.ylabel("Average 2D y profile (10^-6)" ,fontsize=11)
plt.title("Avg 2D y profile in MILCA map", fontsize=13)
plt.savefig("2d_y (linear scale).svg", dpi = 1200)
plt.legend()
plt.show()















