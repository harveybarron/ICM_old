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

def gaussian(sigma,mu,a,b):
 
    k = 1 / (sigma * np.sqrt(2*np.pi))
    s = -1.0 / (2 * sigma * sigma)
    def f(x):
        return k * np.exp(s * (x - mu)*(x - mu))

    return (quad(f, a, b)[0])


sigma = 10/(2*np.sqrt(2*np.log(2)))


from ys_annuli import y_2d

rs,ys,step_size,r_i,r_f,size = y_2d()

length = len(ys)

R_PROJ = np.zeros((length,length))
R_PSF = np.zeros((length,length))

for i in range (length):
    for j in range (length):

        R_PSF[i][j] = gaussian(sigma,rs[j],(rs[i]-step_size/2.),(rs[i]+step_size/2.))
        if j >= i:
            R_PROJ[i][j] = 4/3*((rs[i]+step_size/2.)**2-(rs[i]-step_size/2.)**2)
        elif i<j:
            R_PROJ[i][j] = 0

print("\n")
ys_transpose = ys.reshape(-1,1)
print(ys_transpose)
print("\n")
R_PSF_inv = linalg.inv(R_PSF)
print(R_PSF)
print(R_PSF_inv)
print("\n")
R_PROJ_inv = linalg.inv(R_PROJ)
print(R_PROJ)
print(R_PROJ_inv)
print("\n")

ys_3d_trans = np.matmul(R_PROJ_inv, np.matmul(R_PSF_inv,ys_transpose))

ys_3d = (ys_3d_trans.reshape(1,-1))[0]

print(ys_3d_trans)

pixsize = 1.7177432059
    
conv =  27.052 #kpc
    
#beta model function defined
def f(x,a,b,c):
    return a/(1+(x/b)**2)**c

constants = opt.curve_fit(f, rs, ys_3d)    #optimize module is used to curve fit 
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
ys_abs = np.abs(ys_3d)
ys_log = np.log10(ys_abs)

beta_fit_log = np.log10(beta_fit)

#x and y labels are set accordingly
x_ticks = ['100', '1000']
y_ticks = ['10','1','0.1','0.01']


t11 = [np.log10(100/conv),np.log10(1000/conv)]
t12 = [-5,-6,-7,-8]

plt.xticks(ticks=t11, labels=x_ticks, size='small')
plt.yticks(ticks=t12, labels=y_ticks, size='small')

plt.plot(rs_log, ys_log,'o' ,label = 'Milca y-profile')
plt.xlabel("Distance from centre of cluster (kpc)" ,fontsize=11)
plt.ylabel("Average 3D y profile (10^-6)" ,fontsize=11)
plt.title("Avg 3D y profile in MILCA map", fontsize=13)
plt.legend()
plt.show()















