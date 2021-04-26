# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 11:49:18 2021

@author: Chandraniva
"""

from scipy.integrate import quad
import numpy as np
from astropy.io import fits
from numpy import linalg
import scipy.optimize as opt
import matplotlib.pyplot as plt
import scipy.signal

from ys_annuli import y_2d

rs,ys,step_size,r_i,r_f,size = y_2d()
 
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
    
def beta_model(r, rc, beta):
  return (1+(r/rc)**2)**(-beta)

def los_proj(z, rproj, rc, beta):
  return beta_model(np.sqrt(z**2+rproj**2), rc, beta)

# Set up a 3D beta model y-profile
beta=1.07
rc=350. # kpc
y0=1.
rstep=81.
r=np.arange(r_i, r_f+step_size, step_size)
rmid=(r[1:]+r[:-1])/2.
y_3D=beta_model(rmid, rc, beta)

length = size
psf = np.zeros(length)

i=r_i
t=0

while i<r_f-step_size:
    R_i1 = (rs[t]-step_size/2.)
    R_i = (rs[t]+step_size/2.)
    psf[t] = gauss(i+step_size/2.,sigma,0)#/(R_i**2-R_i1**2)
    t+=1
    i+=step_size
    

# Integrate over line of sight.
# Think this is integrable analytically but I am lazy - you could figure out the integral if you like!
# Cutting off at rmax=2000 initially to avoid complications with outer shells
# Integral is symmetric so go from 0 to zmax and double
y_proj=np.zeros_like(y_3D)
rmax=r[-1]
for i, ri in enumerate(rmid):
  zmax=np.sqrt(rmax**2-ri**2)
  y_proj[i]=y0*quad(los_proj, 0, zmax, args=(ri, rc, beta))[0]*2

def sym(a):
    l=len(a)
    b = np.zeros(2*l)
    for i in range (2*l):
        if i<l:
            b[i] = a[l-i-1]
        elif i>=l:
            b[i] = a[i-l]
    return b
    

y_proj_conv=scipy.signal.convolve(sym(y_proj),sym(psf),'same')

R_PROJ = np.zeros((length,length))
R_PSF = np.zeros((length,length))


def V_int(a,b,c,d):
    return (4/3*((b*b-c*c)**(1.5)-(b*b-d*d)**(1.5)+(a*a-d*d)**(1.5)-(a*a-c*c)**(1.5)))

for i in range (length):
    for j in range (length):
        
        R_i1 = (rs[i]-step_size/2.)
        R_i = (rs[i]+step_size/2.)
        R_j1 = (rs[j]-step_size/2.)
        R_j = (rs[j]+step_size/2.)
        
        R_PSF[i][j] = gauss(rs[i],sigma,rs[j])
        #psf[i] += gauss(rs[i],sigma,rs[j])
        
        if j == i:
            R_PROJ[i][j] = 4/3*np.sqrt((R_i)**2-(R_i1)**2)
        elif j>i:
            R_PROJ[i][j] = V_int(R_j1,R_j,R_i1,R_i)/((R_i)**2-(R_i1)**2) 
        elif j<i:
            R_PROJ[i][j] = 0

        
R_PSF_inv = linalg.inv(R_PSF)
R_PROJ_inv = linalg.inv(R_PROJ)
X = np.matmul(R_PSF, np.matmul(R_PROJ,y_3D))
Y = np.matmul(R_PROJ,y_3D)


plt.plot(rmid,X,'r.',label='with matrix (convoluted)')
plt.plot(rmid,y_proj_conv[length:], label='with LOS integration (convoluted)')
plt.plot(rmid,Y,'b.',label='with matrix (no convolution)')
plt.plot(rmid,y_proj,label='with LOS integration (no convolution)')
plt.legend(loc=1, prop={'size': 7})
plt.savefig("Matrix testing with beta.png", dpi = 400)
plt.show()

    
