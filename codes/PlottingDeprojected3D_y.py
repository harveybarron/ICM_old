"""
This script loads the projection and PSF matrices (R_PROJ.txt and R_PSF.txt) 
and calculates the deprojected 3D y's using:
1. deconvolution followed by deprojection.
2. Only deprojection (Along with beta fit)
Both of these 3D y profiles are then plotted and saved
Warning: You must have the files "R_PROJ.txt" and "R_PSF.txt" for it to work
"""

import numpy as np
from numpy import linalg
import scipy.optimize as opt
import matplotlib.pyplot as plt

from Computing_ys_in_annuli import get_2Dys_in_annuli
rs,ys,step_size = get_2Dys_in_annuli()

R_PSF = np.loadtxt("R_PSF.txt")
R_PROJ = np.loadtxt("R_PROJ.txt")

#inverses of matrices computed    
R_PSF_inv = linalg.inv(R_PSF)
R_PROJ_inv = linalg.inv(R_PROJ)

ys_3d_deconv = np.abs(np.matmul(R_PROJ_inv, np.matmul(R_PSF_inv,ys)))
ys_3d = np.abs(np.matmul(R_PROJ_inv,ys))

factor = ys_3d_deconv[-3]/ys_3d[-3]
ys_3d_deconv = ys_3d_deconv/factor

pixsize = 1.7177432059    
conv =  27.052 #kpc

#beta model function 
#parameters: a = y0 , b = y_c, c = beta
def beta_model(x,a,b,c):
    return a/(1+(x/b)**2)**c

constants = opt.curve_fit(beta_model, rs[:], ys_3d[:])
a_fit = constants[0][0]
b_fit = constants[0][1]
c_fit = constants[0][2]

beta_fit = np.zeros(len(ys))

for i, ri in enumerate(rs):
    beta_fit[i] = beta_model(ri, a_fit, b_fit, c_fit )

    
print("\nFitting parameters:\n")    
print("a = "+str(a_fit)+"  b = " +str( b_fit) + "  c = " +str(c_fit)+ "\n")

x_ticks = ['200', '1000']
y_ticks = ['10','1','0.1','0.01']

t11 = [np.log10(200/conv),np.log10(1000/conv)]
t12 = [-5,-6,-7,-8]

plt.xticks(ticks=t11, labels=x_ticks, size='small')
plt.yticks(ticks=t12, labels=y_ticks, size='small')

plt.plot(np.log10(rs[:]),np.log10(ys_3d_deconv[:]),'o' ,label = 'Milca 3D y-profile with deconvolution')
plt.xlabel("Distance from centre of cluster (kpc)" ,fontsize=11)
plt.ylabel("Average 3D y profile ($10^{-6}$)" ,fontsize=11)
plt.title("3D y profile after deconvolution and deprojection", fontsize=13)
plt.legend()
plt.savefig("3D y with PSF.png", dpi = 1200)
plt.show()

plt.xticks(ticks=t11, labels=x_ticks, size='small')
plt.yticks(ticks=t12, labels=y_ticks, size='small')

plt.plot(np.log10(rs[:]),np.log10(ys_3d[:]),'o' ,label = 'Milca 3D y-profile without deconvolution')
plt.plot(np.log10(rs[:]), np.log10(beta_fit[:]), label = "Beta fit with beta = %1.3f"%c_fit)
plt.xlabel("Distance from centre of cluster (kpc)" ,fontsize=11)
plt.ylabel("Average 3D y profile ($10^{-6}$)" ,fontsize=11)
plt.title("3D y profile after only deprojection", fontsize=13)
plt.legend()
plt.savefig("3D y without PSF.png", dpi = 1200)
plt.show()













