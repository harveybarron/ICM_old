#binned_beta_it

#The script imports the function y_2d from ys_annuli script which calculates the average y profile and fits the data to beta model using optimize module of python.

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import scipy.optimize as opt

from ys_annuli import y_2d

pixsize = 1.7177432059
    
conv =  27.052 

rs,ys,step_size,r_i,r_f,size = y_2d()

#beta model function where c is the beta parameter
def f(x,a,b,c): 
    return a/(1+(x/b)**2)**c
    
#optimize module is used to curve fit and optimized constants are stored
constants = opt.curve_fit(f, rs, ys)
a_fit = constants[0][0]
b_fit = constants[0][1]
c_fit = constants[0][2]

#defining arrays to store beta model function and it's log
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
ys_log = np.log10(ys)
beta_fit_log = np.log10(beta_fit)

#x and y labels are set accordingly
x_ticks = ['100', '1000']
y_ticks = ['10','1','0.1']


t11 = [np.log10(100/conv),np.log10(1000/conv)]
t12 = [-5,-6,-7]

plt.xticks(ticks=t11, labels=x_ticks, size='small')
plt.yticks(ticks=t12, labels=y_ticks, size='small')

plt.plot(rs_log, ys_log,'o' ,label = 'Milca y-profile')
plt.plot( rs_log, beta_fit_log, label = 'Beta-model fit')
plt.xlabel("Distance from centre of cluster (kpc)" ,fontsize=11)
plt.ylabel("Average y profile (10^-6)" ,fontsize=11)
plt.title("Avg y profile in MILCA map", fontsize=13)
plt.legend()
plt.savefig('Figure 2 with beta fit.svg', dpi = 1200)
plt.show()
