"""
This script calls the function "get_2Dys_in_annuli" from Computing_ys_in_annuli.py
It uses the beta model to find the best possible fit for 2D ys 
and plots both the 2D ys in annuli along with the beta model best fit curve
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

from Computing_ys_in_annuli import get_2Dys_in_annuli
rs,ys,step_size = get_2Dys_in_annuli()

pixsize = 1.7177432059    
conv =  27.052 #kpc

#beta model function defined
#params: a = y0, b = r_c, c = beta
def beta_model(x,a,b,c):
    return a/(1+(x/b)**2)**c

constants = opt.curve_fit(beta_model, rs, ys)
a_fit = constants[0][0]
b_fit = constants[0][1]
c_fit = constants[0][2]

beta_fit = np.zeros_like(ys)
for i,ri in enumerate (rs):
    beta_fit[i] = beta_model(ri, a_fit, b_fit, c_fit )
    
print("\n")    
print("a = "+str(a_fit)+"  b = " +str( b_fit) + "  c = " +str(c_fit)+ "\n")

rs_log = np.log10(rs)
ys_log = np.log10(ys)
beta_fit_log = np.log10(beta_fit)

x_ticks = ['100', '1000']
y_ticks = ['10','1','0.1']

t11 = [np.log10(100/conv),np.log10(1000/conv)]
t12 = [-5,-6,-7]

plt.xticks(ticks=t11, labels=x_ticks, size='small')
plt.yticks(ticks=t12, labels=y_ticks, size='small')

plt.plot(rs_log, ys_log,'o' ,label = 'Milca y-profile')
plt.plot( rs_log, beta_fit_log, label = 'Beta-model fit with beta = %1.3f'%c_fit)
plt.xlabel("Distance from centre of cluster (kpc)" ,fontsize=11)
plt.ylabel("Average y profile (10^-6)" ,fontsize=11)
plt.title("Avg y profile in MILCA map", fontsize=13)
plt.legend()
plt.savefig('Figure 2.png', dpi = 1200)
plt.show()