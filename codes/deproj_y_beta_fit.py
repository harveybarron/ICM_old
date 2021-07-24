import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.integrate import quad,dblquad

from Computing_ys_in_annuli import get_2Dys_in_annuli
rs, ys, step_size, maxval = get_2Dys_in_annuli([352.42445296,349.85768166,1,0])

pixsize = 1.7177432059    
conv =  27.052 #kpc
arcmin2kpc=81./3
Planck_res=10.*arcmin2kpc
Planck_sig=Planck_res/2./np.sqrt(2*np.log(2.))

rmid = rs
ys = ys
rstep = step_size

Rproj=np.zeros((len(rmid), len(rmid)))
for i, ri in enumerate(rmid):
  for j, rj in enumerate(rmid[i:]):
    rim1=ri-rstep/2.
    rjm1=rj-rstep/2.
    rii=ri+rstep/2.
    rjj=rj+rstep/2.
    Vint1=rjj**2-rim1**2
    Vint2=rjj**2-rii**2
    Vint3=max(0,rjm1**2-rii**2) # this one can sometimes come out negative
    Vint4=rjm1**2-rim1**2
    # Sanity check
    if (Vint1<0) or (Vint2<0) or (Vint3<0) or (Vint4<0):
      print(i,j,Vint1,Vint2,Vint3,Vint4)
    Vint=4./3*(Vint1**(3./2) - Vint2**(3./2) + Vint3**(3./2) - Vint4**(3./2))
    if Vint==0:
      print(i,j,Vint1,Vint2,Vint3,Vint4)
    Rproj[i,j+i]=Vint/(rii**2-rim1**2)
    

# def gaussian(y, x, x0, y0, sig):                                      
#     return np.exp(-((x-x0)**2+(y-y0)**2)/2/sig**2)  

# def gauss_in_annulus(y0, sig, r1, r2):
#   # Break into pieces to enable integration within circular boundaries
#   int1=dblquad(gaussian, -r1, r1, lambda x: np.sqrt(r1**2-x**2), lambda x: np.sqrt(r2**2-x**2), args=(0, y0, sig))[0]
#   int2=dblquad(gaussian, -r1, r1, lambda x: -np.sqrt(r2**2-x**2), lambda x: -np.sqrt(r1**2-x**2), args=(0, y0, sig))[0]
#   int3=dblquad(gaussian, -r2, -r1, lambda x: 0, lambda x: np.sqrt(r2**2-x**2), args=(0, y0, sig))[0]
#   int4=dblquad(gaussian, -r2, -r1, lambda x: -np.sqrt(r2**2-x**2), lambda x: 0, args=(0, y0, sig))[0]
#   return (int1+int2+int3*2+int4*2)/(2*np.pi*sig**2)

# def av_gauss_in_annulus_integrand(r, sig, r1, r2):
#   return gauss_in_annulus(r, sig, r1, r2)*2*np.pi*r

# # Annulus integrand works in all cases incl innermost circle
# Rpsf=np.zeros((len(rmid), len(rmid)))
# for i, ri in enumerate(rmid):
#   print(".",end="")
#   rim1=ri-rstep/2.
#   rii=ri+rstep/2.
#   for j, rj in enumerate(rmid):
#     rjm1=rj-rstep/2.
#     rjj=rj+rstep/2.
#     # Amount of flux from j'th annulus thrown into i'th annulus
#     Rpsf[i,j]=quad(av_gauss_in_annulus_integrand, rjm1, rjj, args=(Planck_sig, rim1, rii))[0]/(np.pi*(rii**2-rim1**2))

# Rpsf_inv=linalg.inv(Rpsf)

Rproj_inv = linalg.inv(Rproj)
y_deproj=np.squeeze(np.matmul(Rproj_inv, ys.reshape(-1,1)))

def beta_model(x,a,b,c):
    return a/(1+(x/b)**2)**c

idx = np.where (rs<(1800/conv))
rs_fit = rs[idx]
ys_fit = y_deproj[idx]

constants = opt.curve_fit(beta_model, rs_fit, ys_fit,p0=[6e-5,400/conv,1.05])[0]
a_fit = constants[0]
b_fit = constants[1]
c_fit = constants[2]

beta_fit = np.zeros_like(ys)
for i,ri in enumerate (rmid):
    beta_fit[i] = beta_model(ri, a_fit, b_fit, c_fit )
    
print("\n")    
print("p0 = "+str(a_fit)+"  r_c = " +str(conv*b_fit) + "  beta* = " +str(c_fit)+ "\n")

plt.plot(rmid*conv, y_deproj, '.', label='deprojected with Rproj^-1')
plt.plot(rmid*conv, beta_fit, label = 'Beta-model fit with beta = %1.3f'%c_fit)
#plt.plot(rmid,ys,label='2D ys')
plt.legend(loc='best')
plt.show()








