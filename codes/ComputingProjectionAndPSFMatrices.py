"""
This script computes the Projection and PSF matrices and stores them in a text file for later use
Warning: May take some time to compute R_PSF. 
"""

from scipy.integrate import quad, dblquad
import numpy as np

from Computing_ys_in_annuli import get_2Dys_in_annuli
rs,ys,step_size = get_2Dys_in_annuli()

length = len(ys)
R_PROJ = np.zeros((length,length))
R_PSF = np.zeros((length,length))

def V_int(a,b,c,d):
    return (4/3*((b*b-c*c)**(1.5)-(b*b-d*d)**(1.5)+(a*a-d*d)**(1.5)-(a*a-c*c)**(1.5)))  

def gaussian(y, x, x0, y0, sig):                                      
    return np.exp(-((x-x0)**2+(y-y0)**2)/2/sig**2)  

def gauss_in_annulus(y0, sig, r1, r2):
  # Break into pieces to enable integration within circular boundaries
  int1=dblquad(gaussian, -r1, r1, lambda x: np.sqrt(r1**2-x**2), lambda x: np.sqrt(r2**2-x**2), args=(0, y0, sig))[0]
  int2=dblquad(gaussian, -r1, r1, lambda x: -np.sqrt(r2**2-x**2), lambda x: -np.sqrt(r1**2-x**2), args=(0, y0, sig))[0]
  int3=dblquad(gaussian, -r2, -r1, lambda x: 0, lambda x: np.sqrt(r2**2-x**2), args=(0, y0, sig))[0]
  int4=dblquad(gaussian, -r2, -r1, lambda x: -np.sqrt(r2**2-x**2), lambda x: 0, args=(0, y0, sig))[0]
  return (int1+int2+int3*2+int4*2)/(2*np.pi*sig**2)

def av_gauss_in_annulus_integrand(r, sig, r1, r2):
  return gauss_in_annulus(r, sig, r1, r2)*2*np.pi*r

Planck_sig = 10/(2*np.sqrt(2*np.log(2)))

# Annulus integrand works in all cases incl innermost circle
for i, ri in enumerate(rs):
  rim1=ri-step_size/2.
  rii=ri+step_size/2.
  for j, rj in enumerate(rs):    
    rjm1=rj-step_size/2.
    rjj=rj+step_size/2.   
    if j == i:
        R_PROJ[i][j] = 4/3*np.sqrt((rii)**2-(rim1)**2)
    elif j>i:
        R_PROJ[i][j] = V_int(rjm1,rjj,rim1,rii)/((rii)**2-(rim1)**2) 
    elif j<i:
        R_PROJ[i][j] = 0
    # Amount of flux from j'th annulus thrown into i'th annulus
    R_PSF[i,j]=quad(av_gauss_in_annulus_integrand, rjm1, rjj, args=(Planck_sig, rim1, rii))[0]/(np.pi*(rii**2-rim1**2))


# matrices are stored in a text file for later use
f_PSF = open("R_PSF.txt","w+")
np.savetxt(f_PSF, R_PSF)
f_PSF.close()

f_PROJ = open("R_PROJ.txt","w+")
np.savetxt(f_PROJ, R_PROJ)
f_PROJ.close()









