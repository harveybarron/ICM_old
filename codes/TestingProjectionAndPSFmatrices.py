"""
This script tests the Projection and PSF matrices on a predefined beta model as the model for 3D y
It compares the resulting projected 2D y with the analytically expected 2D y through a line of sight
(LOS) integration. However, the LOS integration here has been done using scipy.integrate.quad

"""

from scipy.integrate import quad
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt

from Computing_ys_in_annuli import get_2Dys_in_annuli
#rs,ys,step_size = get_2Dys_in_annuli()

rs = np.arange(1.5,113.5,3)
step_size = 3

sigma = 10/(2*np.sqrt(2*np.log(2)))
    
def beta_model(r, rc, beta):
  return (1+(r/rc)**2)**(-beta)

def los_proj(z, rproj, rc, beta):
  return beta_model(np.sqrt(z**2+rproj**2), rc, beta)

# Set up a 3D beta model y-profile
beta=1.57
rc=350. # kpc
y0=1.
rstep=81.
r=np.arange(rs[0], rs[-1]+step_size, step_size)
y_3D=beta_model(rs, rc, beta)


# Integrate over line of sight.
# Cutting off at rmax=2000 initially to avoid complications with outer shells
# Integral is symmetric so go from 0 to zmax and double
y_proj=np.zeros_like(y_3D)
rmax=r[-1]
for i, ri in enumerate(rs):
  zmax=np.sqrt(rmax**2-ri**2)
  y_proj[i]=y0*quad(los_proj, 0, zmax, args=(ri, rc, beta))[0]*2
  

R_PSF = np.loadtxt("R_PSF.txt")
R_PROJ = np.loadtxt("R_PROJ.txt")
R_PSF_inv = np.loadtxt("R_PSF_inv.txt")
R_PROJ_inv = np.loadtxt("R_PROJ_inv.txt")   

"""  
R_PSF_inv = linalg.inv(R_PSF)
R_PROJ_inv = linalg.inv(R_PROJ)
"""

X = np.matmul(R_PSF, np.matmul(R_PROJ,y_3D))
Y = np.matmul(R_PROJ,y_3D)

plt.plot(rs,X,'r.',label='$y_{2D}=R_{PSF}.R_{PROJ}.y_{3D}$')
plt.plot(rs,Y,'b.',label='$y_{2D}=R_{PROJ}.y_{3D}$ ')
plt.plot(rs[:],y_proj[:],'g',label='$y_{2D}$ with LOS integration')
plt.legend(loc=3, prop={'size': 9})
#plt.savefig("MatrixTest_y2D.png", dpi = 400)
plt.show()

T = np.matmul(R_PROJ_inv,y_proj)
S = np.matmul(R_PROJ_inv,np.matmul(R_PSF_inv,y_proj))

#plt.plot(rs,S,'r.',label='$y_{3D}=R_{PROJ}^{-1}.R_{PSF}^{-1}.y_{2D}$ ')
plt.plot(rs,T,'b.',label='$y_{3D}=R_{PROJ}^{-1}.y_{2D}$ ')
plt.plot(rs[:],y_3D[:],'g',label='Original $y_{3D}$  model')
plt.legend(loc=3, prop={'size': 9})
#plt.savefig("MatrixTest_y3D.png", dpi = 400)
plt.show()
   



