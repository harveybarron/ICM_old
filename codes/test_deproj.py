import numpy as np
from matplotlib import pyplot
from scipy.integrate import quad
from numpy import linalg
from scipy.signal import convolve2d

def beta_model(r, rc, beta):
  return (1+(r/rc)**2)**(-beta)

def los_proj(z, rproj, rc, beta):
  return beta_model(np.sqrt(z**2+rproj**2), rc, beta)

# Set up a 3D beta model y-profile
beta=1.07
rc=350. # kpc
y0=1.
rstep=81.
r=np.arange(0, 2081, rstep)
rmid=(r[1:]+r[:-1])/2.
y_3D=beta_model(rmid, rc, beta)

# Integrate over line of sight
# Think this is integrable analytically but I am lazy
# Cutting off at rmax=2000 initially to avoid complications with outer shells
# Integral is symmetric so go from 0 to zmax and double
y_proj=np.zeros_like(y_3D)
rmax=r[-1]
for i, ri in enumerate(rmid):
  zmax=np.sqrt(rmax**2-ri**2)
  y_proj[i]=y0*quad(los_proj, 0, zmax, args=(ri, rc, beta))[0]*2


# My version
# First check it works forwards
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

y_proj2=np.matmul(Rproj, y_3D.reshape(-1,1))
pyplot.loglog(rmid, y_3D, '.-', label='3D profile')
pyplot.loglog(rmid, y_proj, '.-', label='projected')
pyplot.loglog(rmid, y_proj2, '.-', label='projected with Rproj')

# Now check it works backwards
Rproj_inv = linalg.inv(Rproj)
y_deproj=np.matmul(Rproj_inv, y_proj.reshape(-1,1))
pyplot.loglog(rmid, y_deproj, '.-', label='deprojected with Rproj^-1')
pyplot.legend(loc='best')
pyplot.show()

# Plot ratio
y_deproj=np.squeeze(y_deproj)
pyplot.plot(rmid, y_deproj/y_3D, '.')
pyplot.axhline(y=1.)
pyplot.title('Deprojected / 3D')
pyplot.show()

# Project onto 2D map
pix=1.7/60 # deg
maphw=2. # deg
arcmin2kpc=81./3

npix=np.round(maphw/pix)
x=np.arange(-npix,npix+1)
y=np.arange(-npix,npix+1)
X, Y=np.meshgrid(x,y)
R=np.sqrt(X**2+Y**2)
Rphys=R*pix*60*arcmin2kpc

myord=np.argsort(R.flatten())
Y=np.zeros_like(R.flatten())
Y[myord]=np.interp(Rphys.flatten()[myord], rmid, y_proj, right=0.)
Y=np.reshape(Y, R.shape)

Planck_res=10.*arcmin2kpc
Planck_sig=Planck_res/2./np.sqrt(2*np.log(2.))
PSF=np.exp(-Rphys**2/2/Planck_sig**2)
Y_conv=convolve2d(Y, PSF, mode='same')/np.sum(PSF)

# Analytically
def gaussian(y, x, x0, y0, sig):                                      
    return np.exp(-((x-x0)**2+(y-y0)**2)/2/sig**2)   

from scipy.integrate import dblquad

#def gauss_in_circle(y0, sig, radius):
#  integral=dblquad(gaussian, -radius, radius, lambda x: -np.sqrt(radius**2-x**2), lambda x: np.sqrt(radius**2-x**2), args=(0, y0, sig))[0]
#  return integral/(2*np.pi*sig**2)

#def av_gauss_in_circle_integrand(r, sig, radius):
#  return gauss_in_circle(r, sig, radius)*2*np.pi*r

def gauss_in_annulus(y0, sig, r1, r2):
  # Break into pieces to enable integration within circular boundaries
  int1=dblquad(gaussian, -r1, r1, lambda x: np.sqrt(r1**2-x**2), lambda x: np.sqrt(r2**2-x**2), args=(0, y0, sig))[0]
  int2=dblquad(gaussian, -r1, r1, lambda x: -np.sqrt(r2**2-x**2), lambda x: -np.sqrt(r1**2-x**2), args=(0, y0, sig))[0]
  int3=dblquad(gaussian, -r2, -r1, lambda x: 0, lambda x: np.sqrt(r2**2-x**2), args=(0, y0, sig))[0]
  int4=dblquad(gaussian, -r2, -r1, lambda x: -np.sqrt(r2**2-x**2), lambda x: 0, args=(0, y0, sig))[0]
  return (int1+int2+int3*2+int4*2)/(2*np.pi*sig**2)

def av_gauss_in_annulus_integrand(r, sig, r1, r2):
  return gauss_in_annulus(r, sig, r1, r2)*2*np.pi*r

# Annulus integrand works in all cases incl innermost circle
Rpsf=np.zeros((len(rmid), len(rmid)))
for i, ri in enumerate(rmid):
  rim1=ri-rstep/2.
  rii=ri+rstep/2.
  for j, rj in enumerate(rmid):
    rjm1=rj-rstep/2.
    rjj=rj+rstep/2.
    # Amount of flux from j'th annulus thrown into i'th annulus
    Rpsf[i,j]=quad(av_gauss_in_annulus_integrand, rjm1, rjj, args=(Planck_sig, rim1, rii))[0]/(np.pi*(rii**2-rim1**2))

pyplot.plot(Rphys.flatten(), Y.flatten(), '.', label='2D')
pyplot.plot(Rphys.flatten(), Y_conv.flatten(), '.', label='psf-convolved')
pyplot.plot(rmid, np.matmul(Rpsf, y_proj), '.', label='Rpsf')
pyplot.legend(loc='best')
pyplot.show()

# Plot deconvolution process
Rpsf_inv=linalg.inv(Rpsf)
Rproj_inv=linalg.inv(Rproj)

y_proj2_conv=np.squeeze(np.matmul(Rpsf, y_proj2))
y_deconv=np.squeeze(np.matmul(Rproj_inv, np.matmul(Rpsf_inv, y_proj2_conv)))
pyplot.plot(rmid, y_3D, '.', label='3D y')
pyplot.plot(rmid, y_deconv, '.', label='Rproj^-1 Rpsf^-1 y_2D')
pyplot.legend(loc='best')
pyplot.show()

# Plot ratio
pyplot.plot(rmid, y_deconv/y_3D, '.')
pyplot.title('Rproj^-1 Rpsf^-1 y_2D / y_3D')
pyplot.show()

# One final test - what we would actually do, ie measure average y in annuli on the map
Y_from_map=np.zeros_like(rmid)
for i, ri in enumerate(rmid):
  I=np.nonzero((Rphys>ri-rstep/2.) & (Rphys<=ri+rstep/2.))
  Y_from_map[i]=np.mean(Y_conv[I])

pyplot.plot(rmid, y_proj2_conv, '.', label='Rpsf Rproj y_3D')
pyplot.plot(rmid, Y_from_map, '.', label='Av. y in annuli from map')
pyplot.legend(loc='best')
pyplot.show()

y_deconv2=np.squeeze(np.matmul(Rproj_inv, np.matmul(Rpsf_inv, Y_from_map)))
pyplot.plot(rmid, y_deconv/y_3D, '.', label='[Rproj^-1 Rpsf^-1 (Rpsf Rproj y_3D)] / y_3D')
pyplot.plot(rmid, y_deconv2/y_3D, '.', label='[Rproj^-1 Rpsf^-1 (Av. y from map)] / y_3D')
pyplot.legend(loc='best')
pyplot.show()

# Which part is breaking down?
pyplot.plot(rmid, np.squeeze(np.matmul(Rpsf_inv, y_proj2_conv)), '.', label='Rpsf^-1 (Rpsf Rproj y_3D)')
pyplot.plot(rmid, np.squeeze(np.matmul(Rpsf_inv, Y_from_map)), '.', label='Rpsf^-1 (Av. y from map)')
pyplot.legend(loc='best')
pyplot.show()

pyplot.plot(rmid, np.squeeze(np.matmul(Rproj_inv, y_proj2_conv)), '.', label='Rproj^-1 (Rpsf Rproj y_3D)')
pyplot.plot(rmid, np.squeeze(np.matmul(Rproj_inv, Y_from_map)), '.', label='Rproj^-1 (Av. y from map)')
pyplot.legend(loc='best')
pyplot.show()

# Yep, definitely the inverse PSF matrix that's the problem.  Let's try cutting off some of the smaller elements
cut_frac=[0.01, 0.05, 0.1, 0.2, 0.5]
from copy import deepcopy
for i, cf in enumerate(cut_frac):
  Rpsf2=deepcopy(Rpsf)
  Rpsf2[Rpsf2<cf*np.max(Rpsf2)]=0.
  Rpsf2_inv=linalg.inv(Rpsf2)
  pyplot.plot(rmid, np.squeeze(np.matmul(Rpsf2_inv, Y_from_map))/np.squeeze(np.matmul(Rpsf_inv, y_proj2_conv)), '.', label='Rpsf^-1 (Av. y from map), cutoff fraction='+str(cf))

pyplot.legend(loc='best')
pyplot.show()

# Definitely getting better but not converging to an acceptable level... park for now.

