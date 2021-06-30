import numpy as np
import pymaster as nmt
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

# Dimensions:
# First, a flat-sky field is defined by four quantities:
#  - Nx and Ny: the number of pixels in the x and y dimensions
Nx = 602
Ny = 410
#  - Lx and Ly: the size of the patch in the x and y dimensions (in radians)
# Use square pixels

#I CHANGED THE PIXSIZE TO WHAT WAS USED IN EXAMPLE 7 OF PYMASTER
pix=7.176/60 # deg
Lx = Nx*pix * np.pi/180
Ly = Ny*pix * np.pi/180

# Gaussian simulations:
# pymaster allows you to generate random realizations of both spherical and
# flat fields given a power spectrum. These are returned as 2D arrays with
# shape (Ny,Nx)
# This power spectrum is inside the test folder in the NaMaster github repository, but you could use anything here
l, cl_tt, cl_ee, cl_bb, cl_te = np.loadtxt('cls.txt', unpack=True)

"""
beam = np.exp(-(0.25 * np.pi/180 * l)**2)
cl_tt *= beam
cl_ee *= beam
cl_bb *= beam
cl_te *= beam
"""

# Generate random realisation of the power spectrum, just in temperature (spin-0).  Set random seed for repeatability
rseed=10
mpt, = nmt.synfast_flat(Nx, Ny, Lx, Ly, np.expand_dims(cl_tt, axis=0) , [0]) #, seed=rseed)


"""
mpt, mpq, mpu = nmt.synfast_flat(Nx, Ny, Lx, Ly,
                                 np.array([cl_tt, cl_te, 0 * cl_tt,
                                           cl_ee, 0 * cl_ee, cl_bb]),
                                 [0, 2])
"""

# Convolve with instrumental beam
x=np.arange(Nx)-Nx/2-1
y=np.arange(Ny)-Ny/2-1
X, Y=np.meshgrid(x, y)
R=np.sqrt(X**2+Y**2)*pix # deg
Planck_res=10./60 # deg
Planck_sig=Planck_res/2./np.sqrt(2*np.log(2.)) # deg
PSF=np.exp(-R**2/2/Planck_sig**2)
mpt_conv=convolve2d(mpt, PSF, mode='same')/np.sum(PSF)
np.savetxt("conv_map_test.txt",mpt_conv)
np.savetxt("map_test.txt",mpt)

# You can have a look at the maps using matplotlib's imshow:
plt.figure()
plt.imshow(mpt, interpolation='nearest', origin='lower')
plt.colorbar()
plt.savefig("map_test.png",dpi=400)
plt.figure()
plt.imshow(mpt_conv, interpolation='nearest', origin='lower')
plt.colorbar()
plt.savefig("convolved_map_test.png",dpi=400)
plt.show()