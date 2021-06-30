import numpy as np
import pymaster as nmt
import matplotlib.pyplot as plt

# Dimensions:
# First, a flat-sky field is defined by four quantities:
#  - Nx and Ny: the number of pixels in the x and y dimensions
Nx = 602
Ny = 410
#  - Lx and Ly: the size of the patch in the x and y dimensions (in radians)
# Use square pixels
pix=7.176/60 # deg
Lx = Nx*pix * np.pi/180
Ly = Ny*pix * np.pi/180

l, cl_tt, cl_ee, cl_bb, cl_te = np.loadtxt('cls.txt', unpack=True)
mpt = np.loadtxt("conv_map_test.txt")



def beam(x):
    fwhm = 180*60/10  # in terms of l
    Planck_sig = fwhm/2.355
    return np.exp(-((x)**2)/(4.*Planck_sig**2))   #*(np.sqrt(2*np.pi)*sigma)

bin_number = 40

#min and max l's of cls.txt
l_min = 0
l_max = 767

bin_size = (l_max-l_min)/bin_number

l0_bins=[]
lf_bins=[]

for i in range (bin_number):
    l0_bins.append(l_min+bin_size*i)
    lf_bins.append(l_min+bin_size*(i+1))
    
    
b = nmt.NmtBinFlat(l0_bins, lf_bins)
ells_uncoupled = b.get_effective_ells()
print(ells_uncoupled)

f0 = nmt.NmtFieldFlat(Lx, Ly, np.ones_like(mpt), [mpt],
                      beam=[ells_uncoupled, beam(ells_uncoupled)])

w00 = nmt.NmtWorkspaceFlat()
w00.compute_coupling_matrix(f0, f0, b)
cl00_coupled = nmt.compute_coupled_cell_flat(f0, f0, b)
cl00_uncoupled = w00.decouple_cell(cl00_coupled)[0]


plt.figure()
plt.plot(l[:],cl_tt[:], 'r-', label='Original spectrum')
plt.plot(ells_uncoupled[:], cl00_uncoupled[:], 'b.', label='Recovered spectrum')
plt.legend()
plt.xlabel("effective multipole")
plt.ylabel("power spectrum")
plt.loglog()
plt.savefig("Recovered spectrum.png",dpi=400)
plt.show()


