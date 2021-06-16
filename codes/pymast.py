# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 12:17:16 2021
@author: Chandraniva

"""

import numpy as np
import pymaster as nmt
import matplotlib.pyplot as plt

pixsize = 1.7177432059
conv =  27.052 #kpc

y_fluc = np.loadtxt('y_fluc.txt')

Lx = 4. * np.pi/180
Ly = 4. * np.pi/180
#  - Nx and Ny: the number of pixels in the x and y dimensions
Nx, Ny = len(y_fluc),  len(y_fluc)

# Masks:
# Let's now create a mask
mask = np.zeros((Nx,Ny))
cen_x, cen_y = Nx/2., Ny/2.
cr = 60
I,J=np.meshgrid(np.arange(mask.shape[0]),np.arange(mask.shape[1]))
dist=np.sqrt((I-cen_x)**2+(J-cen_y)**2)
dist = dist * pixsize
idx = np.where(dist<=cr)
theta_ap = 15 
mask[idx]=1-np.exp(-9*(dist[idx]-cr)**2/(2*theta_ap**2))


# You can also apodize it in the same way you do for full-sky masks:
#mask = nmt.mask_apodization_flat(mask, Lx, Ly, aposize=0.05, apotype="Smooth")

plt.figure()
x_ticks = ['-2', '-1','0','1','2']
y_ticks = ['-2', '-1','0','1','2']
t11 = [0,35,70,105,138]
plt.xticks(ticks=t11, labels=x_ticks, size='small')
plt.yticks(ticks=t11, labels=y_ticks, size='small')
plt.imshow(mask, interpolation='nearest', origin='lower')
plt.colorbar()
plt.savefig("Apodized mask.png",dpi = 400)
plt.show()

# Fields:
# Once you have maps it's time to create pymaster fields.
# Note that, as in the full-sky case, you can also pass
# contaminant templates and flags for E and B purification
# (see the documentation for more details)
f0 = nmt.NmtFieldFlat(Lx, Ly, mask, [y_fluc])
#f2 = nmt.NmtFieldFlat(Lx, Ly, mask, [mpq, mpu], purify_b=True)

# Bins:
# For flat-sky fields, bandpowers are simply defined as intervals in ell, and
# pymaster doesn't currently support any weighting scheme within each interval.
bin_num = 5
l0_bins = np.arange(Nx/bin_num) * bin_num * np.pi/Lx
lf_bins = (np.arange(Nx/bin_num)+1) * bin_num * np.pi/Lx

b = nmt.NmtBinFlat(l0_bins, lf_bins)
# The effective sampling rate for these bandpowers can be obtained calling:
ells_uncoupled = b.get_effective_ells()

# Workspaces:
# As in the full-sky case, the computation of the coupling matrix and of
# the pseudo-CL estimator is mediated by a WorkspaceFlat case, initialized
# by calling its compute_coupling_matrix method:
w00 = nmt.NmtWorkspaceFlat()
w00.compute_coupling_matrix(f0, f0, b)
# Workspaces can be saved to and read from disk to avoid recomputing them:
w00.write_to("w00_flat.fits")
w00.read_from("w00_flat.fits")


# Computing power spectra:
# As in the full-sky case, you compute the pseudo-CL estimator by
# computing the coupled power spectra and then decoupling them by
# inverting the mode-coupling matrix. This is done in two steps below,
# but pymaster provides convenience routines to do this
# through a single function call
cl00_coupled = nmt.compute_coupled_cell_flat(f0, f0, b)
cl00_uncoupled = w00.decouple_cell(cl00_coupled)[0]

"""
cl02_coupled = nmt.compute_coupled_cell_flat(f0, f2, b)
cl02_uncoupled = w02.decouple_cell(cl02_coupled)
cl22_coupled = nmt.compute_coupled_cell_flat(f2, f2, b)
cl22_uncoupled = w22.decouple_cell(cl22_coupled)
"""
# Let's look at the results!
plt.figure()
"""
plt.plot(l, cl_tt, 'r-', label='Input TT')
plt.plot(l, cl_ee, 'g-', label='Input EE')
plt.plot(l, cl_bb, 'b-', label='Input BB')
"""
plt.plot(ells_uncoupled, cl00_uncoupled, 'r.', label='Uncoupled')
#plt.plot(ells_uncoupled, cl22_uncoupled[0], 'g--')
#plt.plot(ells_uncoupled, cl22_uncoupled[3], 'b--')
plt.loglog()
plt.xlabel("effective multipole")
plt.ylabel("Angular power spectrum")
plt.savefig("pymast_test.png", dpi = 400)
plt.show()

