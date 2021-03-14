# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 00:53:19 2021

@author: Chandraniva
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

f = fits.open('map2048_MILCA_Coma_20deg_G.fits')

f.info()
print('\n')
print(f[1])
print('\n')
print(f[1].header)
print('\n')

data = f[1].data

print(type(data))
print(data.shape)
print(data.dtype.name)
print("\nMin:",np.min(data))
print("\nMax:",np.max(data))
print("\nMean:",np.mean(data))
print("\nStd:",np.std(data))


from matplotlib.colors import TwoSlopeNorm

x_ticks = ['-10', '-5', '0', '5', '10' ]
y_ticks = ['10', '5', '0', '-5', '-10' ]
t11 = [0,175,350,525,699]

plt.xticks(ticks=t11, labels=x_ticks, size='small')
plt.yticks(ticks=t11, labels=y_ticks, size='small')

norm = TwoSlopeNorm(vmin=data.min(), vcenter=0, vmax=data.max())
pc = plt.pcolormesh(data, norm=norm, cmap="seismic")     
plt.imshow(data, cmap = 'seismic')
plt.colorbar(pc)
plt.xlabel("degrees")
plt.ylabel("degrees")
plt.title("y map in MILCA")
plt.savefig("figure 1.svg", dpi = 1200)
plt.show()


histogram = plt.hist(data.flat, bins=500)
















