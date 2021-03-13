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


plt.imshow(data, cmap = 'RdBu')
plt.colorbar()
plt.show()

print("\nMean:",np.mean(data))

histogram = plt.hist(data.flat, bins=500)