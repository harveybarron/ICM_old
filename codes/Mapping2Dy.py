"""
This script just extracts the y map data from fits file and plots the map
Warning: This script requires the fits files 'map2048_MILCA_Coma_20deg_G.fits' to work
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

f = fits.open('map2048_MILCA_Coma_20deg_G.fits')

#displays info about the fits file
f.info()    
print('\n')
print(f[1])     
print('\n')
print(f[1].header)
print('\n')

#the second element of the tuple is the actual data
data = f[1].data

#more info about the type of data
print(type(data))
print(data.shape)
print(data.dtype.name)
print("\nMin:",np.min(data))
print("\nMax:",np.max(data))
print("\nMean:",np.mean(data))
print("\nStd:",np.std(data))


from matplotlib.colors import TwoSlopeNorm     #Needed in plotting diverging colormaps

#x and y labels at approriate intervals are set
x_ticks = ['-10', '-5', '0', '5', '10' ]        
y_ticks = ['10', '5', '0', '-5', '-10' ]
t11 = [0,175,350,525,699]

plt.xticks(ticks=t11, labels=x_ticks, size='small')
plt.yticks(ticks=t11, labels=y_ticks, size='small')

#the min, max and centre of the colormap is defined and plotted
norm = TwoSlopeNorm(vmin=data.min(), vcenter=0, vmax=data.max())
pc = plt.pcolormesh(data, norm=norm, cmap="seismic")     #colormap used is seismic for better contrast
plt.imshow(data, cmap = 'seismic')
plt.colorbar(pc)
plt.xlabel("degrees")
plt.ylabel("degrees")
plt.title("y map in MILCA")
plt.savefig("figure 1.png", dpi = 1200)
plt.show()

















