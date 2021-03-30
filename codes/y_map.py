#y_map

#the script is to plot colormap of the y values coma cluster from MILCA map using Planck data.
#initially, we are trying to see the kind of data we have in the fits file

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

#reading the data
f = fits.open('map2048_MILCA_Coma_20deg_G.fits')

#displays info about the fits file
f.info()    
print('\n')
print(f[1])     
print('\n')
print(f[1].header)
print('\n')

# extracting the second element of the tuple as it is the actual data and hence we choose to take f[1].
data = f[1].data

#importing to plot colormaps
from matplotlib.colors import TwoSlopeNorm     

#x and y labels are set accordingly
x_ticks = ['-10', '-5', '0', '5', '10' ]        
y_ticks = ['10', '5', '0', '-5', '-10' ]
t11 = [0,175,350,525,699]

plt.xticks(ticks=t11, labels=x_ticks, size='small')
plt.yticks(ticks=t11, labels=y_ticks, size='small')

#the min, max and centre of the colormap is defined and plotted
norm = TwoSlopeNorm(vmin=data.min(), vcenter=0, vmax=data.max())
pc = plt.pcolormesh(data, norm=norm, cmap="seismic")    
plt.imshow(data, cmap = 'seismic')
plt.colorbar(pc)
plt.xlabel("degrees")
plt.ylabel("degrees")
plt.title("y map in MILCA")
plt.savefig("figure 1.svg", dpi = 1200)
plt.show()

#histogran for the data set is also plotted
histogram = plt.hist(data.flat, bins=500)
















