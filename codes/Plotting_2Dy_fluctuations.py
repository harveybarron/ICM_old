"""
This script uses the bin-averaged 2D y values to calculate the 2D y fluctuations for each pixel
The y values are interpolated between the bin-averaged values 
and subtracted from the exact y value at every pixel
Since we also need to account for pixels between r=0 and r=(centre of first bin),
We use the beta model fit value of y at r=0 (i.e y0) and then carry out the interpolation
Finally, the y fluctuations and normalised y fluctuations are plotted
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import scipy.optimize as opt

pixsize = 1.7177432059
conv =  27.052 #kpc

from Computing_ys_in_annuli import get_2Dys_in_annuli
rs,ys,step_size = get_2Dys_in_annuli()

ys_new = np.zeros(len(ys)+1)
rs_new = np.zeros(len(rs)+1)
rs[0] = 0

for i in range (1,len(rs_new)):
    rs_new[i] = rs[i-1]
    ys_new[i] = ys[i-1]

f = fits.open('map2048_MILCA_Coma_20deg_G.fits')
data = f[1].data

x=np.arange(700) # where npix is the number of pixels in the x direction, ie NAXIS1 from the header
y=np.arange(700) # as above, NAXIS2=NAXIS1
maxval = np.unravel_index(np.argmax(data), data.shape) # get the coordinates of the maximum
    
def beta_model(x,a,b,c):
    return a/(1+(x/b)**2)**c

constants = opt.curve_fit(beta_model, rs[1:], ys[1:])    #optimize module is used to curve fit 

a_fit = constants[0][0]
ys[0]=a_fit 
    
image_length=121 #arcmin

print("\nlength of image in arcmin = ",image_length)

x_ind = np.nonzero(((x-maxval[0])*pixsize>=-(image_length/2)) & (((x-maxval[0])*pixsize<=(image_length/2))))
y_ind = np.nonzero(((y-maxval[1])*pixsize>=-(image_length/2)) & (((y-maxval[1])*pixsize<=(image_length/2))))

#arrays used to store y fluctuations and normalised y fluctuations
y_fluc = np.zeros((len(x_ind[0]),len(y_ind[0])))
normalised_y_fluc = np.zeros((len(x_ind[0]),len(y_ind[0])))

t1,t2 = 0,0
for i in y_ind[0]:
    for j in x_ind[0]:
        radius = np.sqrt((i-maxval[0])**2 + (j-maxval[1])**2)*pixsize
        y_radius = np.interp(radius, rs, ys)
        y_fluc[t1][t2] = data[i][j] - y_radius
        normalised_y_fluc[t1][t2] = (data[i][j] - y_radius)/y_radius

        t2 += 1
    t2 = 0
    t1 += 1
 
from matplotlib.colors import TwoSlopeNorm     #Needed in plotting diverging colormaps

x_ticks = ['-1', '-0.75','-0.5','-0.25','0','0.25','0.5', '0.75','1']
y_ticks = ['-1', '-0.75','-0.5','-0.25','0','0.25','0.5', '0.75','1']
t11 = [0,8.75,17.5,26.25,35,43.75,52.5,61.25,69]

plt.xticks(ticks=t11, labels=x_ticks, size='small')
plt.yticks(ticks=t11, labels=y_ticks, size='small')

norm = TwoSlopeNorm(vmin=y_fluc.min(), vcenter=0, vmax=y_fluc.max())
pc = plt.pcolormesh(y_fluc, norm=norm, cmap="seismic")     
plt.imshow(y_fluc, cmap = 'seismic')
plt.colorbar(pc)
plt.xlabel("degrees")
plt.ylabel("degrees")
plt.title("y fluctuations in MILCA")
plt.savefig("figure 3.png", dpi = 1200)
plt.show()

plt.xticks(ticks=t11, labels=x_ticks, size='small')
plt.yticks(ticks=t11, labels=y_ticks, size='small')
 
norm = TwoSlopeNorm(vmin=normalised_y_fluc.min(), vcenter=0, vmax=normalised_y_fluc.max())
pc = plt.pcolormesh( normalised_y_fluc, norm=norm, cmap="seismic") 
plt.imshow(normalised_y_fluc, cmap = 'seismic')
plt.colorbar(pc)
plt.xlabel("degrees")
plt.ylabel("degrees")
plt.title("normalised y fluctuations in MILCA")
plt.savefig("figure 4.png", dpi = 1200)
plt.show()