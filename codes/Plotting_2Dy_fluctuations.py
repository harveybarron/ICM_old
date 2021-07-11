"""
This script uses the bin-averaged 2D y values to calculate the 2D y fluctuations for each pixel
The y values are interpolated between the bin-averaged values 
and subtracted from the exact y value at every pixel
Since we also need to extrapolate for pixels between r=0 and r=(centre of first bin),
We use the beta model fit value of y at r=0 (i.e y0) and then carry out the interpolation
Finally, the y fluctuations and normalised y fluctuations are plotted
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import scipy.optimize as opt
import matplotlib.patches as patches

pixsize = 1.7177432059
conv =  27.052 #kpc

from Computing_ys_in_annuli import get_2Dys_in_annuli

"""
For the original map, the optimal centre coordinates were:
[x,y,f,theta] = [352.42445296,349.85768166,1,0] 
The WCS of these coordinates were estimated

    
For the first and last maps used to calculate Cross power spectrum,
the centre coordinates corresponding to the above estimated WCS were:
[x,y,f,theta] =  [349.3,351.7,1,0]
    
"""

#rs,ys,step_size,maxval = get_2Dys_in_annuli([352.42445296,349.85768166,1,0])
rs,ys,step_size,maxval = get_2Dys_in_annuli([349.3,351.7,1,0])
#rs,ys,step_size,maxval = get_2Dys_in_annuli([352.42445296,349.85768166,0.89232604,-0.63664394])
#rs,ys,step_size,maxval = get_2Dys_in_annuli([352.4023856 , 350.23374859, 0.52490389, -0.51376102])
ys_new = np.zeros(len(ys)+1)
rs_new = np.zeros(len(rs)+1)

rs_new[1:] = rs
ys_new[1:] = ys

f = fits.open('map2048_MILCA_Coma_20deg_last.fits')
data = f[1].data

x=np.arange(700) # where npix is the number of pixels in the x direction, ie NAXIS1 from the header
y=np.arange(700) # as above, NAXIS2=NAXIS1

"""    
def beta_model(x,a,b,c):
    return a/(1+(x/b)**2)**c

constants = opt.curve_fit(beta_model, rs_new[1:], ys_new[1:])    #optimize module is used to curve fit 

a_fit = constants[0][0]
"""
ys_new[0]= ys[0]
 
image_length=120*2 #arcmin

print("\nlength of image in arcmin = ",image_length)

x_ind = np.nonzero(((x-maxval[0])*pixsize>=-(image_length/2)) & (((x-maxval[0])*pixsize<=(image_length/2))))
y_ind = np.nonzero(((y-maxval[1])*pixsize>=-(image_length/2)) & (((y-maxval[1])*pixsize<=(image_length/2))))

#arrays used to store y fluctuations and normalised y fluctuations
y_fluc = np.zeros((len(x_ind[0]),len(y_ind[0])))
normalised_y_fluc = np.zeros((len(x_ind[0]),len(y_ind[0])))

t1,t2 = 0,0
for i in x_ind[0]:
    for j in y_ind[0]:
        #radius = np.sqrt((i-maxval[0])**2 + (j-maxval[1])**2)*pixsize
        f = maxval[2]
        theta = maxval[3]
        r_ellipse = np.sqrt((f*(np.cos(theta))**2 + 1/f*(np.sin(theta))**2)*(i - maxval[0])**2  \
                        + (f*(np.sin(theta))**2 + 1/f*(np.cos(theta))**2)*(j - maxval[1])**2  \
                        + 2*(np.cos(theta))*(np.sin(theta))*(f-1/f)*(i - maxval[0])*(j - maxval[1]))*pixsize
        
        y_radius = np.interp(r_ellipse, rs_new, ys_new)
        y_fluc[t1][t2] = data[i][j] - y_radius
        normalised_y_fluc[t1][t2] = y_fluc[t1][t2]/abs(y_radius)
        t2 += 1
    t2 = 0
    t1 += 1
    
file_norm_yfluc = open("normalised_y_fluc_last.txt","w+")
np.savetxt(file_norm_yfluc, normalised_y_fluc)
file_norm_yfluc.close()

file_yfluc = open("y_fluc_last.txt","w+")
np.savetxt(file_yfluc, y_fluc)
file_yfluc.close()

mid = int(len(y_fluc)/2.)
dist = int((len(y_fluc)/2.)/4)
NGC4839 = [345, 381]

print("\nLength of y fluctuations matrix =",len(y_fluc))

from matplotlib.colors import TwoSlopeNorm     #Needed in plotting diverging colormaps
"""
x_ticks = ['-1', '-0.75','-0.5','-0.25','0','0.25','0.5', '0.75','1']
y_ticks = ['-1', '-0.75','-0.5','-0.25','0','0.25','0.5', '0.75','1']
t11 = [0,8.75,17.5,26.25,35,43.75,52.5,61.25,69]
"""

x_ticks = ['-2', '-1','0','1','2']
y_ticks = ['-2', '-1','0','1','2']
t11 = [0,35,70,105,138]

plt.figure()
plt.xticks(ticks=t11, labels=x_ticks, size='small')
plt.yticks(ticks=t11, labels=y_ticks, size='small')

norm = TwoSlopeNorm(vmin=y_fluc.min(), vcenter=0, vmax=y_fluc.max())
pc = plt.pcolormesh(y_fluc, norm=norm, cmap="seismic")     
plt.imshow(y_fluc, cmap = 'seismic')
ax = plt.gca()
rect = patches.Rectangle((mid-dist,mid-dist), 2*dist, 2*dist, linewidth=2,
                         edgecolor='black', facecolor="none")
ax.add_patch(rect)
circ = patches.Circle((NGC4839[0]-x_ind[0][0],NGC4839[1]-y_ind[0][0]), 15/pixsize , linewidth=2,
                         edgecolor='green', facecolor="none")
ax.add_patch(circ)
plt.colorbar(pc)
plt.xlabel("degrees")
plt.ylabel("degrees")
plt.title("y fluctuations in MILCA (f=%2.2f)"%maxval[2])
#plt.savefig("figure 3.png", dpi = 400)
plt.show()


plt.figure()
plt.xticks(ticks=t11, labels=x_ticks, size='small')
plt.yticks(ticks=t11, labels=y_ticks, size='small')
 
norm = TwoSlopeNorm(vmin=-1.5, vcenter=0, vmax=1.5) 
pc = plt.pcolormesh( normalised_y_fluc, norm=norm, cmap="seismic") 
plt.imshow(normalised_y_fluc, cmap = 'seismic')
ax = plt.gca()
circ = patches.Circle((NGC4839[0]-x_ind[0][0],NGC4839[1]-y_ind[0][0]), 15/pixsize , linewidth=2,
                         edgecolor='green', facecolor="none")
ax.add_patch(circ)
plt.colorbar(pc)
plt.xlabel("degrees")
plt.ylabel("degrees")
plt.title("normalised y fluctuations in MILCA (f=%2.2f)"%maxval[2])
#plt.savefig("figure 4.png", dpi = 400)
plt.show()


