#binned_fluctuations

#This script is used to calculate the y profile fluctuations and normalized y profile fluctuations in come cluster and then plotting them in colormaps. 
#Note: We have used the data at radius 0 from binned_beta_fit.py to interpolate y's between 0 and the centre of the first bin (at 1.5 arcmin). 
#Hence ys_2d function has not been called here. The sizes of the arrays here has thus been incremented by 1 to accomodate the point at radius 0

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import scipy.optimize as opt

#reading and extracting the data
f = fits.open('map2048_MILCA_Coma_20deg_G.fits')
data = f[1].data    

#defining number of pixels i.e, 700 in this case on both the axes-x and y, then converting to a grid 
x=np.arange(700)
y=np.arange(700) 
X, Y = np.meshgrid(x, y) 

#for maximum coordinates
maxval = np.unravel_index(np.argmax(data), data.shape) 

# calculate distance in pixel coordinates from the centre
R=np.sqrt((X-maxval[0])**2+(Y-maxval[1])**2) 

pixsize = 1.7177432059 #pixsize is the length corresponding to 1 pixel in arcmin
conv =  27.052 #conv is the factor to convert arcmin to kpc

#taking initial and final r values for the region of coma cluster taken
r_i = 0    
r_f = 112    
step_size = 3

if (r_f-r_i)%step_size==0:
    size = int((r_f - r_i) / step_size +1)
else:
    size = int((r_f - r_i) / step_size +2 )


#defining arrays to store y profile and r's
ys = np.zeros(size)
rs = np.zeros(size)

i=r_i
t=1

#running loop till the final value of the region of coma cluster to calculate the average y profile in the selected coma cluster region.
while i<r_f:
    #stores the indices of all elements within a given annulus
    in_ann=np.nonzero((R*pixsize>i) & (R*pixsize<(i+step_size)))
    av=np.mean(data[in_ann])
    
    
    ys[t] = av
    rs[t] = i+step_size/2
    
    t+=1
    i+=step_size
    
def f(x,a,b,c):
    return a/(1+(x/b)**2)**c

constants = opt.curve_fit(f, rs[1:], ys[1:])    #optimize module is used to curve fit 

a_fit = constants[0][0]

ys[0]=a_fit
rs[0]=0 
    
a=121 #arcmin

plt.plot(rs,ys)
plt.show()

print("\nlength of image in arcmin = ",a)

#Taking all the elements i.e., y_ind and x_ind are within a square of length a with centre as coma cluster's centre.
x_ind = np.nonzero(((x-maxval[0])*pixsize>=-(a/2)) & (((x-maxval[0])*pixsize<=(a/2))))
y_ind = np.nonzero(((y-maxval[1])*pixsize>=-(a/2)) & (((y-maxval[1])*pixsize<=(a/2))))

#arrays used to store y fluctuations and normalised y fluctuations
y_fluc = np.zeros((len(x_ind[0]),len(y_ind[0])))
normalised_y_fluc = np.zeros((len(x_ind[0]),len(y_ind[0])))


t1,t2 = 0,0

#running loop to interpolate value of y profile using the known ones and then calculating the fluctuation by subtracting the value we get by interpolation from data in the milca map as well as the normalized fluctuations.
for i in y_ind[0]:
    for j in x_ind[0]:
        radius = np.sqrt((i-maxval[0])**2 + (j-maxval[1])**2)*pixsize #distance from centre.
        y_radius = np.interp(radius, rs, ys)
        y_fluc[t1][t2] = data[i][j] - y_radius
        normalised_y_fluc[t1][t2] = (data[i][j] - y_radius)/y_radius

        t2 += 1
    t2 = 0
    t1 += 1


print("\n")
 
from matplotlib.colors import TwoSlopeNorm     

#x and y labels at approriate intervals are set
x_ticks = ['-1', '-0.75','-0.5','-0.25','0','0.25','0.5', '0.75','1']
y_ticks = ['-1', '-0.75','-0.5','-0.25','0','0.25','0.5', '0.75','1']

t11 = [0,8.75,17.5,26.25,35,43.75,52.5,61.25,69]

plt.xticks(ticks=t11, labels=x_ticks, size='small')
plt.yticks(ticks=t11, labels=y_ticks, size='small')

#the min, max and centre of the colormap is defined and y fluctuations are plotted
norm = TwoSlopeNorm(vmin=y_fluc.min(), vcenter=0, vmax=y_fluc.max())
pc = plt.pcolormesh(y_fluc, norm=norm, cmap="seismic")     
plt.imshow(y_fluc, cmap = 'seismic')
plt.colorbar(pc)
plt.xlabel("degrees")
plt.ylabel("degrees")
plt.title("y fluctuations in MILCA")
plt.savefig("figure 3.svg", dpi = 1200)
plt.show()

#x and y labels at approriate intervals are set
x_ticks = ['-1', '-0.75','-0.5','-0.25','0','0.25','0.5', '0.75','1']
y_ticks = ['-1', '-0.75','-0.5','-0.25','0','0.25','0.5', '0.75','1']

t11 = [0,8.75,17.5,26.25,35,43.75,52.5,61.25,69]

plt.xticks(ticks=t11, labels=x_ticks, size='small')

plt.yticks(ticks=t11, labels=y_ticks, size='small')

plt.xticks(ticks=t11, labels=x_ticks, size='small')
plt.yticks(ticks=t11, labels=y_ticks, size='small')
 
#the min, max and centre of the colormap is defined and normalized y fluctuations are plotted
norm = TwoSlopeNorm(vmin=normalised_y_fluc.min(), vcenter=0, vmax=normalised_y_fluc.max())
pc = plt.pcolormesh( normalised_y_fluc, norm=norm, cmap="seismic") 
plt.imshow(normalised_y_fluc, cmap = 'seismic')
plt.colorbar(pc)
plt.xlabel("degrees")
plt.ylabel("degrees")
plt.title("normalised y fluctuations in MILCA")
plt.savefig("figure 4.svg", dpi = 1200)
plt.show()
