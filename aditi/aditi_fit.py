import numpy as np
from numpy import arange
from astropy.io import fits
import scipy.optimize as opt
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from astropy.utils.data import download_file
from astropy.io import fits

hdu_list = fits.open('map2048_MILCA_Coma_20deg_G.fits', memmap=True)
hdu_list.info()
from astropy.table import Table

evt_data = (hdu_list[1].data)

print(evt_data)

x,y = np.unravel_index(evt_data.argmax(), evt_data.shape)
print("centre of cluster is at:", x,y)

initial = 1
final = 65
size = final - initial

y_log = np.zeros(size)
r_log = np.zeros(size)

ys = np.zeros(size)
rs = np.zeros(size)

for k in range(initial, final):
	dth = 1/(100*k)
	th = 0
	c = 0
	s = 0
	while(th<=2*np.pi):
		i = int(abs(x+k*np.cos(th)))
		j = int(abs(y+k*np.sin(th)))
		s += evt_data[i][j]
		c += 1
		th += dth
	avg = s/c

	ys[k - initial] = (avg)
	rs[k - initial] = (k)
	
	y_log[k - initial] = np.log10(avg)
	r_log[k - initial] = np.log10(k)
	
def g(x,p,q,r):
	return p/(1+(x/q)**2)**r

	
l = r_log[:]
t = y_log[:]

popt,_ = opt.curve_fit(g,rs,ys)
p,q,r = popt
print(p,q,r)

beta = np.zeros(size)
beta_log = np.zeros(size)

for j in range(initial, final):
	beta[j-initial] = g(j, p, q, r)
	beta_log[j - initial] = np.log10(g(j, p, q, r))

plt.plot(r_log, y_log, color='blue',label = 'Milca profile')
plt.plot(r_log, beta_log,'--', color = 'red', label = 'beta model')
plt.title("milca map with beta fitting")
plt.xlabel("distance from centre")
plt.ylabel("avg y profile")
plt.legend()
plt.grid()
plt.show()


