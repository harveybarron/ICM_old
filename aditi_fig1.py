import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm   
from astropy.visualization import astropy_mpl_style
plt.style.use(astropy_mpl_style)
from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits
image_file = get_pkg_data_filename('map2048_MILCA_Coma_20deg_G.fits')
fits.info(image_file)
image_data = fits.getdata(image_file, ext=0)
print(image_data.shape)
plt.figure()

norm = TwoSlopeNorm(vmin=image_data.min(), vcenter=0, vmax=image_data.max())
plt.imshow(image_data, norm=norm, cmap='GnBu')
plt.title("Y map")
plt.xlabel("degrees")
plt.ylabel("degrees")

plt.colorbar()
plt.savefig('img1.png')


