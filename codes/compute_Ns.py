import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as fftpack
import scipy.signal as ss
import scipy.integrate as si
import math

m,c,sigma_T = 9.1093e-31, 299792458, 6.652e-29

# def abel(theta,r,y0,theta_c,beta):
#     integrand = m*c/(sigma_T*np.pi)*y0*2*theta/((1+theta**2/theta_c)**(beta+1)*(theta**2-r**2))
#     return integrand

y0 = 5.713907103243261e-05
theta_c = 613.0520060958096
beta = 1.4086450050856862

r_c = 745.0890823973396 
p0 = 1.4840887724753505e-06


def windowFunc(theta,z):
    w = (p0/y0)*(1+theta**2/theta_c**2)**beta/(1+(theta**2+z**2)/r_c**2)**(beta+1)
    return w

zs = np.arange(-2000,2000,1)
ws1 = windowFunc(0,zs)
ws2 = windowFunc(500,zs)

plt.plot(zs,ws1,label="theta=0 kpc")
plt.plot(zs,ws2,label="theta=500 kpc")
plt.legend()
plt.show()

ws = np.zeros((2000,4000))
ft_ws = np.zeros((2000,4000))
freqs_ws = np.zeros((2000,4000))

i=0
for theta in range (0,2000,1):
    ws[i,:] = windowFunc(theta,zs)
    ft_ws[i,:] = np.abs(fftpack.fft(ws[i,:]))**2
    freqs_ws[i,:] = fftpack.fftfreq(len(ws[i,:]), d=(zs[1]-zs[0]))
    i+=1
    
plt.imshow(ws)
plt.xlabel("z (-2000 to 2000 kpc)")
plt.ylabel("theta (0-2000 kpc)")
plt.title("W(z,theta)")
plt.show()

plt.plot(freqs_ws[0,:],ft_ws[0,:],label="theta=0 kpc")
plt.plot(freqs_ws[500,:],ft_ws[500,:],label="theta=500 kpc")
plt.loglog()
plt.legend()
plt.show()

Ns = np.zeros(2000)

i=0
for theta in range (0,2000,1):
    Ns[i] = si.simps(ft_ws[i,:],freqs_ws[i,:])/(2*np.pi)
    i+=1
    
plt.plot(Ns)
plt.xlabel("thetas (kpc)")
plt.ylabel("N")
plt.show()

















