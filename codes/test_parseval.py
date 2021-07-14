import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as fftpack
import scipy.signal as ss
import scipy.integrate as si

pi = np.pi

print("\n------------------------ For 1D data ----------------------------------\n")

tdata = np.arange(5999.)/300
dt = tdata[1]-tdata[0]
datay = np.sin(pi*tdata)+2*np.sin(pi*2*tdata)

plt.figure()
plt.plot(tdata,datay)
plt.show()

fouriery_1 = fftpack.fft(datay)
fouriery_2 = np.fft.fft(datay)

N = len(datay)

parseval_1 = np.sum((datay)**2)
parseval_2_1 = np.sum(np.abs(fouriery_1)**2) / N
parseval_2_2 = np.sum(np.abs(fouriery_2)**2) / N

freqs = fftpack.fftfreq(len(datay), d=(tdata[1]-tdata[0]))
freq , PS = ss.periodogram((datay),1./(tdata[1]-tdata[0]),return_onesided=False)

parseval_2_3 = si.simps(PS,freq)


print("\nParseval's thorem check for FFT:\n")
print ("Difference using scipy fft:",parseval_1 - parseval_2_1)
print ("Difference using numpy fft:",parseval_1 - parseval_2_2)

plt.figure()
plt.plot(freqs, abs(fouriery_2), 'b-')
plt.xlim(0,3)
plt.show()

print("\nParseval's theorem check for Power Spectrum using periodogram: \n")
print ("Variance of signak:",parseval_1/N) 
print ("Integral of P(f)df =",parseval_2_3)

plt.figure()
plt.plot(freq, PS )
plt.xlim(-3,3)
plt.show()


print("\n------------------------ For 2D data ----------------------------------\n")

x=np.arange(100)/10
y=np.arange(100)/10   
X, Y = np.meshgrid(x, y) 
data2 = np.sin(pi*X)+2*np.sin(pi*2*Y)

plt.figure()
plt.imshow(data2,interpolation='nearest')
plt.colorbar()
plt.savefig("2D test map.png",dpi=400)
plt.show()

fourier_data2 = np.abs(np.fft.fft2(data2))
fourier_shift = np.fft.fftshift(fourier_data2)

N = len(data2)
parseval_3 = np.sum((data2-data2.mean())**2)
parseval_4_1 = np.sum(np.abs(fourier_data2)**2) / N**2


print("\nParseval theorem check for FFT: \n")
print(parseval_3)
print(parseval_4_1)

plt.figure()
plt.imshow(fourier_shift)
plt.colorbar()
plt.savefig("2D test map FT.png",dpi=400)
plt.show()

import pymaster as nmt
conv =  27.052 #kpc

Nx = 100
Ny = 100

pix=1.7/60 # deg
Lx = Nx*pix * np.pi/180
Ly = Ny*pix * np.pi/180

mask=np.ones_like(data2)
f0 = nmt.NmtFieldFlat(Lx, Ly, mask, [data2])


bin_number = 100

l_min = 0 # (180*60*conv/2000)
l_max = 7000 # (180*60*conv/500)

bin_size = (l_max-l_min)/bin_number

l0_bins=[]
lf_bins=[]

for i in range (bin_number):
    l0_bins.append(l_min+bin_size*i)
    lf_bins.append(l_min+bin_size*(i+1))
    
b = nmt.NmtBinFlat(l0_bins, lf_bins)
ells_uncoupled = b.get_effective_ells()
w00 = nmt.NmtWorkspaceFlat()
w00.compute_coupling_matrix(f0, f0, b)
cl00_coupled = nmt.compute_coupled_cell_flat(f0, f0, b)
cl00_uncoupled = w00.decouple_cell(cl00_coupled)[0]

lambdas_inv = ells_uncoupled/(conv*60*180)
k = 2*np.pi*lambdas_inv
amp = abs((ells_uncoupled**2)*cl00_uncoupled/(2*np.pi))**(1/2)

print("\nParseval's theorem check for Power Spectrum using NaMaster: \n")
parseval_4_4 = si.simps(cl00_uncoupled, ells_uncoupled)
parseval_4_5 = np.sum(cl00_uncoupled)

print("Variance of map:",parseval_3/(Nx*Ny))
print("Integral = ",parseval_4_4)
print("Summation=",parseval_4_5)

plt.figure()
plt.plot(ells_uncoupled, cl00_uncoupled, 'r-', label='Uncoupled, no beam')
plt.title("Using NaMaster")
plt.savefig("2D test map power using NaMaster.png",dpi=400)
plt.show()





