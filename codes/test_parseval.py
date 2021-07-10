import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as fftpack
import scipy.signal as ss
import scipy.integrate as si

pi = np.pi

tdata = np.arange(5999.)/300
dt = tdata[1]-tdata[0]

datay = np.sin(pi*tdata)+2*np.sin(pi*2*tdata)
fouriery_1 = fftpack.fft(datay)
fouriery_2 = np.fft.fft(datay)

N = len(datay)

parseval_1 = np.sum((datay)**2)
parseval_2_1 = np.sum(np.abs(fouriery_1)**2) / N
parseval_2_2 = np.sum(np.abs(fouriery_2)**2) / N


freqs = fftpack.fftfreq(len(datay), d=(tdata[1]-tdata[0]))
plt.figure()
plt.plot(freqs, abs(fouriery_2), 'b-')
plt.xlim(0,3)
plt.show()

freq , PS = ss.periodogram((datay),1./(tdata[1]-tdata[0]),return_onesided=False)

parseval_2_3 = si.simps(PS,freq)

plt.figure()
plt.plot(freq, PS )
plt.xlim(-3,3)
plt.show()

print("\nParseval's thorem check for FFT:\n")
print ("Difference using scipy fft:",parseval_1 - parseval_2_1)
print ("Difference using numpy fft:",parseval_1 - parseval_2_2)

print("\nParseval's theorem check for Power Spectrum using periodogram: \n")
print ("Variance of map:",parseval_1/N) 
print ("Integral of P(f)df =",parseval_2_3)















