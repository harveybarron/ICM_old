import numpy as np
import pymaster as nmt
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import scipy.integrate as si

pixsize = 1.7177432059
conv =  27.052 #kpc

y_fluc = np.loadtxt('y_fluc.txt')

Lx = 4. * np.pi/180
Ly = 4. * np.pi/180
Nx, Ny = len(y_fluc),  len(y_fluc)

"------------------------------- CREATING MASKS -------------------------------"


mask = np.zeros((Nx,Ny))
cen_x, cen_y = Nx/2., Ny/2.
cr = 60
I,J=np.meshgrid(np.arange(mask.shape[0]),np.arange(mask.shape[1]))
dist=np.sqrt((I-cen_x)**2+(J-cen_y)**2)
dist = dist * pixsize
idx = np.where(dist<=cr)
theta_ap = 15 
mask[idx]=1-np.exp(-9*(dist[idx]-cr)**2/(2*theta_ap**2))


"---------------------------- PLOTTING APODIZED MASK --------------------------"

plt.figure()
x_ticks = ['-2', '-1','0','1','2']
y_ticks = ['-2', '-1','0','1','2']
t11 = [0,35,70,105,138]
plt.xticks(ticks=t11, labels=x_ticks, size='small')
plt.yticks(ticks=t11, labels=y_ticks, size='small')
plt.imshow(mask, interpolation='nearest', origin='lower')
plt.colorbar()
plt.savefig("Apodized mask.png",dpi = 400)
plt.show()

"------------------------------ CREATING BINS ---------------------------------"


bin_number = 6

l_min =  (180*60*conv/2000)
l_max = (180*60*conv/500)

bin_size = (l_max-l_min)/bin_number

l0_bins=[]
lf_bins=[]

for i in range (bin_number):
    l0_bins.append(l_min+bin_size*i)
    lf_bins.append(l_min+bin_size*(i+1))


b = nmt.NmtBinFlat(l0_bins, lf_bins)
ells_uncoupled = b.get_effective_ells()


print("\n************************  effective l's  *****************************\n")
print(ells_uncoupled)

lambdas_inv = ells_uncoupled/(conv*60*180)
k = 2*np.pi*lambdas_inv

"----------------------------- DEFINING BEAM ------------------------------------"


def beam(l):
    Planck_res=10./60 
    Planck_sig = Planck_res/2.3548
    return np.exp(-0.5*l*(l+1)*(Planck_sig*np.pi/180)**2)
    


"----------------------------- PLOTTING FIELD WITH MASK -------------------------"

bin_number = 200
l_min = (180*60*conv/(120*conv))
l_max = (180*60*conv/conv)
bin_size = (l_max-l_min)/bin_number
ls = []
for i in range (bin_number):
    ls.append(l_min+bin_size*(i+0.5))

f0 = nmt.NmtFieldFlat(Lx, Ly, mask, [y_fluc] ,beam=[ells_uncoupled, beam(ells_uncoupled)])
#,beam = [ells_uncoupled, beam(ells_uncoupled)]) np.array(ls), beam(np.array(ls))
plt.figure()
plt.imshow(f0.get_maps()[0] * mask, interpolation='nearest', origin='lower')
plt.colorbar()
plt.savefig('map with mask.png', dpi = 400)
plt.show()



print("\n--------------------------- ANGULAR POWER SPECTRUM ------------------------------------\n")

w00 = nmt.NmtWorkspaceFlat()
w00.compute_coupling_matrix(f0, f0, b)

cl00_coupled = nmt.compute_coupled_cell_flat(f0, f0, b)
cl00_uncoupled = w00.decouple_cell(cl00_coupled)[0]
print(cl00_uncoupled)

amp = abs((ells_uncoupled**2)*cl00_uncoupled/(2*np.pi))**(1/2)



print("\n*************************  Covariance matrix  *************************************\n")

cw = nmt.NmtCovarianceWorkspaceFlat()
cw.compute_coupling_coefficients(f0, f0, b)
covar = nmt.gaussian_covariance_flat(cw, 0, 0, 0, 0, ells_uncoupled,
                                     [cl00_uncoupled], [cl00_uncoupled],
                                     [cl00_uncoupled], [cl00_uncoupled], w00)

print(covar)
std_power = (np.diag(covar))
std_amp = np.sqrt(abs((ells_uncoupled**2)*std_power/(2*np.pi))**(1/2))



"--------------------------------- Fitting a power law -------------------------------------"

def power_law(x,a,p):
    return a*np.power(x,p)

a_fit, p_fit = curve_fit(power_law, lambdas_inv, amp, p0 = [1e-2,5/3])[0]

lambdas_inv_curve = np.linspace(min(lambdas_inv),max(lambdas_inv),100)
curve = power_law(lambdas_inv_curve, a_fit,p_fit)



"------------------------- Plotting amplitude of Power Spectrum vs 1/lambda ---------------------------"

plt.figure()
plt.plot(lambdas_inv, amp, 'r.', label='Amplitude of power spectrum')
plt.errorbar(lambdas_inv,amp, yerr=std_amp, fmt='r.',ecolor='black',elinewidth=1,
            capsize = 4)
plt.plot(lambdas_inv_curve, curve, 'b', 
         label='Best fit: Power law (power = %1.2f)'%p_fit)
plt.xlabel("$1/\lambda$ ($kpc^{-1}$)")
plt.ylabel("Amplitude of power spectrum")
plt.legend()
plt.title("Power Spectrum of Coma")
plt.savefig("power_spectrum.png", dpi = 400)
plt.show()



print("\n---------------------------- PARSEVAL CHECK ---------------------------------\n")

len_y_fluc = np.shape(y_fluc)
variance = np.sum((y_fluc-y_fluc.mean())**2)/(len_y_fluc[0]*len_y_fluc[1])
print("Variance of map =",variance)

meanSq = np.sum(amp**2)/len(amp)
print("Average of amplitude^2 of power = ",meanSq)








