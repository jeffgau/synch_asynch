# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 09:41:51 2020

@author: Administrator
"""

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.fftpack import fft



def find_nearest(array,value):
    """
    Find nearest value in array
    """
    array = np.asarray(array)
    idx = (np.abs(array-value)).argmin()
    return idx


def fourier_analysis_plot(data, sampling_f):
    N = len(data)
    T = 1.0 / sampling_f
    yf = 2 * fft(data)/N
    # nyquist_f = np.squeeze(1/(2*T))
    xf = np.squeeze(np.linspace(0.0, 1.0/(2.0*T), N//2))
    y_real = np.squeeze(np.abs(yf[0:N//2]))
    return (xf, y_real)


def create_kernel(t, r3, r4, k3, k4):
# Create a dSA kernel given rates (r3, r4) and gains (k3, k4)


    f_sa = k3*(1 - np.exp(-r3*t))  + k4*(np.exp(-r4*t) - 1)


    #f_temp = - (f_sa - (k3 - k4))
    f_temp = -f_sa
    f_temp = f_temp - f_temp[0]
    f_temp = f_temp/max(abs(f_sa))
    
    f_temp = np.array(f_temp)

    idx_peak = np.where(abs(f_sa) == max(abs(f_sa)))[0][0]
    idx_end = np.where(f_temp[idx_peak:] > - 0.01)[0][0] + idx_peak
    f_kernel = f_sa[:idx_end]
    
    # Normalize area under the curve to 1
    auc = np.trapz(f_kernel, t[0:len(f_kernel)])
    f_kernel = f_kernel/auc

    return f_kernel


"""
plot kernels vs r3
"""
sampling_f = 100000
ntests = 5
t = np.linspace(0, 10, sampling_f)
k3 = 1
k4 = 1
r4_ratio = 0.1

r3_range = np.logspace(1, 4, ntests)

plt.figure(figsize = (3, 2.5))
sns.set(font_scale = 1, style = 'ticks')
for r3 in r3_range:
    r4 = r3 * r4_ratio
    f_kernel = create_kernel(t, r3, r4, k3, k4)
    t_plot = np.linspace(1/sampling_f, len(f_kernel)/sampling_f, len(f_kernel))
    
    plt.semilogx(t_plot, f_kernel/np.max(f_kernel), linewidth = 2, c = '#C1272D')
    
sns.despine()
plt.xlabel('time (s)')
#plt.ylabel(r'g$_*$x')
plt.ylabel(r'$\hat{g}$(t)')
plt.tight_layout()
plt.savefig('figures/kernels.svg', format = 'svg')
plt.savefig('figures/kernels.png', format = 'png', dpi = 500)

plt.show()


"""
##############################################################################
Plot t_o vs r3
##############################################################################
"""

ntests = 100
r3_range = np.logspace(1, 4, ntests)
plt.figure(figsize = (3, 2.5))
to = np.log(k3*r3_range/k4*r3_range*r4_ratio)/(r3_range - r3_range*r4_ratio)

plt.loglog(r3_range, to, linewidth = 2)
sns.despine()
plt.xlabel(r'r3 (s$^{-1}$)')
plt.ylabel(r't$_o$ (s)')
plt.tight_layout()
plt.savefig('figures/r3_to.svg', format = 'svg')
plt.savefig('figures/r3_to.png', format = 'png', dpi = 500)

plt.show()


"""
Plot g*x for sinusoidal length inputs at different frequencies and fixed g
"""

f_range = [1, 5,25, 50, 100]
sampling_f = 10000
t_end = 1
t = np.linspace(0, t_end, t_end * sampling_f)


r3 = 100000
r4 = r3 * r4_ratio
k3 = 1 
k4 = 1

kernel = create_kernel(t, r3, r4, k3, k4)
sampling_f = 1000
for f in f_range:
    w = 2*np.pi*f
    x = np.sin(w*t)
    
    f_a = np.convolve(x, kernel, 'full')
    #plt.plot(t, f_a)
    plt.plot(kernel)
    plt.plot(f_a)
    plt.plot(x)
    plt.show()
   # plt.show()
    
"""
plot g*x for varying r3
"""

f = 25
sampling_f = 10000
t_end = 10
t = np.linspace(0, t_end, t_end * sampling_f)

x = np.sin(f*2*np.pi*t)

r3_range = np.logspace(1, 4, 5)
k3 = 1 
k4 = 1

for r3 in r3_range:
    r4 = r3 * r4_ratio
    kernel = create_kernel(t, r3, r4, k3, k4)
    f_a = np.convolve(x, kernel, 'valid')

    plt.plot(x[-1000:]/max(x))
    #plt.plot(kernel/max(kernel))
    plt.plot(f_a[-1000:]/max(f_a))
    plt.show()
    
#%%
    
"""
##############################################################################
plot power freq amp from MATLAB simulation
##############################################################################
"""

"""
Notes:
    
  if data.shape = (20, 10), then the resulting pcolor has 20 rows and 10 columns.  
"""
from scipy.io import loadmat
synch_gain_range = loadmat('data/synch_gain_range.mat')['synch_gain_range'][0].flatten()
yax = loadmat('data/yax.mat')['yax'][0].flatten()
power = loadmat('data/power.mat')['power']
freq = loadmat('data/freq.mat')['freq']
osc_amp = loadmat('data/osc_amp.mat')['osc_amp']
psd = loadmat('data/psd.mat')['psd']
r3_range = loadmat('data/r3_range.mat')['r3_range']

power_df = pd.DataFrame(data = power, columns = synch_gain_range, index = yax)
power_df = power_df.rename_axis('yax')
power_df = power_df.rename_axis('synch_gain_range', axis = 'columns')



from matplotlib import colors

cmap = colors.ListedColormap(['red', 'blue'])

class MidpointNormalize(colors.Normalize):
	"""
	Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)

	e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
	"""
	def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
		self.midpoint = midpoint
		colors.Normalize.__init__(self, vmin, vmax, clip)

	def __call__(self, value, clip=None):
		# I'm ignoring masked values and all kinds of edge cases to make a
		# simple example...
		x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
		return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))

X, Y = np.meshgrid(synch_gain_range, yax)
fig, ax = plt.subplots(2, 1, figsize = (3,5))
sns.set(font_scale = 1, style = 'ticks')
#pos = ax[0].imshow(power_df[::-1], cmap = 'Reds', extent = [np.min(synch_gain_range), np.max(synch_gain_range),0.01, 1])

pos = ax[1].pcolor(X, Y, power_df, cmap = 'Reds', edgecolors = 'face', linewidth = .1)
ax[1].set_yscale('log')
cbar = fig.colorbar(pos, ax = ax[1])#,fraction=0.046, pad=0.04) 
cbar.ax.set_ylabel('power (au)')
ax[1].set_ylabel(r'$t_o/T_n$')
ax[1].axis('equal')
freqcmap = colors.LinearSegmentedColormap.from_list('freqcmap', ['#FFFFFF', '#B2CCFB','#FF3032'])
#fig, ax = plt.subplots(1, 1, figsize = (3,3))
sns.set(font_scale = 1, style = 'ticks')
pos = ax[0].pcolor(X,Y, freq, cmap = freqcmap, norm=MidpointNormalize(midpoint=1,vmin=0, vmax=3), vmin = 0, vmax = 3, edgecolors = 'face', linewidth = .1)
#ax.set_ysclale('log')
cbar = fig.colorbar(pos, ax = ax[0])#,fraction=0.046, pad=0.04) 
cbar.ax.set_ylabel(r'$f/f_s$')
cbar.set_ticks([0, 1, 2, 3, 4])
#pos.set_clim(0,4)
ax[0].set_ylabel(r'$t_o/T_n$')
ax[1].set_xlabel(r'$K_r$')
ax[0].set_yscale('log')
ax[0].axis('equal')

#ax[0].scatter(1, 0.55)
sns.despine()
#plt.tight_layout()
plt.savefig('figures/osc_meaurements.svg', format = 'svg', transparent = True)
plt.savefig('figures/osc_measurements.png', format = 'png', dpi = 500)

plt.show()


#%%
"""
##############################################################################
Plot slices from simulation
##############################################################################
"""
# Manduca t_o/T_n = 0.37

m_val = 0.37
moth_slice = find_nearest(yax, m_val)

asynch_power = power[:,0] # slice along K_r = 0

asynch_slice = np.where(asynch_power == np.max(asynch_power))[0][0]  
transition_slice = find_nearest(yax, .35)

fig, ax = plt.subplots(1,1, figsize = (2.4, 1))
sns.set(font_scale = 1, style = 'ticks')
plt.plot(synch_gain_range, power[moth_slice,:], c = '#0071BC', linewidth = 2)
plt.plot(synch_gain_range, power[asynch_slice,:], c = '#C1272D', linewidth = 2)
#plt.plot(synch_gain_range, power[transition_slice,:], c = 'k', linewidth = 2, linestyle = '--')

ax.set_xticks([0, 0.5, 1])
ax.set_xlabel('$K_r$')
ax.set_ylabel('power (au)')
ax.axis([0, 1, 0, np.max(power)])
sns.despine()
plt.savefig('figures/power_slices.svg', format = 'svg')
plt.show()


fig, ax = plt.subplots(1,1, figsize = (4.9, 1))
sns.set(font_scale = 1, style = 'ticks')
plt.plot(synch_gain_range, freq[moth_slice,:], c = '#0071BC', linewidth = 3)
plt.plot(synch_gain_range, freq[asynch_slice,:], c = '#C1272D', linewidth = 3)
#plt.plot(synch_gain_range, freq[transition_slice,:], c = 'k', linewidth = 2, linestyle = '--')

ax.set_xticks([0, 0.5, 1])
ax.set_xlabel('$K_r$')
ax.set_ylabel('$f/f_s$')
ax.axis([0, 1, 0, np.max(freq)])
sns.despine()
plt.savefig('figures/freq_slices.svg', format = 'svg')
plt.show()




#%%
"""
##############################################################################
Plot freq alone from simulation
##############################################################################
"""


X, Y = np.meshgrid(synch_gain_range, yax)
fig, ax = plt.subplots(1, 1, figsize = (6,4.8))
sns.set(font_scale = 1, style = 'ticks')
#pos = ax[0].imshow(power_df[::-1], cmap = 'Reds', extent = [np.min(synch_gain_range), np.max(synch_gain_range),0.01, 1])

#freqcmap = colors.LinearSegmentedColormap.from_list('', ['#E0DB26', '#B2CCFB','#FF3032'])
freqcmap = colors.LinearSegmentedColormap.from_list('freqcmap', ['white', '#FF272D'])
freqcmap.set_bad('#8FB0ED')


test_freq = np.ma.masked_values(freq, 1, rtol = 1E-2, copy = True)
#fig, ax = plt.subplots(1, 1, figsize = (3,3))
sns.set(font_scale = 1, style = 'ticks')
#pos = ax.pcolormesh(X,Y, test_freq, cmap = cmap, norm=MidpointNormalize(midpoint=1,vmin=0, vmax=3), vmin = 0, vmax = 3, edgecolor = 'face', linewidth = 1)
pos = ax.pcolormesh(X,Y, test_freq, cmap = freqcmap, vmin = 0, vmax = 3, edgecolor = 'face', linewidth = 1)

#ax.set_ysclale('log')
cbar = fig.colorbar(pos, ax = ax)#,fraction=0.046, pad=0.04) 
cbar.ax.set_ylabel(r'$f/f_s$')
cbar.set_ticks([0, 1, 2, 3, 4])
ax.scatter(0.9, yax[moth_slice], c = '#0071BC', s = 20)
ax.scatter(0.05, yax[asynch_slice], c = '#C1272D', s = 20)
ax.plot([0, 1], [yax[moth_slice], yax[moth_slice]], c = '#0071BC', linewidth = 2)
ax.plot([0, 1], [yax[asynch_slice], yax[asynch_slice]], c = '#C1272D', linewidth = 2)

ax.set_yscale('log')
ax.axis('equal')
ax.set_xticks([])
ax.set_yticks([])
#ax[0].scatter(1, 0.55)
sns.despine()
plt.axis('off')
#plt.tight_layout()
plt.savefig('figures/freq.svg', format = 'svg', transparent = True)
plt.savefig('figures/freq.png', format = 'png', dpi = 500)

plt.show()

#%%
"""
##############################################################################
Plot Manduca power slice from simulation
##############################################################################
"""

from scipy.io import loadmat
synch_gain_range = loadmat('data/synch_gain_range.mat')['synch_gain_range'][0].flatten()
yax = loadmat('data/yax.mat')['yax'][0].flatten()
power = loadmat('data/power.mat')['power']
freq = loadmat('data/freq.mat')['freq']
osc_amp = loadmat('data/osc_amp.mat')['osc_amp']
psd = loadmat('data/psd.mat')['psd']
r3_range = loadmat('data/r3_range.mat')['r3_range']

power_df = pd.DataFrame(data = power, columns = synch_gain_range, index = yax)
power_df = power_df.rename_axis('yax')
power_df = power_df.rename_axis('synch_gain_range', axis = 'columns')



from matplotlib import colors
import scipy




power_cmap = colors.LinearSegmentedColormap.from_list('freqcmap', ['white', '#6018D1'])
#freqcmap.set_bad('#8FB0ED')



X, Y = np.meshgrid(synch_gain_range, yax)
fig, ax = plt.subplots(1, 1, figsize = (3,2.3))
sns.set(font_scale = 1, style = 'ticks')
#pos = ax[0].imshow(power_df[::-1], cmap = 'Reds', extent = [np.min(synch_gain_range), np.max(synch_gain_range),0.01, 1])

pos = ax.pcolor(X, Y, power_df, cmap = power_cmap, edgecolors = 'face', linewidth = .1)

freq_smoothed = scipy.ndimage.filters.gaussian_filter(freq, sigma = 0.8)
power_smoothed = scipy.ndimage.filters.gaussian_filter(power, sigma = 0.8)
cs = ax.contour(X, Y, freq_smoothed, levels = [0.99, 1.01], colors = '#808080', linewidths = 1.5, linestyles = '-', alpha = 1)
#cs = ax.contourf(X, Y, freq_smoothed, levels = [0.99, 1.01], hatches = ['//', '\\' ], linewidths = 2)

#cs = ax.contour(X, Y, power_smoothed, levels = [15], colors = 'r', linewidths = 1.5, linestyles = '-', alpha = 1)
#cs = ax.contour(X, Y, power_smoothed, levels = [np.max(power_smoothed)*0.1], colors = '#B3B3B3', alpha = 0.4)
cs = ax.contourf(X, Y, power, levels = [np.min(power), np.max(power)*0.1], colors = '#B3B3B3', alpha = 0.4)
ax.plot([0, 1], [yax[moth_slice], yax[moth_slice]], c = '#0071BC', linewidth = 2)
ax.plot([0, 1], [yax[asynch_slice], yax[asynch_slice]], c = '#C1272D', linewidth = 2)
ax.scatter(0.9, yax[moth_slice], c = '#0071BC', s = 20)
ax.scatter(0.05, yax[asynch_slice], c = '#C1272D', s = 20)
#plt.plot([0, 1], [yax[transition_slice], yax[transition_slice]], c = 'k', linewidth = 2, linestyle = '--')
#ax.clabel(cs, inline = 1, fontsize = 6)
ax.set_yscale('log')
cbar = fig.colorbar(pos, ax = ax)#,fraction=0.046, pad=0.04) 
cbar.ax.set_ylabel('power (au)')
ax.set_ylabel(r'$t_o/T_n$')
ax.axis('equal')
#fig, ax = plt.subplots(1, 1, figsize = (3,3))
sns.set(font_scale = 1, style = 'ticks')
#pos.set_clim(0,4)
ax.set_ylabel(r'$t_o/T_n$')
ax.set_xlabel(r'$K_r$')
ax.set_xticks([0, 0.5, 1])
ax.axis('equal')
#plt.plot([0, 1], [m_val, m_val], c = '#0071BC', linewidth = 3, solid_capstyle = 'butt')
#plt.plot([0, 1], [yax[asynch_slice], yax[asynch_slice]], c = '#C1272D', linewidth = 3, solid_capstyle = 'butt')
#ax[0].scatter(1, 0.55)
sns.despine()
#plt.tight_layout()
plt.savefig('figures/power.svg', format = 'svg', transparent = True)
plt.savefig('figures/power.png', format = 'png', dpi = 500)

plt.show()




#%%
fig, ax = plt.subplots(1, 1, figsize = (3,3))
sns.set(font_scale = 1, style = 'ticks')
pos = ax.imshow(psd[::-1], cmap = 'seismic', extent = [np.min(synch_gain_range), np.max(synch_gain_range),0.01, 1], vmin = 0, vmax = 1)
#ax.set_ysclale('log')
cbar = fig.colorbar(pos, ax = ax) 
cbar.ax.set_ylabel('energy ratio (au)')
plt.xlabel(r'$K_r$')
plt.ylabel(r'$t_o/T_n$')
sns.despine()
plt.show()

"""
plot freq vs r3
"""
sns.set(font_scale = 1, style = 'ticks')

fig, ax = plt.subplots(1,1, figsize =(3,3))
freq_plot = freq[:,0] * 16
ax.scatter(freq_plot, r3_range)
plt.axis([0, 200, 0, 1000])
sns.despine()
plt.tight_layout()
plt.show()

#%%
"""
##############################################################################
plot limit cycle data
##############################################################################
"""
synch_gain_range = loadmat('data/limit_cycle/synch_gain_range.mat')['synch_gain_range'][0].flatten()

t = loadmat('data/limit_cycle/t.mat')['t'][0].flatten()
r3 = loadmat('data/limit_cycle/r3.mat')['r3'][0].flatten()
pos = loadmat('data/limit_cycle/pos.mat')['pos']
vel = loadmat('data/limit_cycle/vel.mat')['vel']


pos = pos[:len(t),:]
vel = vel[:len(t),:]

colors = plt.cm.jet(np.linspace(0, 1, len(synch_gain_range)))

colors = sns.color_palette("vlag", len(synch_gain_range))


sns.set(font_scale = 1, style = 'ticks')

fig, ax = plt.subplots(1, 10, figsize = (8, 1))
for i in range(len(synch_gain_range)):
    pos_plot = pos[int(len(t)/2):,i]
    pos_plot = pos[:,i]
    pos_plot = pos_plot - np.median(pos_plot)
    pos_plot = pos_plot/np.max(pos[:,-1])
    
    vel_plot = vel[int(len(t)/2):,i]
    vel_plot = vel[:,i]
    vel_plot = vel_plot - np.median(vel_plot)
    vel_plot = vel_plot/np.max(vel[:,-1])
    
    ax[i].axis([-1.5, 1.5, -1.5, 1.5])
    ax[i].plot(pos_plot, vel_plot, linewidth = 1, color = colors[i], alpha = 1)
    ax[i].set_yticks([])
    ax[i].set_xticks([])
    ax[i].set_aspect('equal', adjustable='box')
    
sns.despine()
fig.text(0.5, 0.05, 'position (au)', ha = 'center')
ax[0].set_ylabel('velocity (au)')
plt.savefig('figures/limit_cycle.svg', format = 'svg')
plt.savefig('figures/limit_cycle.png', format = 'png', dpi = 500)

plt.show()

fig = plt.figure(figsize = (7, 0.1)) 
sns.palplot(colors)
sns.despine()
plt.savefig('figures/kr_range.svg', format = 'svg')
plt.savefig('figures/kr_range.png', format = 'png', dpi = 500)

plt.show()



#%%
"""
Plot LC data for entire matrix
"""

lc_array = loadmat('data/limit_cycle/lc_array.mat')['lc_array']

"""
https://stackoverflow.com/questions/17316880/reading-v-7-3-mat-file-in-python
import hdf5storage
mat = hdf5storage.loadmat('test.mat')
"""

"""
import h5py
arrays = {}

f = h5py.File('data/limit_cycle/lc_array.mat')

for k, v in f.items():
    arrays[k] = np.array(v)
"""
import mat73

#lc_array = mat73.loadmat('data/limit_cycle/lc_array_single.mat')

slice_idx = 5;
lc_array = lc_array[::2,::2, :, :]

colors = sns.color_palette("vlag", len(synch_gain_range))

j_max, i_max, _, _ = lc_array.shape

fig, ax = plt.subplots(10, 10, figsize = (5, 5))
sns.set(font_scale = 1, style = 'ticks')

power_cmap = 'reds'
for j in range(j_max):
    ax_j = 9 - j
    for i in range(i_max):
        
        pos = np.squeeze(lc_array[j, i, 0, 90000:])
        vel = np.squeeze(lc_array[j, i, 1, 90000:])
        pos_plot = pos[::20]
        pos_plot = pos_plot - np.median(pos_plot)
        pos_plot = pos_plot/np.max(np.abs(lc_array[:,:,0,:])) # Normalize by the same 
        
        vel_plot = vel[::20]
        vel_plot = vel_plot - np.median(vel_plot)
        vel_plot = vel_plot/np.max(np.abs(lc_array[:,:,1,:]))
        
        ax[ax_j,i].axis([-1.1, 1.1, -1.1, 1.1])
        ax[ax_j,i].plot(pos_plot, vel_plot, linewidth = .5, color ='#262626', alpha = 1)
        ax[ax_j,i].set_yticks([])
        ax[ax_j,i].set_xticks([])
        ax[ax_j,i].set_aspect('equal', adjustable='box')
        ax[ax_j,i].patch.set_alpha(0)
        
    sns.despine()
ax[5,0].set_ylabel('velocity (au)')
ax[9,5].set_xlabel('position (au)')

fig.text(0.5, 0.05, r'$K_r$', ha = 'center')
fig.text(0.02, 0.5, r'$t_o/T_n$',  va = 'center', rotation = 'vertical')
plt.savefig('figures/limit_cycle.svg', format = 'svg')
plt.savefig('figures/limit_cycle.png', format = 'png', dpi = 500)

plt.show()



X, Y = np.meshgrid(synch_gain_range, yax)
fig,ax = plt.subplots(1,1,figsize = (5.3,5))
sns.set(font_scale = 1, style = 'ticks')
pos = ax.pcolor(X, Y, power_df, cmap = 'Reds', edgecolors = 'none')

plt.xlabel(r'$K_r$')
plt.ylabel(r'$t_o/T_n$')
ax.set_yscale('log')
plt.axis([0, 1, 0.01, 1])
#ax.set_yticks([0.01, 0.01, 1])
#ax.set_xticks([0, 0.5, 1])
#ax.set_xticks([])
#ax.set_yticks([])
sns.despine(offset = 20)
plt.tight_layout()
plt.savefig('figures/axes.svg', format = 'svg', transparent = True)
plt.show()

#%%
"""
##############################################################################
plot roboflapper limit cycles
##############################################################################
"""

flapper_data = loadmat('data/20201122_flapper/roboflapperParamSweep_20x20_02to1.mat')
raw_data = flapper_data['raw_data'] # raw_data[20x10][1]

nSynch = 20
synch_gain_range = np.linspace(0, 1, nSynch)
r3_range = [306.564856593887, 249.487438045280, 203.036911777704, 165.234722305919, 134.470689178953, 109.434421506060, 89.0595019887825, 72.4780629835994, 58.9838197671064, 48.0019864094016, 39.0647928253259, 31.7915601548261, 25.8724857852742, 21.0554473404036, 17.1352635531001, 13.9449545900065, 11.3486295623484, 9.23569826722170, 7.51616060904457, 6.11677305455622]
r3_range = np.logspace(-1.7, 0, 20)

print('to do: update r3_range')
t = raw_data[0,0][0][0][0]

r3_range = r3_range[::]
raw_data = raw_data[::,::]


max_pos = 0
max_vel = 0

freq_total = []
for j in range(len(r3_range)):
    ax_j = len(r3_range) - j - 1
    freq = []
    for i in range(len(synch_gain_range)):
        pos = np.squeeze(raw_data[j, i][0][0][1][0][0][0]) # position
        vel = np.squeeze(raw_data[j, i][0][0][1][0][1][0]) # filtered velocity
        
        # Extract osc frqe
        pos_fft = pos[30000:]
        pos_fft = pos_fft - np.mean(pos_fft)
        pos_freq, pos_amp = fourier_analysis_plot(pos_fft, 1000)
        max_ind = np.where(pos_amp == np.max(pos_amp))[0][0]
        freq.append(pos_freq[max_ind])
        
        max_pos = max(max_pos, np.max(pos))
        max_vel = max(max_vel, np.max(vel))        
    freq_total.append(freq)

f_n = 2.3908 *1.56

freq_total = np.squeeze(np.array(freq_total))/f_n# dimensions are: synch_gain_range (x) by r3_range (y)
#plt.pcolor(freq_total)


#fig, ax = plt.subplots(len(r3_range), nSynch, figsize = (3, 3))

sns.set(font_scale = 1, style = 'ticks')
raw_data_plot = raw_data[::2, ::2]

x, y = raw_data_plot.shape
fig, ax = plt.subplots(x, y, figsize = (5, 5))

for j in range(x):
    ax_j = x - j - 1
    for i in range(y):
        pos = np.squeeze(raw_data_plot[j, i][0][0][1][0][0][0]) # position
        vel = np.squeeze(raw_data_plot[j, i][0][0][1][0][1][0]) # filtered velocity
        pos_plot = pos[-2000:]
        pos_plot = pos_plot - np.median(pos_plot)
        pos_plot = pos_plot/max_pos # Normalize by the same 
        
        vel_plot = vel[-2000:]
        vel_plot = vel_plot - np.median(vel_plot)
        vel_plot = vel_plot/max_vel
        
        ax[ax_j,i].axis([-1.1, 1.1, -1.1, 1.1])
        ax[ax_j,i].plot(pos_plot, vel_plot, linewidth = .5, color ='#262626', alpha = 1)
        ax[ax_j,i].set_yticks([])
        ax[ax_j,i].set_xticks([])
        ax[ax_j,i].set_aspect('equal', adjustable='box')
        ax[ax_j,i].patch.set_alpha(0)

    sns.despine()
ax[9,0].set_ylabel(r'$\hat{\dot{x}}$')
ax[9,0].set_xlabel(r'$\hat{x}$')

#fig.text(0.5, 0.05, r'$K_r$', ha = 'center')
#fig.text(0.02, 0.5, r'$t_o/T_n$',  va = 'center', rotation = 'vertical')
plt.savefig('figures/flapper_limit_cycle.svg', format = 'svg', rasterized = True)
plt.savefig('figures/flapper_limit_cycle.png', format = 'png', dpi = 500)

plt.show()

T_n = 1/f_n
r4_ratio = 0.62

r3_range = np.array(r3_range)
t_o = np.log(1/r4_ratio)/((1-r4_ratio) * r3_range)

yax = t_o/T_n

X, Y = np.meshgrid(synch_gain_range, yax)
fig,ax = plt.subplots(1,1,figsize = (5.3,5))
sns.set(font_scale = 1, style = 'ticks')
pos = ax.pcolor(X, Y, power, cmap = 'Reds', edgecolors = 'none')

plt.xlabel(r'$K_r$')
plt.ylabel(r'$t_o/T_n$')
ax.set_yscale('log')
#plt.axis([0, 1, 0.01, 1])
#ax.set_yticks([0.01, 0.01, 1])
#ax.set_xticks([0, 0.5, 1])
#ax.set_xticks([])
#ax.set_yticks([])
sns.despine(offset = 5)
plt.tight_layout()
plt.savefig('figures/flapper_axes.svg', format = 'svg', transparent = True)
plt.show()

#%%
"""
##############################################################################
plot roboflapper heat maps
##############################################################################
"""

from scipy.io import loadmat
synch_gain_range = loadmat('data/20201122_flapper/synch_gain_range.mat')['synch_gain_range'][0].flatten()
power = loadmat('data/20201122_flapper/power.mat')['conv_array']
freq = loadmat('data/20201122_flapper/freq.mat')['freq_array']
osc_amp = loadmat('data/20201122_flapper/osc_amp.mat')['est_amp_array']
r3_range = loadmat('data/20201122_flapper/r3_range.mat')['r3_range']

#power_df = pd.DataFrame(data = power, columns = synch_gain_range, index = yax)
#power_df = power_df.rename_axis('yax')
#power_df = power_df.rename_axis('synch_gain_range', axis = 'columns')

f_n = 2.3908
T_n = 1/f_n
r4_ratio = 0.62

r3_range = np.array(r3_range)
t_o = np.log(1/r4_ratio)/((1-r4_ratio) * r3_range)

yax = t_o/T_n
yax = np.squeeze(yax)

yax = np.logspace(-1.7, 0, 20)
#yax = r3_range


from matplotlib import colors

cmap = colors.ListedColormap(['red', 'blue'])

class MidpointNormalize(colors.Normalize):
	"""
	Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)

	e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
	"""
	def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
		self.midpoint = midpoint
		colors.Normalize.__init__(self, vmin, vmax, clip)

	def __call__(self, value, clip=None):
		# I'm ignoring masked values and all kinds of edge cases to make a
		# simple example...
		x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
		return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))

X, Y = np.meshgrid(synch_gain_range, yax)
fig, ax = plt.subplots(2, 1, figsize = (3,5))
sns.set(font_scale = 1, style = 'ticks')
#pos = ax[0].imshow(power_df[::-1], cmap = 'Reds', extent = [np.min(synch_gain_range), np.max(synch_gain_range),0.01, 1])

pos = ax[1].pcolor(X, Y, power, cmap = 'Reds', edgecolors = 'face', linewidth = 1)
ax[1].set_yscale('log')
cbar = fig.colorbar(pos, ax = ax[1])#,fraction=0.046, pad=0.04) 
cbar.ax.set_ylabel('power (au)')
#ax[1].set_ylabel(r'$t_o/T_n$')

ax[1].axis('equal')

#freqcmap = colors.LinearSegmentedColormap.from_list('freqcmap', ['black', 'blue', 'red'])
#fig, ax = plt.subplots(1, 1, figsize = (3,3))
sns.set(font_scale = 1, style = 'ticks')
pos = ax[0].pcolor(X,Y, freq_total, cmap = freqcmap, norm=MidpointNormalize(midpoint=1,vmin=0, vmax=4), vmin = 0, vmax = 4, edgecolors = 'face', linewidth = 1)
#ax.set_ysclale('log')
cbar = fig.colorbar(pos, ax = ax[0])#,fraction=0.046, pad=0.04) 
cbar.ax.set_ylabel(r'$f/f_s$')
cbar.set_ticks([0, 1, 2, 3, 4])
#pos.set_clim(0,4)
ax[0].set_ylabel(r'$t_o/T_n$')
ax[1].set_ylabel(r'$t_o/T_n$')

ax[1].set_xlabel(r'$K_r$')
#ax[0].set_xlabel(r'$K_r$')
ax[0].set_xticks([])

ax[0].set_yscale('log')
ax[0].axis('equal')

#ax[0].scatter(1, 0.55)
sns.despine()
#plt.tight_layout()
plt.savefig('figures/flapper_osc_meaurements.svg', format = 'svg', transparent = True)
plt.savefig('figures/flapper_osc_measurements.png', format = 'png', dpi = 500)

plt.show()

#%% Plot freq alone




freqcmap = colors.LinearSegmentedColormap.from_list('freqcmap', ['black', 'white', 'red','#FF272D', 'blue'])
freqcmap.set_bad('#8FB0ED')

X, Y = np.meshgrid(synch_gain_range, yax)
fig, ax = plt.subplots(1, 1, figsize = (6.23,5))
sns.set(font_scale = 1, style = 'ticks')
#pos = ax[0].imshow(power_df[::-1], cmap = 'Reds', extent = [np.min(synch_gain_range), np.max(synch_gain_range),0.01, 1])

test_freq_total = np.ma.masked_values(freq_total, 1, atol = 1E-2, copy = True)

sns.set(font_scale = 1, style = 'ticks')
pos = ax.pcolormesh(X,Y, test_freq_total, cmap = freqcmap, vmin = 0, vmax = 4, edgecolors = 'face', linewidth = 1)
#ax.set_ysclale('log')
cbar = fig.colorbar(pos, ax = ax)#,fraction=0.046, pad=0.04) 
cbar.ax.set_ylabel(r'$f/f_s$')
cbar.set_ticks([0, 1, 2, 3, 4])
#pos.set_clim(0,4)
ax.set_ylabel(r'$t_o/T_n$')

#ax[0].set_xlabel(r'$K_r$')
ax.set_xticks([])

ax.set_yscale('log')
ax.axis('equal')
plt.axis('off')
#ax[0].scatter(1, 0.55)
sns.despine()
#plt.tight_layout()
plt.savefig('figures/flapper_freq.svg', format = 'svg', transparent = True)
plt.savefig('figures/flapper_freq.png', format = 'png', dpi = 500)

plt.show()

#%% 

"""
==========
Plot power alone for roboflapper
=========
"""

# Extract t_o/T_n for peak asynch power

asynch_power = power[:,0]
asynch_index = np.where(asynch_power == np.max(asynch_power))[0][0] - 1# Add one to align with 10x 10 grid.
#asynch_slice = find_nearest(yax,)
arnold_index = 11

from scipy.io import loadmat
synch_gain_range = loadmat('data/20201122_flapper/synch_gain_range.mat')['synch_gain_range'][0].flatten()
power = loadmat('data/20201122_flapper/power.mat')['conv_array']
freq = loadmat('data/20201122_flapper/freq.mat')['freq_array']
osc_amp = loadmat('data/20201122_flapper/osc_amp.mat')['est_amp_array']
r3_range = loadmat('data/20201122_flapper/r3_range.mat')['r3_range']

power_cmap = colors.LinearSegmentedColormap.from_list('freqcmap', ['white', '#6018D1'])
#freqcmap.set_bad('#8FB0ED')



X, Y = np.meshgrid(synch_gain_range, yax)
fig, ax = plt.subplots(1, 1, figsize = (6,4.8))
sns.set(font_scale = 1, style = 'ticks')

pos = ax.pcolor(X, Y, power, cmap = power_cmap, edgecolors = 'face', linewidth = .1)

freq_smoothed = scipy.ndimage.filters.gaussian_filter(freq_total, sigma = .1)
power_smoothed = scipy.ndimage.filters.gaussian_filter(power, sigma = 0.8)


cs = ax.contourf(X, Y, power, levels = [np.min(power), np.max(power)*0.1], colors = '#B3B3B3', alpha = 0.4)

cs = ax.contour(X, Y, freq_smoothed, levels = [.99, 1.01], colors = '#808080', linewidths = 1.5, linestyles = '-', alpha = 1)
#cs = ax.contourf(X, Y, power, levels = [np.min(power), np.max(power[:,0])*0.1], colors = '#B3B3B3', alpha = 0.4)


ax.plot([0, 1], [yax[arnold_index], yax[arnold_index]], c = 'k', linewidth = 2, solid_capstyle = 'butt', alpha = 1, linestyle = '--')
ax.scatter(synch_gain_range[18], yax[arnold_index], c = '#0071BC', s = 20, zorder = 10)
ax.scatter(synch_gain_range[2], yax[arnold_index], c = 'k', s = 20, zorder = 10)
ax.scatter(synch_gain_range[0], yax[arnold_index], c = '#C1272D', s = 20, zorder = 10)

ax.plot([0, 1], [yax[asynch_index], yax[asynch_index]], c = 'k', linewidth = 2, solid_capstyle = 'butt', alpha = 1)
ax.scatter(synch_gain_range[18], yax[asynch_index], c = '#0071BC', s = 20, zorder = 10)
ax.scatter(synch_gain_range[2], yax[asynch_index], c = 'k', s = 20, zorder = 10)
ax.scatter(synch_gain_range[0], yax[asynch_index], c = '#C1272D', s = 20, zorder = 10)

"""
ax.plot([0, 1], [yax[moth_slice], yax[moth_slice]], c = '#0071BC', linewidth = 2)
ax.plot([0, 1], [yax[asynch_slice], yax[asynch_slice]], c = '#C1272D', linewidth = 2)
ax.scatter(0.9, yax[moth_slice], c = '#0071BC', s = 20)
ax.scatter(0.05, yax[asynch_slice], c = '#C1272D', s = 20)
"""
ax.set_yscale('log')
cbar = fig.colorbar(pos, ax = ax)#,fraction=0.046, pad=0.04) 
cbar.ax.set_ylabel('power (au)')
ax.set_ylabel(r'$t_o/T_n$')
ax.axis('equal')
#fig, ax = plt.subplots(1, 1, figsize = (3,3))
sns.set(font_scale = 1, style = 'ticks')
#pos.set_clim(0,4)
ax.set_ylabel(r'$t_o/T_n$')
ax.set_xlabel(r'$K_r$')
ax.set_xticks([0, 0.5, 1])
ax.axis('equal')
ax.set_xlim([0, 1])
#plt.plot([0, 1], [m_val, m_val], c = '#0071BC', linewidth = 3, solid_capstyle = 'butt')
#plt.plot([0, 1], [yax[asynch_slice], yax[asynch_slice]], c = '#C1272D', linewidth = 3, solid_capstyle = 'butt')
#ax[0].scatter(1, 0.55)
sns.despine()
#plt.tight_layout()
plt.savefig('figures/flapper_power.svg', format = 'svg', transparent = True)
plt.savefig('figures/flapper_power.png', format = 'png', dpi = 500)
plt.show()

#%%
"""
================
Plot select limit time traces from flapper
===============
"""

f_s = 1000

flapper_data = loadmat('data/20201122_flapper/roboflapperParamSweep_20x20_02to1.mat')
raw_data = flapper_data['raw_data'] # raw_data[20x10][1]
k = 0

start_idx = 36000


a_trace = np.squeeze(raw_data[arnold_index+k, 0][0][0][1][0][0][0])[start_idx:]
mid_trace = np.squeeze(raw_data[arnold_index+k, 2][0][0][1][0][0][0])[start_idx:]
s_trace = np.squeeze(raw_data[arnold_index+k, 18][0][0][1][0][0][0])[start_idx:]


f_s = 1000
t = np.linspace(0, len(a_trace)/f_s, len(a_trace))

fig, ax = plt.subplots(2,3, figsize = (7, 3))
sns.set(font_scale = 1, style = 'ticks')
ax[0, 0].plot(t, a_trace, c = '#C1272D', linestyle = '-')#, dashes=(5, 1))
ax[0, 0].set_xticks([])
ax[0,0].set_yticks([-0.5, 0, 0.5])
#ax[0,0].set_ylabel('position (rad)')
ax[0, 1].plot(t, mid_trace, c = 'k', linestyle = '-')
ax[0, 1].set_yticks([])
ax[0, 1].set_xticks([])
ax[0, 2].plot(t, s_trace, c = '#0071BC', linestyle = '-')
ax[0, 2].set_yticks([])
ax[0, 2].set_xticks([])



a_trace = np.squeeze(raw_data[asynch_index+k, 0][0][0][1][0][0][0])[start_idx:]
mid_trace = np.squeeze(raw_data[asynch_index+k, 2][0][0][1][0][0][0])[start_idx:]
s_trace = np.squeeze(raw_data[asynch_index+k, 18][0][0][1][0][0][0])[start_idx:]


ax[1, 0].plot(t, a_trace, c = '#C1272D')
ax[1, 0].set_xlabel('time (s)')
ax[1,0].set_ylabel('position (rad)')

ax[1,0].set_yticks([-0.5, 0, 0.5])
ax[1, 1].plot(t, mid_trace, c = 'k')
ax[1, 1].set_yticks([])
ax[1, 1].set_xlabel('time (s)')
ax[1, 2].plot(t, s_trace, c = '#0071BC')
ax[1, 2].set_yticks([])
ax[1, 2].set_xlabel('time (s)')
plt.ylabel('pos (au)')
sns.despine()
plt.savefig('figures/defense_roboflapper_traces.svg', format = 'svg')
plt.show()



"""
for i in range(20):
    pos =  np.squeeze(raw_data[asynch_index, i][0][0][1][0][0][0])
    plt.plot(pos[32000:])
    plt.title(i)    
    plt.show()
"""
