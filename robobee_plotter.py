# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 09:41:51 2020

@author: Administrator
"""

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat

    

"""
plot robobee limit cycle data
"""

path = 'data/20201118_robobee/'
data = loadmat(path + 'negVisc_mu035_cofreq1000.mat')

t = data['sdat'][0][0][0]
pos = data['sdat'][0][0][1][0][0][0]
vel = data['sdat'][0][0][1][0][1][0]
torque = data['sdat'][0][0][1][0][2][0]

pos = np.squeeze(pos)
vel = np.squeeze(vel)

plt.plot(pos)
plt.show()

start_ind = 40000
end_ind = 60000
pos = pos[start_ind:end_ind]
vel = vel[start_ind:end_ind]

#colors = plt.cm.jet(np.linspace(0, 1, len(synch_gain_range)))

colors = sns.color_palette("vlag", 10)


sns.set(font_scale = 1, style = 'ticks')

fig, ax = plt.subplots(1, 1, figsize = (1.5, 1.5))
#pos_plot = pos[int(len(t)/2):]
pos_plot = pos
pos_plot = pos_plot - np.median(pos_plot)
pos_plot /= np.max(np.abs(pos_plot))
#pos_plot = pos_plot/np.max(np.abs(pos_plot))

#vel_plot = vel[int(len(t)/2):]
vel_plot = vel
vel_plot = vel_plot - np.median(vel_plot)
vel_plot /= np.max(np.abs(vel_plot))
#vel_plot = vel_plot/np.max(np.abs(vel_plot))

#ax.axis([-1, 1, -1, 1])
ax.plot(pos_plot, vel_plot, linewidth = 1, color = colors[0], alpha = 1)
plt.xticks([-1, 0, 1])

plt.yticks([-1,0, 1])#ax.set_yticks([])
#ax.set_xticks([])
#ax.set_aspect('equal', adjustable='box')
#plt.xticks([-.25, 0, .25])

sns.despine()
#fig.text(0.5, 0.05, 'position (au)', ha = 'center')
ax.set_ylabel(r'$\hat{\dot{x}}$ (V)')
ax.set_xlabel(r'$\hat{x}$ (V)')
#plt.tight_layout()
plt.savefig('figures/robobee_limit_cycle.svg', format = 'svg', transparent = True)
plt.savefig('figures/robobee_limit_cycle.png', format = 'png', dpi = 500)
plt.show()

#%%

"""
Return single-ended amplitude
"""
from scipy.fftpack import fft

def fourier_analysis_plot(data, sampling_f):
    N = len(data)
    T = 1.0 / sampling_f
    yf = 2 * fft(data)/N
    # nyquist_f = np.squeeze(1/(2*T))
    xf = np.squeeze(np.linspace(0.0, 1.0/(2.0*T), N//2))
    y_real = np.squeeze(np.abs(yf[0:N//2]))
    return (xf, y_real)

x, y_pos = fourier_analysis_plot(pos_plot, 10000)
x, y_vel = fourier_analysis_plot(vel_plot, 10000)


fig, ax = plt.subplots(1,1,figsize = (3,1.5))
sns.set(font_scale = 1, style = 'ticks')
plt.plot([90, 90], [0, np.max(y_pos[:500])], linewidth = 3, c = 'k')

plt.plot(x[:500], y_pos[:500], linewidth = 3)

plt.xlabel('f (Hz)')
plt.ylabel(r'$x$ (V)')
sns.despine()
#plt.tight_layout()

plt.savefig('figures/robobee_fft.svg', format = 'svg', transparent = True)
plt.savefig('figures/robobee_fft.png', format = 'png', dpi = 500)
plt.show()

f_ind = np.where(y_pos == np.max(y_pos))[0][0]

print('robobee oscillation f: %.2f Hz' % (x[f_ind]))
#plt.plot(pos_plot[3000:4000])

#plt.plot(vel_plot[3000:4000])

# Number of oscillations
# 20,000 data points at 10 kHz. so 2 seconds
# Oscillation freq of... 57 Hz.




