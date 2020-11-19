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

fig, ax = plt.subplots(1, 1, figsize = (2, 2))
#pos_plot = pos[int(len(t)/2):]
pos_plot = pos
pos_plot = pos_plot - np.median(pos_plot)
#pos_plot = pos_plot/np.max(np.abs(pos_plot))

#vel_plot = vel[int(len(t)/2):]
vel_plot = vel
vel_plot = vel_plot - np.median(vel_plot)
#vel_plot = vel_plot/np.max(np.abs(vel_plot))

#ax.axis([-1, 1, -1, 1])
ax.plot(pos_plot, vel_plot, linewidth = 1, color = colors[0], alpha = 1)
#ax.set_yticks([])
#ax.set_xticks([])
#ax.set_aspect('equal', adjustable='box')
    
sns.despine()
#fig.text(0.5, 0.05, 'position (au)', ha = 'center')
ax.set_ylabel('velocity (V)')
ax.set_xlabel('position (V)')
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


fig, ax = plt.subplots(1,1,figsize = (2,2))
sns.set(font_scale = 1, style = 'ticks')

plt.plot(x[:500], y_pos[:500])

plt.xlabel('f (Hz)')
plt.ylabel('position (V)')
sns.despine()
plt.savefig('figures/robobee_fft.svg', format = 'svg', transparent = True)
plt.savefig('figures/robobee_fft.png', format = 'png', dpi = 500)
plt.show()


#plt.plot(pos_plot[3000:4000])
#plt.plot(vel_plot[3000:4000])


