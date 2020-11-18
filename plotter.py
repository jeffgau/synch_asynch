# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 09:41:51 2020

@author: Administrator
"""

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
Plot t_o vs r3
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
    
    
"""
plot power freq amp from MATLAB simulation
"""
from scipy.io import loadmat
synch_gain_range = loadmat('data/synch_gain_range.mat')['synch_gain_range'][0].flatten()
yax = loadmat('data/yax.mat')['yax'][0].flatten()
power = loadmat('data/power.mat')['power']
freq = loadmat('data/freq.mat')['freq']
osc_amp = loadmat('data/osc_amp.mat')['osc_amp']

power_df = pd.DataFrame(data = power, columns = synch_gain_range, index = yax)
power_df = power_df.rename_axis('yax')
power_df = power_df.rename_axis('synch_gain_range', axis = 'columns')



fig, ax = plt.subplots(figsize = (5,5))
sns.heatmap(power_df)

ax.set_xticklabels([0, 1])
ax.set_yticklabels([0, 1])
ax
sns.despine()
plt.show()


"""
plot limit cycle data
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


"""
plot robobee limit cycle
"""
print('to do')







