# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 14:42:17 2020

@author: Administrator
"""

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


sns.set(font_scale = 1, style = 'ticks')
fig, ax = plt.subplots(1,1, figsize = (3,.75))

bee = 20
locust = 1 #Josephson2000
agrotis = 30/30 # Roeder1951                                                                # Lepidoptera
periplanta = 28/28 # Roeder1951

calliphora = 138/11 # Roeder1951
lucilia = 156/10 # Roeder 1951


vespa = 119/22 # Roeder1951                                                             # Hymenoptera
c_mutabilis = (76/21) # Josephson2000
d_melanogaster = 40 # Gordon2006


plt.plot([0, 0.1], [1, 1], c = '#0071BC', linewidth = 5, solid_capstyle = 'butt')
#plt.plot([0.9, 1], [calliphora, calliphora], c = '#C1272D', linewidth = 3)              # Diptera
#plt.plot([0.9, 1], [lucilia, lucilia], c = '#C1272D', linewidth = 2, solid_capstyle = 'butt')                    # Diptera
#plt.plot([0.9, 1], [vespa, vespa], c = '#C1272D', linewidth = 2, solid_capstyle = 'butt')                        # Hymenoptera
plt.plot([0.9, 1], [c_mutabilis, c_mutabilis], c = '#C1272D', linewidth = 5, solid_capstyle = 'butt')            # Coleoptera
#plt.plot([0.9, 1], [d_melanogaster, d_melanogaster], c = '#C1272D', linewidth = 3)      # Diptera

#plt.fill_between([0.1, 0.9], 0, 4, color = 'k', alpha = 0.2, linewidth = 0)

plt.plot([0.1, 0.9], [1, c_mutabilis], linewidth = 5, solid_capstyle = 'butt', c = 'k', alpha = 1, linestyle = 'dashed')

plt.plot([0.1, 0.5, 0.5, 0.9], [1, 1, c_mutabilis, c_mutabilis], linewidth = 5, solid_capstyle = 'butt', c = 'k', alpha = 1, linestyle = 'dashed')


ax.set_xlim(0, 1)
ax.set_ylim(0, 4.1)
ax.set_yticks([0, 1, 4])
ax.set_ylabel('$f_s/f$')
#ax.set_xlabel('$K_r$')
ax.set_xticks([])
sns.despine()
plt.savefig('figures/summary_fig.svg', format = 'svg')
plt.show()


#%% Plot resonance response

import scipy.integrate as igr

def v2_damper(t, z, k, c, w, I, F0):
    F = F0 * np.sin(w*t)
    #F = F0 * signal.square(2*np.pi*w*t)
    x1, x2 = z
    return [x2, F/I -k*x1/I - c*np.abs(x2)*x2/I]

def simulate_one_freq(I_total, c, k, T, w, F_amp):
    # Function to loop over frequency sweep
    F0 = F_amp/T
    k_eff = k/(T**2)
    SAMPLING_F = 10000
    t0 = 0
    tf = 5
    t_desired = np.linspace(0, tf, tf* SAMPLING_F)
    x0 = [1, 0]

    f = w/(2*np.pi)
    period = 1/f
    t_fft = 10 * period
    tf = t_fft*10
    t_span = [t0, tf]
    t_desired = np.linspace(0, tf, int(tf* SAMPLING_F))

    sol = igr.solve_ivp(v2_damper, t_span, x0, method = 'DOP853', args = (k_eff, c, w, I_total, F0), dense_output = True, t_eval = t_desired)

    t = sol.t
    y = sol.y

    return t, y

I_total = 5.6896692627282575e-08
k =  3760.869565217391
c = 3.6914234134857026e-08
T = 2225.610979128276


w = 2 * np.pi * 25
F0 = 0
t, y = simulate_one_freq(I_total, c, k, T, w, F0)

fig, ax = plt.subplots(1, 1, figsize = (3,1))
sns.set(font_scale = 1, style = 'ticks')
plt.axis([0, 0.3, -1.1, 1.1])

plt.plot(t, y[0,:]/np.max(y[0,:]), c = 'k')
#plt.plot(t, np.sin(w*t), c = '#0071BC')
ax.set_xticks([])
ax.set_yticks([])
ax.set_ylabel('$\phi$')
ax.set_xlabel('$t$')
sns.despine()
plt.savefig('figures/res_decay.svg', format = 'svg')
plt.show()


#%%% Forced oscillations

w = 2 * np.pi * 25
F0 = 1
t, y = simulate_one_freq(I_total, c, k, T, w, F0)

fig, ax = plt.subplots(1, 1, figsize = (3,1))
sns.set(font_scale = 1, style = 'ticks')
plt.axis([.5, .8, -1.1, 1.1])

plt.plot(t, y[0,:]/np.max(y[0,1000:]), c = 'k')
plt.plot(t, np.sin(w*t), c = '#0071BC')
ax.set_xticks([])
ax.set_yticks([])
ax.set_ylabel('$\phi$, $F_o$')
ax.set_xlabel('$t$')
sns.despine()
plt.savefig('figures/synch_osc.svg', format = 'svg')
plt.show()

#%% Plot sum of sines

t = np.linspace(0, 10, 1000)
A1 = 1
A2 = .5
A3 = 0.3
w1 = 10
w2 = 25
w3 = 39

x = A1*np.sin(w1*t) + A2*np.sin(w2*t) + A3 *np.sin(w3*t)
fig, ax = plt.subplots(1,1, figsize = (5,1))
plt.plot(t,x, linewidth = 2, c = 'k')
sns.despine()
plt.axis('off')
plt.savefig('figures/sum_of_sines.svg', format = 'svg')