# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 15:19:26 2020

@author: Administrator
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def power(w, r3, alpha = 0.62, A = 1):
    e = w/r3
    coeff = -0.5*A**2*w**2*alpha
    power = coeff*(e**2-alpha)/((e**2+alpha**2) *(e**2+1))
    return power/np.max(power)


r3 = 36.39 # from r3 r4 fit of experimental data

f =  np.linspace(0,250,1000);
w = 2*np.pi*f
e = w/r3



r3 = 150
dros_power = power(w, 1350)
leth_power = power(w, 160)

fig, ax = plt.subplots(1,1,figsize = (3,3))
sns.set(font_scale = 1, style = 'ticks')
plt.plot(f, dros_power, label = 'Drosophila \n $r_3$ = 1350 s$^{-1}$', linewidth = 3, c = '#FFA052')
plt.plot(f, leth_power, label = 'Lethocerus \n $r_3$ = 160 s$^{-1}$', linewidth = 3, c = '#0390A8')
plt.ylim(0, 1.1)
ax.set_yticks([0, 0.5, 1])
plt.ylabel('power (au)')
plt.xlabel(r'$f$ (Hz)')
sns.despine()
plt.legend()
plt.tight_layout()
plt.savefig('figures/conv_power.eps', format = 'eps', transparent = True)
plt.show()


f_dros = f[np.where(dros_power == np.max(dros_power))[0][0]]
f_leth = f[np.where(leth_power == np.max(leth_power))[0][0]]

print('Drosophila peak power: %.2f Hz' % f_dros)
print('Lethocerus peak power: %.2f Hz' % f_leth)