import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.integrate import trapezoid as trap
from scipy.interpolate import PchipInterpolator
from utils import *


### Basic conversion parameters
conv_par = {
    'deg2rad': np.pi/180.,
    'day2yr':  1. / 365.
}

### Simulation parameters
sim_par = {
    'loge_min': 17.000, # log_10(E/EeV)
    'loge_max': 20.001,
    'zen_min':  65. - 0.01,
    'zen_max':  88. + 0.01,
    'azi_min':  0.,
    'azi_max':  360
}
sim_par['cosz_min'] = np.cos(sim_par['zen_max']*conv_par['deg2rad'])
sim_par['cosz_max'] = np.cos(sim_par['zen_min']*conv_par['deg2rad'])

### Observation parameters with which you want to calculate the exposure
obs_par = {
    'S_geo': 180, # km^2
    'T_obs': 1 # day
}



### Read output data of judge_trigger_*.py
### output "data" is a dictionary, and has the following keys:
### 'run':  Run number
### 'eve':  Event number
### 'ped':  primary particle type (proton == 14, iron == 56)
### 'e_eV': primary energy (GeV)
### 'zen':  zenith angle (deg.)
### 'azi':  azimuth angle (deg.)
### 'corex': X coordinate of a shower core position (m)
### 'corey': Y coordinate of a shower core position (m)
### 'corez': Z coordinate of a shower core position (m)
### 'trig':  trigger flag (1 = triggered, 0 = NOT triggered)
data = read_data_trigger_information('out_judge_trigger/sim_*')



### Calculate weighting factors of events as a function of energy and zenith
### Now MC adopts a uniform distribution in energy and zenith angle (NOT cosine of zenith!).
### Below we make weighting factors to modify
### 1. the energy distribution to that of CRs observed by Pierre Auger Observatory, &
### 2. the zenith distribution to the uniforma distribution in cos(zenith).
w_e = calculate_weighting_factor_energy_PAO_spectrum(
    data['e_eV'],
    10 ** sim_par['loge_min'],
    10 ** sim_par['loge_max']
)
w_cosz = np.sin(data['zen']*conv_par['deg2rad'])



### Bin events
nbin_loge, nbin_cosz, nbin_azi = 30, 10, 12
data_loge, data_cosz = np.log10(data['e_eV']), np.cos(data['zen']*conv_par['deg2rad'])
idx_trig = np.where(data['trig'] == 1)

N_sim_all, loge_edges, cosz_edges = np.histogram2d(
    data_loge,
    data_cosz,
    bins=(nbin_loge, nbin_cosz),
    range=[(sim_par['loge_min'], sim_par['loge_max']),
           (sim_par['cosz_min'], sim_par['cosz_max'])]
)

N_w_all, loge_edges, cosz_edges = np.histogram2d(
    data_loge,
    data_cosz,
    bins=(nbin_loge, nbin_cosz),
    weights=w_e*w_cosz,
    range=[(sim_par['loge_min'], sim_par['loge_max']),
           (sim_par['cosz_min'], sim_par['cosz_max'])]
)

N_sim_trig, loge_edges, cosz_edges = np.histogram2d(
    data_loge[idx_trig],
    data_cosz[idx_trig],
    bins=(nbin_loge, nbin_cosz),
    range=[(sim_par['loge_min'], sim_par['loge_max']),
           (sim_par['cosz_min'], sim_par['cosz_max'])]
)

N_w_trig, loge_edges, cosz_edges = np.histogram2d(
    data_loge[idx_trig],
    data_cosz[idx_trig],
    bins=(nbin_loge, nbin_cosz),
    weights=w_e[idx_trig]*w_cosz[idx_trig]*data_cosz[idx_trig],
    range=[(sim_par['loge_min'], sim_par['loge_max']),
           (sim_par['cosz_min'], sim_par['cosz_max'])]
)

### Normalization w.r.t. the cos(theta) space
norm_coeff_e, norm_coeff_cosz = data['run'].shape[0] / np.sum(w_e), data['run'].shape[0] / np.sum(w_cosz)
N_w_all  *= norm_coeff_e * norm_coeff_cosz
N_w_trig *= norm_coeff_e * norm_coeff_cosz



### Exposure as a function of energy & cos(zenith)
### Error is calculated following the binomial distribution
omega_cosz = 2.* np.pi * (sim_par['cosz_max'] - sim_par['cosz_min']) / nbin_cosz
exp_loge_cosz = obs_par['S_geo'] * obs_par['T_obs'] * omega_cosz * N_w_trig / N_w_all
exp_loge_cosz_err = exp_loge_cosz * np.sqrt(1. / N_sim_trig - 1. / N_sim_all)

J = np.array([
    integrate.quad(
        calculate_PAO_spectrum,
        10 ** loge_edges[n],
        10 ** loge_edges[n+1]
    )[0]
    for n in range(len(loge_edges)-1)
])

Neve_loge_cosz = J.reshape(-1,1) * exp_loge_cosz * conv_par['day2yr'] 
Neve_loge_cosz_err = J.reshape(-1,1) * exp_loge_cosz_err * conv_par['day2yr']
print("np.sum(Neve_loge_cosz):", np.sum(Neve_loge_cosz))

### Exposure as a function of energy (integrated along zenith)
omega_tot = 2. * np.pi * (sim_par['cosz_max'] - sim_par['cosz_min'])
exp_loge = obs_par['S_geo'] * obs_par['T_obs'] * omega_tot * np.sum(N_w_trig, axis=1) / np.sum(N_w_all, axis=1)
exp_loge_err = exp_loge * np.sqrt(1. / np.sum(N_sim_trig, axis=1) - 1. / np.sum(N_sim_all, axis=1))
Neve_loge = J * exp_loge * conv_par['day2yr']
Neve_loge_err = J * exp_loge_err * conv_par['day2yr']
print("np.sum(Neve_loge):", np.sum(Neve_loge))

### Exposure as a function of cos(Zenith) (integrated along energy)
exp_cosz = obs_par['S_geo'] * obs_par['T_obs'] * omega_cosz * np.sum(N_w_trig, axis=0) / np.sum(N_w_all, axis=0)
exp_cosz_err = exp_cosz * np.sqrt(1. / np.sum(N_sim_trig, axis=0) - 1. / np.sum(N_sim_all, axis=0))
J = integrate.quad(calculate_PAO_spectrum, 10 ** sim_par['loge_min'], 10 ** sim_par['loge_max'])[0]
Neve_cosz = J * exp_cosz * conv_par['day2yr']
Neve_cosz_err = J * exp_cosz_err * conv_par['day2yr']
print("np.sum(Neve_cosz):", np.sum(Neve_cosz))



### Plot figures to check the weighting scheme
plt_par = {
    "legend.fontsize": 30,
    "axes.labelsize":  30,
    "axes.titlesize":  23,
    "xtick.labelsize": 30,
    "ytick.labelsize": 30,
    "axes.grid": False,
}
plt.rcParams.update(plt_par)

"""
fig = plt.figure(figsize=(80, 20))
ax = fig.add_subplot(1,2,1)
norm = colors.LogNorm(vmin=0.1, vmax=20.)
c = ax.pcolor(loge_edges, cosz_edges, exp_loge_cosz.T, norm=norm, shading='auto', cmap='viridis')
ax.set_xlabel('log10(E_{CR} [eV])')
ax.set_ylabel('cos(Zenith [deg.])')
fig.colorbar(c, ax=ax, label='One-day exposure (km$^2$ day sr)')

ax = fig.add_subplot(1,2,2)
norm = colors.LogNorm(vmin=1e-4, vmax=5.)
c = ax.pcolor(loge_edges, cosz_edges, Neve_loge_cosz.T, norm=norm, shading='auto', cmap='viridis')
ax.set_xlabel('log10(E_{CR} [eV])')
ax.set_ylabel('cos(Zenith [deg.])')
fig.colorbar(c, ax=ax, label='Number of CR events / day / bin')

plt.savefig('calculate_exposure_2d.pdf')
"""

fig = plt.figure(figsize=(30, 20))
x_loge = (loge_edges[:-1] + loge_edges[1:]) / 2
x_cosz = (cosz_edges[:-1] + cosz_edges[1:]) / 2

ax = fig.add_subplot(2,2,1)
ax.set_yscale('log')
ax.plot(x_loge, exp_loge, c='black')
ax.errorbar(x_loge, exp_loge, yerr = exp_loge_err, capsize=5, fmt='o', markersize=7, ecolor='black', markeredgecolor = "black", color='black')
ax.set_title('')
ax.set_xlabel(r'${\rm log}_{10}$(Energy [eV])')
ax.set_ylabel('One-day exposure (km$^2$ day sr)')
ax.set_xlim(16.9, 20.0)
ax.set_ylim(8.e-2,1.5e2)
ax.tick_params(which='both', direction='in')
ax.tick_params(which='major', direction='in', length=9, width=1)
ax.tick_params(which='minor', direction='in', length=5, width=1)
ax.set_xticks([17, 17.5, 18, 18.5, 19.0, 19.5, 20.0])
ax.minorticks_on()

ax = fig.add_subplot(2,2,2)
ax.set_yscale('log')
ax.plot(x_loge, Neve_loge, c='black')
ax.errorbar(x_loge, Neve_loge, yerr = Neve_loge_err, capsize=5, fmt='o', markersize=7, ecolor='black', markeredgecolor = "black", color='black')
ax.set_title('')
ax.set_xlabel(r'${\rm log}_{10}$(Energy [eV])')
ax.set_ylabel('Number of CR events / day / bin')
ax.set_xlim(16.9, 20.0)
ax.set_ylim(1.e-5,5.e1)
ax.tick_params(which='both', direction='in')
ax.tick_params(which='major', direction='in', length=9, width=1)
ax.tick_params(which='minor', direction='in', length=5, width=1)
ax.set_xticks([17, 17.5, 18, 18.5, 19.0, 19.5, 20.0])
ax.minorticks_on()

ax = fig.add_subplot(2,2,3)
ax.set_yscale('log')
ax.plot(x_cosz, exp_cosz, c='black')
ax.errorbar(x_cosz, exp_cosz, yerr = exp_cosz_err, capsize=5, fmt='o', markersize=7, ecolor='black', markeredgecolor = "black", color='black')
ax.set_title('')
ax.set_xlabel(r'cos(Zenith [deg.])')
ax.set_ylabel('Exposure (km$^2$ day sr) / bin')
ax.set_xlim(0, 0.5)
ax.set_ylim(1.e-3,3)
ax.tick_params(which='both', direction='in')
ax.tick_params(which='major', direction='in', length=9, width=1)
ax.tick_params(which='minor', direction='in', length=5, width=1)
ax.minorticks_on()

ax = fig.add_subplot(2,2,4)
ax.set_yscale('log')
ax.plot(x_cosz, Neve_cosz, c='black')
ax.errorbar(x_cosz, Neve_cosz, yerr = Neve_cosz_err, capsize=5, fmt='o', markersize=7, ecolor='black', markeredgecolor = "black", color='black')
ax.set_title('')
ax.set_xlabel(r'cos(Zenith [deg.])')
ax.set_ylabel('Number of CR events / day / bin')
ax.set_xlim(0, 0.5)
ax.set_ylim(1.e-2,3.e1)
ax.tick_params(which='both', direction='in')
ax.tick_params(which='major', direction='in', length=9, width=1)
ax.tick_params(which='minor', direction='in', length=5, width=1)
ax.minorticks_on()

plt.savefig('calculate_exposure.pdf')
