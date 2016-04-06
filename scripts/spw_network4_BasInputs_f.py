#!/usr/bin/python
# -*- coding: utf8 -*-

from brian import *
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.optimize import minimize, curve_fit
import os
from detect_oscillations import ripple

seed = 12345
random.seed(seed)
np.random.seed(seed)

SWBasePath = '/home/bandi/workspace/KOKI/SharpWaves'  # os.path.split(os.path.split(__file__)[0])[0]

fOut = 'Input_Bas_f4.txt'

A = 12
m = 0.1

first = 50
last = 250
dp = 21
Fs = np.linspace(first, last, dp)


def generate_spike_train(time, rate):

    spikes = []
    for t, r in zip(time, rate):
        for i in range(int(r)):
            if random.random() < 0.8:
                rnd_neurons = random.sample(range(1000), 160)
                rnd_time = (t - random.random()*0.001) * np.ones(len(rnd_neurons))
                z = zip(rnd_neurons, rnd_time)
                spikes.append(z)

    spikes = [item for sublist in spikes for item in sublist]

    return spikes


def opt_rate_func_min(time, params):

    return params[0] * (1 + (params[1] * np.cos(2*np.pi*params[2]*time)))

bounds = [(0, None), (0, 1), (0, None)]

opt_method = 'L-BFGS-B'


def opt_rate_func_cf(time, A, m, f):

    return A * (1 + (m * np.cos(2*np.pi*f*time)))


# ==================================================================================================

NI = 1000

eps_bas = 0.4  # 0.4

z = 1*nS

gL_Bas = 5.0e-3*uS
tauMem_Bas = 14.0*ms
Cm_Bas = tauMem_Bas * gL_Bas
Vrest_Bas = -70.0*mV
reset_Bas = -64.0*mV
theta_Bas = -50.0*mV
tref_Bas = 0.1*ms

J_BasInh = 0.25  # 0.25  # 0.15
J_BasMF = 4.5

print 'J_BasInh', J_BasInh
print 'J_BasMF', J_BasMF

# Synaptic reversal potentials
E_Exc = 0.0*mV
E_Inh = -70.0*mV

# Synaptic time constants
tauSyn_BasExc = 3.0*ms
tauSyn_BasInh = 1.5*ms

# Synaptic delays
delay_BasExc = 3.0*ms
delay_BasInh = 1.5*ms


eqs_bas = '''
dvm/dt = (-gL_Bas*(vm-Vrest_Bas)-(g_ampa*z*(vm-E_Exc)+g_gaba*z*(vm-E_Inh)))/Cm_Bas :volt
dg_ampa/dt = -g_ampa/tauSyn_BasExc : 1
dg_gaba/dt = -g_gaba/tauSyn_BasInh : 1
'''

X = np.zeros((11, dp))
time = np.linspace(0.001, 10, 10000)

for k, f in enumerate(Fs):

    PI = NeuronGroup(NI, model=eqs_bas, threshold=theta_Bas, reset=reset_Bas, refractory=tref_Bas)
    PI.vm = Vrest_Bas
    PI.g_ampa = 0
    PI.g_gaba = 0

    rate = A * (1 + m*np.cos(2*np.pi*f*time))  # Hz
    print 'input rate: %s(1+%scos(2*pi*%s))'%(A, m, f)

    spikes = generate_spike_train(time, rate)

    MF = SpikeGeneratorGroup(NI, spikes)

    Cii = Connection(PI, PI, 'g_gaba', weight=J_BasInh, sparseness=eps_bas, delay=delay_BasInh)
    Cext = IdentityConnection(MF, PI, 'g_ampa', weight=J_BasMF)

    smi = SpikeMonitor(PI)
    popri = PopulationRateMonitor(PI, bin=0.001)

    run(10000*ms, report='text')

    spikes = []

    meanIr, rIAC, maxIAC, tMaxIAC, maxIACR, tMaxIACR, fI, PxxI, avgRippleFI, ripplePI = ripple(popri.rate, 1000)

    print 'Mean inh. rate:', meanIr
    
    params = [A, m, f]
    err = lambda params: np.mean((popri.rate - opt_rate_func_min(time, params))**2)  # MSE for param. optimalization
    params0 = [np.mean(popri.rate), m, avgRippleFI]  # initial geuss for optimalzation

    popt_min = minimize(err, x0=params0, bounds=bounds, method=opt_method)

    A_opt_min, m_opt_min, f_opt_min = popt_min.x[0], popt_min.x[1], popt_min.x[2]

    print 'output rate: %s(1+%scos(2*pi*%s)) .minimize'%(A_opt_min, m_opt_min, f_opt_min)


    popt_cf, pcov_cf = curve_fit(opt_rate_func_cf, time, popri.rate, p0=params0)

    A_opt_cf, m_opt_cf, f_opt_cf = popt_cf[0], popt_cf[1], popt_cf[2]

    print 'output rate: %s(1+%scos(2*pi*%s)) .curve_fit'%(A_opt_cf, m_opt_cf, f_opt_cf)


    X[:, k] = [f, meanIr, maxIAC, avgRippleFI, ripplePI, A_opt_min, A_opt_cf, m_opt_min, m_opt_cf, f_opt_min, f_opt_cf]

# =============================================================================================================

    fig = plt.figure(figsize=(10, 8))

    ax = fig.add_subplot(3, 1, 1)
    ax.plot(np.linspace(0, 10000, len(popri.rate)), popri.rate, 'g-')
    ax.set_title('Bas. population rate')
    ax.set_xlabel('Time [ms]')

    rIACPlot = rIAC[2:201]  # 500 - 5 Hz interval

    ax2 = fig.add_subplot(3, 1, 2)
    ax2.plot(np.linspace(2, 200, len(rIACPlot)), rIACPlot, 'g-')
    ax2.set_title('Autocorrelogram 2-200 ms')
    ax2.set_xlabel('Time [ms]')
    ax2.set_xlim([2, 200])
    ax2.set_ylabel('AutoCorrelation')

    PxxIPlot = 10 * np.log10(PxxI / max(PxxI))
    fI = np.asarray(fI)

    rippleS = np.where(145 < fI)[0][0]
    rippleE = np.where(fI < 250)[0][-1]
    fI.tolist()

    PxxRipple = PxxI[rippleS:rippleE]
    fRipple = fI[rippleS:rippleE]

    PxxRipplePlot = 10 * np.log10(PxxRipple / max(PxxI))

    ax3 = fig.add_subplot(3, 1, 3)
    ax3.plot(fI, PxxIPlot, 'g-', marker='o', linewidth=1.5)
    ax3.plot(fRipple, PxxRipplePlot, 'r-', marker='o', linewidth=2)
    ax3.set_title('Power Spectrum Density')
    ax3.set_xlim([0, 500])
    ax3.set_xlabel('Frequency [Hz]')
    ax3.set_ylabel('PSD [dB]')

    fig.tight_layout()

    figName = os.path.join(SWBasePath, 'figures', '%s(1+%scos(2pi%s))Hz_bas.png'%(A, m, f))
    fig.savefig(figName)


    fig3 = plt.figure(figsize=(10, 8))

    spikes = smi.spikes
    spikingNeurons = [i[0] for i in spikes]
    spikeTimes = [i[1] for i in spikes]

    tmp = np.asarray(spikeTimes)
    ROI = np.where(tmp > 9.9)[0].tolist()

    rasterX = np.asarray(spikeTimes)[ROI] * 1000
    rasterY = np.asarray(spikingNeurons)[ROI]

    ax = fig3.add_subplot(2, 1, 1)
    ax.scatter(rasterX, rasterY, c='green', marker='.', lw=0)
    ax.set_title('Bas. raster plot (last 100 ms)')
    ax.set_xlim([9900, 10000])
    ax.set_xlabel('Time [ms]')
    ax.set_ylim([0, 1000])
    ax.set_ylabel('Neuron number')

    ax2 = fig3.add_subplot(2, 1, 2)
    ax2.plot(np.linspace(9900, 10000, len(popri.rate[9900:10000])), popri.rate[9900:10000], 'g-', linewidth=1.5)
    ax2.set_title('Bas. population rate (last 100 ms)')
    ax2.set_xlabel('Time [ms]')

    fig3.tight_layout()

    figName = os.path.join(SWBasePath, 'figures', '%s(1+%scos(2pi%s))Hz_bas2.png'%(A, m, f))
    fig3.savefig(figName)

    plt.close('all')

# =============================================================================================================

    # Reinitialize variables
    reinit(states=True)

# Plots
fig = plt.figure(figsize=(10, 12))

ax = fig.add_subplot(3, 1, 1)
ax.plot(Fs, X[1, :], 'g-', linewidth=2.5, marker='|', label='mean firing rate')
ax.plot(Fs, X[5, :], 'm-', linewidth=1.5, marker='|', label='l-bfgs-b (with bounds)')
ax.plot(Fs, X[6, :], 'b-', linewidth=1.5, marker='|', label='leastsq (without bounds)')
ax.set_ylabel('mean rate [Hz]')
ax.set_xlim(first, last)
ax.set_title('Mean firing rates')
ax.legend()

ax2 = fig.add_subplot(3, 1, 2)
ax2.plot([first, last], [0, 0], 'k-')
ax2.plot([first, last], [m, m], 'k--')
ax2.plot(Fs, X[7, :], 'm-', linewidth=1.5, marker='o', label='l-bfgs-b (with bounds)')
ax2.plot(Fs, X[8, :], 'b-', linewidth=1.5, marker='o', label='leastsq (without bounds)')
ax2.set_xlim([first, last])
ax2.set_ylabel('output m')
ax2.set_title('input-output m')
ax2.legend()

ax3 = fig.add_subplot(3, 1, 3)
ax3.plot(Fs, X[3, :], 'g-', linewidth=2.5, marker='o', label='ripple freq')
ax3.plot(Fs, X[9, :], 'm-', linewidth=1.5, marker='o', label='l-bfgs-b (with bounds)')
ax3.plot(Fs, X[10, :], 'b-', linewidth=1.5, marker='o', label='leastsq (without bounds)')
ax3.set_xlim(first, last)
ax3.set_ylabel('freq [Hz]')
ax3.set_ylim([np.nanmin(X[3, :])-5, np.nanmax(X[3, :])+8])
ax3.set_title('Ripple oscillation')
ax3.set_xlabel('input f: (A(1+mcos(2pif))Hz)')
ax3.legend()

fig.tight_layout()

fig.savefig(os.path.join(SWBasePath, 'figures', 'A_m_f.png'))


# Save result array (X)
fName= os.path.join(SWBasePath, 'files', fOut)
header = 'f, meanIr, maxIAC, avgRippleFI, ripplePI, A_opt_min, A_opt_fc, m_opt_min, m_opt_cf, f_opt_min, f_opt_cf'
np.savetxt(fName, X, fmt='%.6f', delimiter='\t', header=header)
