#!/usr/bin/python
# -*- coding: utf8 -*-

from brian import *
import numpy as np
import matplotlib.pyplot as plt
import os
from detect_oscillations import replay, ripple, gamma

pop = 'Pyr'  # ['Pyr', 'Bas'] population which gets the outer input
fOut = 'input_Pyr.txt'

SWBasePath = '/home/bandi/workspace/KOKI/SharpWaves'  # os.path.split(os.path.split(__file__)[0])[0]

# outer input (Poisson group)
J_PopMF = 10

first = 50
last = 100
dataPoints = 3

rates_mf = np.linspace(first, last, dataPoints)  # Hz

NE = 4000
NI = 1000

eps_pyr = 0.16  # 0.16
eps_bas = 0.4  # 0.4

z = 1*nS
gL_Pyr = 4.333e-3 * uS  # 3.3333e-3
tauMem_Pyr = 60.0 * ms
Cm_Pyr = tauMem_Pyr * gL_Pyr
Vrest_Pyr = -70.0 * mV
reset_Pyr = -53.0*mV
theta_Pyr = -50.0*mV
tref_Pyr = 5*ms
# Adaptation parameters for pyr cells
a_Pyr = -0.8*nS  # nS subthreshold adaptation conductance
# moves threshold up
b_Pyr = 0.04*nA  # nA    Spike-triggered adaptation
# -> decreases the slope of the f-I curve
delta_T_Pyr = 2.0*mV  # 0.8    # mV    Slope factor
tau_w_Pyr = 300*ms  # 88 # 144.0   # ms    Adaptation time constant
v_spike_Pyr = theta_Pyr + 10 * delta_T_Pyr

gL_Bas = 5.0e-3*uS
tauMem_Bas = 14.0*ms
Cm_Bas = tauMem_Bas * gL_Bas
Vrest_Bas = -70.0*mV
reset_Bas = -64.0*mV
theta_Bas = -50.0*mV
tref_Bas = 0.1*ms  # 0.1*ms

J_PyrInh = 0.15  # 0.15  # 0.125
J_BasExc = 4.5  # 4.5  # 5.2
J_BasInh = 0.25  # 0.25  # 0.15

print 'J_PyrInh', J_PyrInh
print 'J_BasExc', J_BasExc
print 'J_BasInh', J_BasInh

# Synaptic reversal potentials
E_Exc = 0.0*mV
E_Inh = -70.0*mV

# Synaptic time constants
tauSyn_PyrExc = 10.0*ms
tauSyn_PyrInh = 3.0*ms
tauSyn_BasExc = 3.0*ms
tauSyn_BasInh = 1.5*ms

# Synaptic delays
delay_PyrInh = 1.5*ms
delay_BasExc = 3.0*ms
delay_BasInh = 1.5*ms


# Creating populations
eqs_adexp = '''
dvm/dt=(gL_Pyr*(Vrest_Pyr-vm)+gL_Pyr*delta_T_Pyr*exp((vm- theta_Pyr)/delta_T_Pyr)-w -(g_ampa*z*(vm-E_Exc)+g_gaba*z*(vm-E_Inh)))/Cm_Pyr : volt
dw/dt=(a_Pyr*(vm- Vrest_Pyr )-w)/tau_w_Pyr : amp
dg_ampa/dt = -g_ampa/tauSyn_PyrExc : 1
dg_gaba/dt = -g_gaba/tauSyn_PyrInh : 1
'''

reset_adexp = '''
vm = reset_Pyr
w += b_Pyr
'''

eqs_bas = '''
dvm/dt = (-gL_Bas*(vm-Vrest_Bas)-(g_ampa*z*(vm-E_Exc)+g_gaba*z*(vm-E_Inh)))/Cm_Bas :volt
dg_ampa/dt = -g_ampa/tauSyn_BasExc : 1
dg_gaba/dt = -g_gaba/tauSyn_BasInh : 1
'''


def myresetfunc(P, spikes):
    P.vm_[spikes] = reset_Pyr   # reset voltage
    P.w_[spikes] += b_Pyr  # low pass filter of spikes (adaptation mechanism)


X = np.zeros((20, dataPoints))

assert pop in ['Pyr', 'Bas']

for k in range(0, dataPoints):

    rate_mf = rates_mf[k]
    print 'freq. of outer input:', rate_mf, '[Hz]'

    SCR = SimpleCustomRefractoriness(myresetfunc, tref_Pyr, state='vm')

    if pop == 'Pyr':
        PE = NeuronGroup(NE, model=eqs_adexp, threshold=v_spike_Pyr, reset=SCR)
        PE.vm = Vrest_Pyr
        PE.g_ampa = 0
        PE.g_gaba = 0

    PI = NeuronGroup(NI, model=eqs_bas, threshold=theta_Bas, reset=reset_Bas, refractory=tref_Bas)
    PI.vm = Vrest_Bas
    PI.g_ampa = 0
    PI.g_gaba = 0

    if pop == 'Pyr':
        MF = PoissonGroup(NE, rate_mf*Hz)
        Cext = IdentityConnection(MF, PE, 'g_ampa', weight=J_PopMF)
        Cei = Connection(PE, PI, 'g_ampa', weight=J_BasExc, sparseness=eps_pyr, delay=delay_BasExc)
        Cie = Connection(PI, PE, 'g_gaba', weight=J_PyrInh, sparseness=eps_bas, delay=delay_PyrInh)

        sme = SpikeMonitor(PE)
        popre = PopulationRateMonitor(PE, bin=0.001)
    else:
        MF = PoissonGroup(NI, rate_mf*Hz)
        Cext = IdentityConnection(MF, PI, 'g_ampa', weight=J_PopMF)

    Cii = Connection(PI, PI, 'g_gaba', weight=J_BasInh, sparseness=eps_bas, delay=delay_BasInh)

    smi = SpikeMonitor(PI)
    popri = PopulationRateMonitor(PI, bin=0.001)


    run(10000*ms, report='text')

    if pop == 'Pyr':
        avgReplayInterval = np.nan
        meanEr, rEAC, maxEAC, tMaxEAC, maxEACR, tMaxEACR, fE, PxxE, avgRippleFE, ripplePE = ripple(popre.rate, 1000)
        avgGammaFE, gammaPE = gamma(fE, PxxE)
    else:
        avgReplayInterval = np.nan
        meanEr, rEAC, maxEAC, tMaxEAC, maxEACR, tMaxEACR, fE, PxxE, avgRippleFE, ripplePE, avgGammaFE, gammaPE = \
        np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,

    meanIr, rIAC, maxIAC, tMaxIAC, maxIACR, tMaxIACR, fI, PxxI, avgRippleFI, ripplePI = ripple(popri.rate, 1000)
    avgGammaFI, gammaPI = gamma(fI, PxxI)

    X[:, k] = [rate_mf,
               meanEr, maxEAC, tMaxEAC, maxEACR, tMaxEACR,
               meanIr, maxIAC, tMaxIAC, maxIACR, tMaxIACR,
               avgReplayInterval,
               avgRippleFE, ripplePE, avgGammaFE, gammaPE,
               avgRippleFI, ripplePI, avgGammaFI, gammaPI]


    # Saved figures
    if pop == 'Pyr':

        # Pyr population
        fig2 = plt.figure(figsize=(10, 8))

        ax = fig2.add_subplot(3, 1, 1)
        ax.plot(np.linspace(0, 10000, len(popre.rate)), popre.rate, 'b-')
        ax.set_title('Pyr. population rate')
        ax.set_xlabel('Time [ms]')

        rEACPlot = rEAC[2:201] # 500 - 5 Hz interval

        ax2 = fig2.add_subplot(3, 1, 2)
        ax2.plot(np.linspace(2, 200, len(rEACPlot)), rEACPlot, 'b-')
        ax2.set_title('Autocorrelogram 2-200 ms')
        ax2.set_xlabel('Time [ms]')
        ax2.set_xlim([2, 200])
        ax2.set_ylabel('AutoCorrelation')

        PxxEPlot = 10 * np.log10(PxxE / max(PxxE))

        fE = np.asarray(fE)
        rippleS = np.where(145 < fE)[0][0]
        rippleE = np.where(fE < 250)[0][-1]
        gammaS = np.where(30 < fE)[0][0]
        gammaE = np.where(fE < 80)[0][-1]
        fE.tolist()

        PxxRipple = PxxE[rippleS:rippleE]
        PxxGamma = PxxE[gammaS:gammaE]

        fRipple = fE[rippleS:rippleE]
        fGamma = fE[gammaS:gammaE]

        PxxRipplePlot = 10 * np.log10(PxxRipple / max(PxxE))
        PxxGammaPlot = 10 * np.log10(PxxGamma / max(PxxE))

        ax3 = fig2.add_subplot(3, 1, 3)
        ax3.plot(fE, PxxEPlot, 'b-', marker='o', linewidth=1.5)
        ax3.plot(fRipple, PxxRipplePlot, 'r-', marker='o', linewidth=2)
        ax3.plot(fGamma, PxxGammaPlot, 'k-', marker='o', linewidth=2)
        ax3.set_title('Power Spectrum Density')
        ax3.set_xlim([0, 500])
        ax3.set_xlabel('Frequency [Hz]')
        ax3.set_ylabel('PSD [dB]')

        fig2.tight_layout()

        figName = os.path.join(SWBasePath, 'figures', str(rate_mf)+'Hz_pyr.png')
        fig2.savefig(figName)


    # Bas population
    fig3 = plt.figure(figsize=(10, 8))

    ax = fig3.add_subplot(3, 1, 1)
    ax.plot(np.linspace(0, 10000, len(popri.rate)), popri.rate, 'g-')
    ax.set_title('Bas. population rate')
    ax.set_xlabel('Time [ms]')

    rIACPlot = rIAC[2:201] # 500 - 5 Hz interval

    ax2 = fig3.add_subplot(3, 1, 2)
    ax2.plot(np.linspace(2, 200, len(rIACPlot)), rIACPlot, 'g-')
    ax2.set_title('Autocorrelogram 2-200 ms')
    ax2.set_xlabel('Time [ms]')
    ax2.set_xlim([2, 200])
    ax2.set_ylabel('AutoCorrelation')

    PxxIPlot = 10 * np.log10(PxxI / max(PxxI))

    fI = np.asarray(fI)
    rippleS = np.where(145 < fI)[0][0]
    rippleE = np.where(fI < 250)[0][-1]
    gammaS = np.where(30 < fI)[0][0]
    gammaE = np.where(fI < 80)[0][-1]
    fI.tolist()

    PxxRipple = PxxI[rippleS:rippleE]
    PxxGamma = PxxI[gammaS:gammaE]

    fRipple = fI[rippleS:rippleE]
    fGamma = fI[gammaS:gammaE]

    PxxRipplePlot = 10 * np.log10(PxxRipple / max(PxxI))
    PxxGammaPlot = 10 * np.log10(PxxGamma / max(PxxI))

    ax3 = fig3.add_subplot(3, 1, 3)
    ax3.plot(fI, PxxIPlot, 'g-', marker='o', linewidth=1.5)
    ax3.plot(fRipple, PxxRipplePlot, 'r-', marker='o', linewidth=2)
    ax3.plot(fGamma, PxxGammaPlot, 'k-', marker='o', linewidth=2)
    ax3.set_title('Power Spectrum Density')
    ax3.set_xlim([0, 500])
    ax3.set_xlabel('Frequency [Hz]')
    ax3.set_ylabel('PSD [dB]')

    fig3.tight_layout()

    figName = os.path.join(SWBasePath, 'figures', str(rate_mf)+'Hz_bas.png')
    fig3.savefig(figName)

    if pop == 'Pyr':
        # raster plot and rate pyr (higher resolution)
        fig4 = plt.figure(figsize=(10, 8))

        spikes = sme.spikes
        spikingNeurons = [i[0] for i in spikes]
        spikeTimes = [i[1] for i in spikes]

        tmp = np.asarray(spikeTimes)
        ROI = np.where(tmp > 9.9)[0].tolist()

        rasterX = np.asarray(spikeTimes)[ROI] * 1000
        rasterY = np.asarray(spikingNeurons)[ROI]

        if rasterY.min()-50 > 0:
            ymin = rasterY.min()-50
        else:
            ymin = 0

        if rasterY.max()+50 < 4000:
            ymax = rasterY.max()+50
        else:
            ymax = 4000

        ax = fig4.add_subplot(2, 1, 1)
        ax.scatter(rasterX, rasterY, c='blue', marker='.', lw=0)
        ax.set_title('Raster plot (last 100 ms)')
        ax.set_xlim([9900, 10000])
        ax.set_xlabel('Time [ms]')
        ax.set_ylim([ymin, ymax])
        ax.set_ylabel('Neuron number')

        ax2 = fig4.add_subplot(2, 1, 2)
        ax2.plot(np.linspace(9900, 10000, len(popre.rate[9900:10000])), popre.rate[9900:10000], 'b-', linewidth=1.5)
        ax2.set_title('Pyr. population rate (last 100 ms)')
        ax2.set_xlabel('Time [ms]')

        fig4.tight_layout()

        figName = os.path.join(SWBasePath, 'figures', str(rate_mf)+'Hz_pyr_rate.png')
        fig4.savefig(figName)

    # raster plot and rate bas (higher resolution)
    fig5 = plt.figure(figsize=(10, 8))

    spikes = smi.spikes
    spikingNeurons = [i[0] for i in spikes]
    spikeTimes = [i[1] for i in spikes]

    tmp = np.asarray(spikeTimes)
    ROI = np.where(tmp > 9.9)[0].tolist()

    rasterX = np.asarray(spikeTimes)[ROI] * 1000
    rasterY = np.asarray(spikingNeurons)[ROI]

    if rasterY.min()-50 > 0:
        ymin = rasterY.min()-50
    else:
        ymin = 0

    if rasterY.max()+50 < 1000:
        ymax = rasterY.max()+50
    else:
        ymax = 1000

    ax = fig5.add_subplot(2, 1, 1)
    ax.scatter(rasterX, rasterY, c='green', marker='.', lw=0)
    ax.set_title('Bas. raster plot (last 100 ms)')
    ax.set_xlim([9900, 10000])
    ax.set_xlabel('Time [ms]')
    ax.set_ylim([ymin, ymax])
    ax.set_ylabel('Neuron number')

    ax2 = fig5.add_subplot(2, 1, 2)
    ax2.plot(np.linspace(9900, 10000, len(popri.rate[9900:10000])), popri.rate[9900:10000], 'g-', linewidth=1.5)
    ax2.set_title('Bas. population rate (last 100 ms)')
    ax2.set_xlabel('Time [ms]')

    fig5.tight_layout()

    figName = os.path.join(SWBasePath, 'figures', str(rate_mf)+'Hz_bas_rate.png')
    fig5.savefig(figName)

    plt.close('all')

    print 'Mean exc. rate:', meanEr
    print 'Mean inh. rate:', meanIr

    # Reinitialize variables
    reinit(states=True)


# Plots
fig = plt.figure(figsize=(10, 8))

ax = fig.add_subplot(2, 1, 1)
ax.plot(rates_mf, X[11, :], linewidth=2, marker='|')
ax.set_title('Average replay interval')
ax.set_xlim(first, last)
ax.set_ylabel('Time (ms)')

ax2 = fig.add_subplot(2, 1, 2)
ax3 = ax2.twinx()
ax2.plot(rates_mf, X[1, :], 'b-', linewidth=2, marker='|')
ax2.set_ylabel(ylabel='PC (exc.) rate', color='blue')
ax3.plot(rates_mf, X[6, :], 'g-', linewidth=2, marker='|')
ax3.set_ylabel(ylabel='BC (inh.) rate', color='green')
ax2.set_xlabel('input rate')
ax2.set_xlim(first, last)
ax2.set_title('Mean firing rates')

fig.tight_layout()

fig.savefig(os.path.join(SWBasePath, 'figures', 'replay_and_firing_rates.png'))


fig2 = plt.figure(figsize=(10, 8))

ax = fig2.add_subplot(2, 1, 1)
ax.plot(rates_mf, X[2, :], label='PC (exc.)', color='blue', linewidth=2, marker='|')
ax.plot(rates_mf, X[7, :], label='BC (inh.)', color='green', linewidth=2, marker='|')
ax.set_xlim(first, last)
ax.set_title('Maximum autocerrelations')
ax.legend()

ax2 = fig2.add_subplot(2, 1, 2)
ax2.plot(rates_mf, X[4, :], label='PC (exc.)', color='blue', linewidth=2, marker='|')
ax2.plot(rates_mf, X[9, :], label='BC (inh.)', color='green', linewidth=2, marker='|')
ax2.set_xlim(first, last)
ax2.set_title('Maximum autocerrelations in ripple range')
ax2.set_xlabel('input rate')
ax2.legend()

fig2.tight_layout()

fig2.savefig(os.path.join(SWBasePath, 'figures', 'autocorrelations.png'))

fig3 = plt.figure(figsize=(10, 8))

ax = fig3.add_subplot(2, 1, 1)
ax.plot(rates_mf, X[12, :], label='Ripple freq (exc.)', color='blue', linewidth=2, marker='o')
ax2 = ax.twinx()
ax2.plot(rates_mf, X[13, :], label='Ripple power (exc.)', color='red', linewidth=2, marker='|')
ax.set_xlim(first, last)
ax.set_ylabel(ylabel='freq [Hz]', color='blue')
ax.set_ylim([np.nanmin(X[12, :])-5, np.nanmax(X[12, :])+8])
ax2.set_ylabel(ylabel='power %', color='red')
ax2.set_ylim([0, 100])
ax.set_title('Ripple oscillation')
ax.legend(loc=2)
ax2.legend()

ax3 = fig3.add_subplot(2, 1, 2)
ax3.plot(rates_mf, X[16, :], label='Ripple freq (inh.)', color='green', linewidth=2, marker='o')
ax4 = ax3.twinx()
ax4.plot(rates_mf, X[17, :], label='Ripple power (inh.)', color='red', linewidth=2, marker='|')
ax3.set_xlim(first, last)
ax3.set_ylabel(ylabel='freq [Hz]', color='green')
ax3.set_ylim([np.nanmin(X[16, :])-5, np.nanmax(X[16, :])+8])
ax4.set_ylabel(ylabel='power %', color='red')
ax4.set_ylim([0, 100])
ax3.set_xlabel('input rate')
ax3.legend(loc=2)
ax4.legend()

fig3.tight_layout()

fig3.savefig(os.path.join(SWBasePath, 'figures', 'ripple.png'))

fig4 = plt.figure(figsize=(10, 8))

ax = fig4.add_subplot(2, 1, 1)
ax.plot(rates_mf, X[14, :], label='Gamma freq (exc.)', color='blue', linewidth=2, marker='o')
ax2 = ax.twinx()
ax2.plot(rates_mf, X[15, :], label='Gamma power (exc.)', color='red', linewidth=2, marker='|')
ax.set_xlim(first, last)
ax.set_ylabel(ylabel='freq [Hz]', color='blue')
ax.set_ylim([np.nanmin(X[14, :])-5, np.nanmax(X[14, :])+8])
ax2.set_ylabel(ylabel='power %', color='red')
ax2.set_ylim([0, 100])
ax.set_title('Gamma oscillation')
ax.legend(loc=2)
ax2.legend()

ax3 = fig4.add_subplot(2, 1, 2)
ax3.plot(rates_mf, X[18, :], label='Gamma freq (inh.)', color='green', linewidth=2, marker='o')
ax4 = ax3.twinx()
ax4.plot(rates_mf, X[19, :], label='Gamma power (inh.)', color='red', linewidth=2, marker='|')
ax3.set_xlim(first, last)
ax3.set_ylabel(ylabel='freq [Hz]', color='green')
ax3.set_ylim([np.nanmin(X[18, :])-5, np.nanmax(X[18, :])+8])
ax4.set_ylabel(ylabel='power %', color='red')
ax4.set_ylim([0, 100])
ax3.set_xlabel('input rate')
ax3.legend(loc=2)
ax4.legend()

fig4.tight_layout()

fig4.savefig(os.path.join(SWBasePath, 'figures', 'gamma.png'))

plt.close('all')

# Save result array (X)
fName = os.path.join(SWBasePath, 'files', fOut)
header = 'Input_rate, Mean_exc.rate, Max.exc.AC., at[ms], Max.exc.AC.in_ripple_range, at[ms],' \
         'Mean_inh.rate, Max.inh.AC., at[ms], Max.inh.AC.in_ripple_range, at[ms],' \
         'avg. replay interval,' \
         'avgRippleFE, ripplePE, avgGammaFE, ripplePE,' \
         'avgRippleFI, ripplePI, avgGammaFI, ripplePI'
np.savetxt(fName, X, fmt='%.6f', delimiter='\t', header=header)