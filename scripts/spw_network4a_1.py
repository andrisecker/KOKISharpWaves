#!/usr/bin/python
# -*- coding: utf8 -*-

from brian import *
import numpy as np
import matplotlib.pyplot as plt
import os
from detect_oscillations import replay, ripple, gamma

fIn = 'wmxR.txt'

SWBasePath = os.path.split(os.path.split(__file__)[0])[0]

NE = 4000
NI = 1000

# sparseness
eps_pyr = 0.16
eps_bas = 0.4

# parameters for pyr cells
z = 1*nS
gL_Pyr = 4.333e-3 * uS
tauMem_Pyr = 60.0 * ms
Cm_Pyr = tauMem_Pyr * gL_Pyr
Vrest_Pyr = -70.0 * mV
reset_Pyr = -53.0*mV
theta_Pyr = -50.0*mV
tref_Pyr = 5*ms
# Adaptation parameters for pyr cells
a_Pyr = -0.8*nS  # nS    Subthreshold adaptation conductance
b_Pyr = 0.04*nA  # nA    Spike-triggered adaptation
delta_T_Pyr = 2.0*mV  # Slope factor
tau_w_Pyr = 300*ms  # Adaptation time constant
v_spike_Pyr = theta_Pyr + 10 * delta_T_Pyr

# parameters for bas cells
gL_Bas = 5.0e-3*uS
tauMem_Bas = 14.0*ms
Cm_Bas = tauMem_Bas * gL_Bas
Vrest_Bas = -70.0*mV
reset_Bas = -64.0*mV
theta_Bas = -50.0*mV
tref_Bas = 0.1*ms

# synaptic weights
J_PyrInh = 0.125  # 0.15  # 0.125
J_BasExc = 5.2083  # 4.5  # 5.2083
J_BasInh = 0.15  # 0.25  # 0.15

print 'J_PyrInh:', J_PyrInh
print 'J_BasExc:', J_BasExc
print 'J_BasInh:', J_BasInh

# Synaptic reversal potentials
E_Exc = 0.0*mV
E_Inh = -70.0*mV

# Synaptic time constants
tauSyn_PyrExc = 10.0*ms
tauSyn_PyrInh = 3.0*ms
tauSyn_BasExc = 3.0*ms
tauSyn_BasInh = 1.5*ms

# Synaptic delays
delay_PyrExc = 3.0*ms
delay_PyrInh = 1.5*ms
delay_BasExc = 3.0*ms
delay_BasInh = 1.5*ms

# input parameters
p_rate_mf = 5.0*Hz
J_PyrMF = 5.0

# Creating populations
eqs_adexp = '''
dvm/dt = (-gL_Pyr*(vm-Vrest_Pyr) + gL_Pyr*delta_T_Pyr*exp((vm- theta_Pyr)/delta_T_Pyr)-w - (g_ampa*z*(vm-E_Exc) + g_gaba*z*(vm-E_Inh)))/Cm_Pyr : volt
dw/dt = (a_Pyr*(vm- Vrest_Pyr )-w)/tau_w_Pyr : amp
dg_ampa/dt = -g_ampa/tauSyn_PyrExc : 1
dg_gaba/dt = -g_gaba/tauSyn_PyrInh : 1
'''

reset_adexp = '''
vm = reset_Pyr
w += b_Pyr
'''

eqs_bas = '''
dvm/dt = (-gL_Bas*(vm-Vrest_Bas) - (g_ampa*z*(vm-E_Exc) + g_gaba*z*(vm-E_Inh)))/Cm_Bas :volt
dg_ampa/dt = -g_ampa/tauSyn_BasExc : 1
dg_gaba/dt = -g_gaba/tauSyn_BasInh : 1
'''

def myresetfunc(P, spikes):
    P.vm[spikes] = reset_Pyr   # reset voltage
    P.w[spikes] += b_Pyr  # low pass filter of spikes (adaptation mechanism)


SCR = SimpleCustomRefractoriness(myresetfunc, tref_Pyr, state='vm')

PE = NeuronGroup(NE, model=eqs_adexp, threshold=v_spike_Pyr, reset=SCR)

PI = NeuronGroup(NI, model=eqs_bas, threshold=theta_Bas, reset=reset_Bas, refractory=tref_Bas)

PE.vm = Vrest_Pyr
PE.g_ampa = 0
PE.g_gaba = 0

PI.vm  = Vrest_Bas
PI.g_ampa = 0
PI.g_gaba = 0

MF = PoissonGroup(NE, p_rate_mf)

print 'Connecting the network'

Cext = IdentityConnection(MF, PE, 'g_ampa', weight=J_PyrMF)

Cee = Connection(PE, PE, 'g_ampa', delay=delay_PyrExc)

fName = os.path.join(SWBasePath, 'files', fIn)
f = file(fName, 'r')

Wee = [line.split() for line in f]

f.close()

for i in range(NE):
    Wee[i][:] = [float(x) * 1.e9 for x in Wee[i]]

Cee.connect(PE, PE, Wee)

Cei = Connection(PE, PI, 'g_ampa', weight=J_BasExc, sparseness=eps_pyr, delay=delay_BasExc)
Cie = Connection(PI, PE, 'g_gaba', weight=J_PyrInh, sparseness=eps_bas, delay=delay_PyrInh)
Cii = Connection(PI, PI, 'g_gaba', weight=J_BasInh, sparseness=eps_bas, delay=delay_BasInh)

print 'Connections done'

# Monitors
sme = SpikeMonitor(PE)
smi = SpikeMonitor(PI)
popre = PopulationRateMonitor(PE, bin=1*ms)
popri = PopulationRateMonitor(PI, bin=1*ms)
poprext = PopulationRateMonitor(MF, bin=1*ms)
bins = [0*ms, 50*ms, 100*ms, 150*ms, 200*ms, 250*ms, 300*ms, 350*ms, 400*ms, 450*ms, 500*ms,
        550*ms, 600*ms, 650*ms, 700*ms, 750*ms, 800*ms, 850*ms, 900*ms, 950*ms, 1000*ms]
isi = ISIHistogramMonitor(PE, bins)


run(10000*ms, report='text')

avgReplayInterval = replay(isi.count[3:17])  # bins from 150 to 850 (range of interest)

meanEr, rEAC, maxEAC, tMaxEAC, maxEACR, tMaxEACR, fE, PxxE, avgRippleFE, ripplePE = ripple(popre.rate, 1000)
avgGammaFE, gammaPE = gamma(fE, PxxE)
meanIr, rIAC, maxIAC, tMaxIAC, maxIACR, tMaxIACR, fI, PxxI, avgRippleFI, ripplePI = ripple(popri.rate, 1000)
avgGammaFI, gammaPI = gamma(fI, PxxI)

print 'Mean excitatory rate: ', meanEr
print 'Maximum exc. autocorrelation:', maxEAC, 'at', tMaxEAC, '[ms]'
print 'Maximum exc. AC in ripple range:', maxEACR, 'at', tMaxEACR, '[ms]'
print 'Mean inhibitory rate: ', meanIr
print 'Maximum inh. autocorrelation:', maxIAC, 'at', tMaxIAC, '[ms]'
print 'Maximum inh. AC in ripple range:', maxIACR, 'at', tMaxIACR, '[ms]'
print ''
print 'Average exc. ripple freq:', avgRippleFE
print 'Exc. ripple power:', ripplePE
print 'Average exc. gamma freq:', avgGammaFE
print 'Exc. gamma power:', gammaPE
print 'Average inh. ripple freq:', avgRippleFI
print 'Inh. ripple power:', ripplePI
print 'Average inh. gamma freq:', avgGammaFI
print 'Inh. gamma power:', gammaPI


fName = os.path.join(SWBasePath, 'files', 'spikes_0.8.npz')
# np.savez(fName, spikes=sme.spikes, spiketimes=sme.spiketimes.values())


# Plots
fig = plt.figure(figsize=(10, 8))

subplot(2, 1, 1)
raster_plot(sme, spacebetweengroups=1, title='Raster plot', newfigure=False)

subplot(2, 1, 2)
hist_plot(isi, title='ISI histogram', newfigure=False)
xlim([0, 1000])

fig.tight_layout()

# Pyr population
fig2 = plt.figure(figsize=(10, 8))

ax = fig2.add_subplot(3, 1, 1)
ax.plot(np.linspace(0, 10000, len(popre.rate)), popre.rate, 'b-')
ax.set_title('Pyr. population rate')
ax.set_xlabel('Time [ms]')

rEACPlot = rEAC[2:201]  # 500 - 5 Hz interval

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
ax.set_title('Pyr. raster plot (last 100 ms)')
ax.set_xlim([9900, 10000])
ax.set_xlabel('Time [ms]')
ax.set_ylim([ymin, ymax])
ax.set_ylabel('Neuron number')

ax2 = fig4.add_subplot(2, 1, 2)
ax2.plot(np.linspace(9900, 10000, len(popre.rate[9900:10000])), popre.rate[9900:10000], 'b-', linewidth=1.5)
ax2.set_title('Pyr. population rate (last 100 ms)')
ax2.set_xlabel('Time [ms]')

fig4.tight_layout()

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

plt.show()