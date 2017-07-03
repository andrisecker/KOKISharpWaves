#!/usr/bin/python
# -*- coding: utf8 -*-
'''
analyse EPSC (with the given cell model and synapse parameters) based on the learned weight matrix
author: András Ecker last update: 06.2017
'''

import os
from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
SWBasePath = os.path.sep.join(os.path.abspath('__file__').split(os.path.sep)[:-2])
# add the 'scripts' directory to the path (import the modules)
sys.path.insert(0, os.path.sep.join([SWBasePath, 'scripts']))
from detect_oscillations import load_Wee


fIn = "wmxR_asym.txt"
Wee_mult = 2.9

np.random.seed(12345)
SWBasePath = os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-2])
figFolder = os.path.join(SWBasePath, "figures")

wmx = load_Wee(os.path.join(SWBasePath, "files", fIn)) * Wee_mult
wmx_nz = wmx[np.nonzero(wmx)]
print "mean(nonzero weights): %s (nS)"%np.mean(wmx_nz)


# synaptic parameters:
PyrExc_rise = 1.3 * ms  # Gupta 2016 (only from Fig.1 H - 20-80%)
PyrExc_decay = 9.5 * ms  # Gupta 2016 ("needed for temporal summation of EPSPs") 
invpeak_PyrExc = (PyrExc_decay / PyrExc_rise) ** (PyrExc_rise / (PyrExc_decay - PyrExc_rise))
delay_PyrExc = 2.2 * ms  # Gupta 2016
E_Exc = 0.0 * mV

# parameters for pyr cells (optimized by Bence)
z = 1 * nS
gL_Pyr = 4.49581428461e-3 * uS
tauMem_Pyr = 37.97630516 * ms
Cm_Pyr = tauMem_Pyr * gL_Pyr
Vrest_Pyr = -59.710040237 * mV
reset_Pyr = -24.8988661181 * mV
theta_Pyr = -13.3139788756 * mV
tref_Pyr = 3.79313737057 * ms
a_Pyr = -0.255945300382 * nS
b_Pyr = 0.22030375858 * nA
delta_T_Pyr = 3.31719795927 * mV
tau_w_Pyr = 80.1747780694 * ms
v_spike_Pyr = theta_Pyr + 10 * delta_T_Pyr

eqs_Pyr = '''
dvm/dt = (-gL_Pyr*(vm-Vrest_Pyr) + gL_Pyr*delta_T_Pyr*exp((vm- theta_Pyr)/delta_T_Pyr)- w - EPSC)/Cm_Pyr : volt (unless refractory)
dw/dt = (a_Pyr*(vm- Vrest_Pyr )-w)/tau_w_Pyr : amp
dg_ampa/dt = (invpeak_PyrExc * x_ampa - g_ampa) / PyrExc_rise : 1
dx_ampa/dt = -x_ampa / PyrExc_decay : 1
EPSC = g_ampa*z*(vm-E_Exc): amp
'''

n = 100
weights = np.random.choice(wmx_nz, n, replace=False)
EPSPs = np.zeros((n, 2000))
EPSCs = np.zeros((n, 2000))
peakEPSPs = np.zeros(n)
peakEPSCs = np.zeros(n)

for i, weight in enumerate(weights):

    PE = NeuronGroup(1, model=eqs_Pyr, threshold="vm>v_spike_Pyr",
                     reset="vm=reset_Pyr; w+=b_Pyr", refractory=tref_Pyr, method="exponential_euler")
    PE.vm = Vrest_Pyr
    PE.g_ampa = 0

    inp = SpikeGeneratorGroup(1, np.array([0]), np.array([50])*ms)

    Cee = Synapses(inp, PE, 'w_exc:1', on_pre='x_ampa+=w_exc')
    Cee.connect(i=0, j=0)
    Cee.w_exc = weight  # nS
    Cee.delay = delay_PyrExc

    vSM = StateMonitor(PE, 'vm', record=True)
    iSM = StateMonitor(PE, 'EPSC', record=True)

    run(200*ms)
    
    EPSPs[i,:] = vSM[0].vm/mV
    EPSCs[i,:] = iSM[0].EPSC/pA
    peakEPSPs[i] = np.max(vSM[0].vm/mV) - Vrest_Pyr/mV
    peakEPSCs[i] = np.min(iSM[0].EPSC/pA)


# finall run with the actual average   
PE = NeuronGroup(1, model=eqs_Pyr, threshold="vm>v_spike_Pyr",
                     reset="vm=reset_Pyr; w+=b_Pyr", refractory=tref_Pyr, method="exponential_euler")
PE.vm = Vrest_Pyr
PE.g_ampa = 0

inp = SpikeGeneratorGroup(1, np.array([0]), np.array([50])*ms)

Cee = Synapses(inp, PE, 'w_exc:1', on_pre='x_ampa+=w_exc')
Cee.connect(i=0, j=0)
Cee.w_exc = np.mean(wmx_nz) # nS
Cee.delay = delay_PyrExc

vSM = StateMonitor(PE, 'vm', record=True)
iSM = StateMonitor(PE, 'EPSC', record=True)

run(200*ms)


# Plots
t = np.linspace(0, 200, 2000)

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(2, 1, 1)
ax.plot(t, np.mean(EPSPs, axis=0), "b-", label="mean of %i random weights"%n)
ax.plot(t, vSM[0].vm/mV, "g-", label="mean of all weights (%f nS)"%np.mean(wmx_nz))
ax.set_title("average EPSP")
ax.set_xlim([0,200])
ax.set_ylabel("EPSP (mV)")
ax.legend()
ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(t, np.mean(EPSCs, axis=0), "b-", label="mean of %i random weights"%n)
ax2.plot(t, iSM[0].EPSC/pA, "g-", label="mean of all weights (%f nS)"%np.mean(wmx_nz))
ax2.set_title("average EPSC")
ax2.set_xlabel("time (ms)")
ax2.set_xlim([0,200])
ax2.set_ylabel("EPSC (pA)")
ax2.legend()
fig.tight_layout()
figName = os.path.join(figFolder, "EPS*_sym.png")
fig.savefig(figName)

fig2 = plt.figure(figsize=(10,8))
ax = fig2.add_subplot(2, 1, 1)
ax.violinplot(peakEPSPs, vert=False, showmeans=True, showextrema=False, showmedians=False, points=n, bw_method='silverman')
ax.set_title("%i random EPSPs (mean: %f mV)"%(n, np.mean(peakEPSPs)))
ax.set_xlabel("EPSP (mV)")
ax.set_yticks([])
ax2 = fig2.add_subplot(2, 1, 2)
ax2.violinplot(peakEPSCs, vert=False, showmeans=True, showextrema=False, showmedians=False, points=n, bw_method='silverman')
ax2.set_title("%i random EPSCs (mean: %f pA)"%(n, np.mean(peakEPSCs)))
ax2.set_xlabel("EPSC (pA)")
ax2.set_yticks([])
fig2.tight_layout()
figName = os.path.join(figFolder, "distEPS*_sym.png")
fig2.savefig(figName)

plt.show()
    
    
