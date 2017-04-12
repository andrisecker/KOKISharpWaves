#!/usr/bin/python
# -*- coding: utf8 -*-
'''
crates PC (adExp IF) and BC (IF) population in Brian2, loads in recurrent connection matrix for PC population 
runs simulation and checks the dynamics
see more: https://drive.google.com/file/d/0B089tpx89mdXZk55dm0xZm5adUE/view
author: András Ecker last update: 04.2017
'''

import os
from brian2 import *
set_device('cpp_standalone')  # speed up the simulation with generated C++ code
import numpy as np
import matplotlib.pyplot as plt
from detect_oscillations import *
from plots import *

fIn = 'wmxR_asym.txt'

SWBasePath = '/'.join(os.path.abspath(__file__).split('/')[:-2])

np.random.seed(12345)

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
J_PyrInh = 0.15  # 0.125
J_BasExc = 4.5  # 5.2083
J_BasInh = 0.25  # 0.15

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
dvm/dt = (-gL_Pyr*(vm-Vrest_Pyr) + gL_Pyr*delta_T_Pyr*exp((vm- theta_Pyr)/delta_T_Pyr) - w - (g_ampa*z*(vm-E_Exc) + g_gaba*z*(vm-E_Inh)))/Cm_Pyr : volt (unless refractory)
dw/dt = (a_Pyr*(vm - Vrest_Pyr )-w)/tau_w_Pyr : amp
dg_ampa/dt = -g_ampa/tauSyn_PyrExc : 1
dg_gaba/dt = -g_gaba/tauSyn_PyrInh : 1
'''

eqs_bas = '''
dvm/dt = (-gL_Bas*(vm-Vrest_Bas) - (g_ampa*z*(vm-E_Exc) + g_gaba*z*(vm-E_Inh)))/Cm_Bas :volt (unless refractory)
dg_ampa/dt = -g_ampa/tauSyn_BasExc : 1
dg_gaba/dt = -g_gaba/tauSyn_BasInh : 1
'''


# ====================================== end of parameters ======================================


# load in Wee only once ...
def load_Wee(fName):  # this way the file will closed and memory will cleaned
    """dummy function, just to make python clear the memory"""
    Wee = np.genfromtxt(fName) * 1e9
    np.fill_diagonal(Wee, 0)  # just to make sure

    print "weight matrix loded"
    return Wee

fName = os.path.join(SWBasePath, 'files', fIn)
Wee = load_Wee(fName)

PE = NeuronGroup(NE, model=eqs_adexp, threshold="vm>v_spike_Pyr",
                 reset="vm=reset_Pyr; w+=b_Pyr", refractory=tref_Pyr, method="exponential_euler")
PI = NeuronGroup(NI, model=eqs_bas, threshold="vm>theta_Bas",
                 reset="vm=reset_Bas", refractory=tref_Bas, method="exponential_euler")

PE.vm = Vrest_Pyr
PE.g_ampa = 0
PE.g_gaba = 0

PI.vm  = Vrest_Bas
PI.g_ampa = 0
PI.g_gaba = 0

MF = PoissonGroup(NE, p_rate_mf)

print 'Connecting the network'

Cext = Synapses(MF, PE, on_pre="g_ampa+=J_PyrMF")
Cext.connect(j='i')

# weight matrix used here:
Cee = Synapses(PE, PE, 'w_exc:1', on_pre='g_ampa+=w_exc')
Cee.connect()
Cee.w_exc = Wee.flatten()
Cee.delay = delay_PyrExc

Cei = Synapses(PE, PI, on_pre='g_ampa+=J_BasExc')
Cei.connect(p=eps_pyr)
Cei.delay = delay_BasExc

Cie = Synapses(PI, PE, on_pre='g_gaba+=J_PyrInh')
Cie.connect(p=eps_bas)
Cie.delay = delay_PyrInh

Cii = Synapses(PI, PI, on_pre='g_gaba+=J_BasInh')
Cii.connect(p=eps_bas)
Cii.delay = delay_BasInh

print 'Connections done'


# Monitors
sme = SpikeMonitor(PE)
smi = SpikeMonitor(PI)
#selection = np.arange(0, 4000, 100) # subset of neurons for recoring variables
del Wee  # cleary memory


run(10000*ms, report='text')


# Raster + ISI plot
spikeTimesE, spikingNeuronsE, poprE, ISI = preprocess_spikes(sme.spike_trains(), NE)
ISI = plot_raster_ISI(spikeTimesE, spikingNeuronsE, ISI, "blue", multiplier_=1)

if np.max(poprE > 0):  # check if there is any activity
   
    spikeTimesI, spikingNeuronsI, poprI = preprocess_spikes(smi.spike_trains(), NI, calc_ISI=False)

    # calling detect_oscillation functions:
    avgReplayInterval = replay(ISI[3:16])  # bins from 150 to 850 (range of interest)

    meanEr, rEAC, maxEAC, tMaxEAC, maxEACR, tMaxEACR, fE, PxxE, avgRippleFE, ripplePE = ripple(poprE, 1000)
    avgGammaFE, gammaPE = gamma(fE, PxxE)
    meanIr, rIAC, maxIAC, tMaxIAC, maxIACR, tMaxIACR, fI, PxxI, avgRippleFI, ripplePI = ripple(poprI, 1000)
    avgGammaFI, gammaPI = gamma(fI, PxxI)

    # Print out some info
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
    print "--------------------------------------------------"


    # Plots
    plot_PSD(poprE, rEAC, fE, PxxE, "Pyr_population", 'b-', multiplier_=1)
    plot_PSD(poprI, rIAC, fI, PxxI, "Bas_population", 'g-', multiplier_=1)

    ymin, ymax = plot_zoomed(spikeTimesE, spikingNeuronsE, poprE, "Pyr_population", "blue", multiplier_=1)
    plot_zoomed(spikeTimesI, spikingNeuronsI, poprI, "Bas_population", "green", multiplier_=1, Pyr_pop=False)
    #subset = select_subset(selection, ymin, ymax)
    #plot_detailed(msMe, subset, dWee, multiplier_=1)
    #plot_adaptation(msMe, selection, multiplier_=1)

else:  # if there is no activity the auto-correlation function will throw an error!
    
    print "No activity !"
    print "--------------------------------------------------"

plt.show()


