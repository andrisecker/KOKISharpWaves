#!/usr/bin/python
# -*- coding: utf8 -*-
'''
updated network, parameters are (should be) closer to the experimental data!
still in development - search for #TODO: !!!
creates PC (adExp IF) and BC (adExp IF) population in Brian2, loads in recurrent connection matrix for PC population
runs simulation and checks the dynamics
authors: Bagi Bence, András Ecker last update: 05.2017
'''

import os
import sys
from brian2 import *
set_device('cpp_standalone')  # speed up the simulation with generated C++ code
import numpy as np
import matplotlib.pyplot as plt

SWBasePath = os.path.sep.join(os.path.abspath('__file__').split(os.path.sep)[:-2])
# add the 'scripts' directory to the path (import the modules)
sys.path.insert(0, os.path.sep.join([SWBasePath, 'scripts']))
from detect_oscillations import *
from plots import *

fIn = 'wmxR_asym.txt'

np.random.seed(12345)

NE = 4000
NI = 1000

# sparseness
eps_pyr = 0.16
eps_bas = 0.4

z = 1*nS
# parameters for pyr cells
gL_Pyr = 4.49581428461e-3 * uS
tauMem_Pyr = 37.97630516 * ms
Cm_Pyr = tauMem_Pyr * gL_Pyr
Vrest_Pyr = -59.710040237 * mV
reset_Pyr = -24.8988661181 * mV
theta_Pyr = -13.3139788756 * mV
tref_Pyr = 3.79313737057 * ms
# adaptation parameters for pyr cells
a_Pyr = -0.255945300382 * nS  # subthreshold adaptation conductance
b_Pyr = 0.22030375858 * nA   # spike-triggered adaptation
delta_T_Pyr = 3.31719795927 * mV   # slope factor
tau_w_Pyr = 80.1747780694 * ms  # adaptation time constant
v_spike_Pyr = theta_Pyr + 10 * delta_T_Pyr

# parameters for bas cells
gL_Bas = 7.0102757369e-3 * uS
tauMem_Bas = 37.7598232668 * ms
Cm_Bas = tauMem_Bas * gL_Bas
Vrest_Bas = -58.9682231705 * mV
reset_Bas = -39.1229822301 * mV
theta_Bas = -39.5972788689 * mV
tref_Bas = 1.06976577195 * ms
# adaptation parameters for bas cells
a_Bas = 0.821975246336 * nS  # subthreshold adaptation conductance
b_Bas = 0.00398843790629 * nA   # spike-triggered adaptation
delta_T_Bas = 2.21103724225 * mV   # slope factor
tau_w_Bas = 415.241939453 * ms  # adaptation time constant
v_spike_Bas = theta_Bas + 10 * delta_T_Bas

# synaptic reversal potentials
E_Exc = 0.0 * mV
E_Inh = -70.0 * mV

# synaptic weights: hand tuned by Bence #TODO: update these!
J_PyrInh = 0.04 
J_BasExc = 32. 
J_BasInh = 10. 

# synaptic time constants: same as in Janke 2015 #TODO: update these!
# rise time constants
PyrExc_rise = 0.6  * ms 
PyrInh_rise = 0.35 * ms
BasExc_rise = 0.35 * ms 
BasInh_rise = 0.3  * ms
# decay time constants
PyrExc_decay = 4.8  * ms 
PyrInh_decay = 3.3  * ms
BasExc_decay = 2.0  * ms
BasInh_decay = 1.2  * ms
# Normalization factors (normalize the peak of the PSP curve to 1)
invpeak_PyrExc = (PyrExc_decay / PyrExc_rise) ** (PyrExc_rise / (PyrExc_decay - PyrExc_rise))
invpeak_PyrInh = (PyrInh_decay / PyrInh_rise) ** (PyrInh_rise / (PyrInh_decay - PyrInh_rise))
invpeak_BasExc = (BasExc_decay / BasExc_rise) ** (BasExc_rise / (BasExc_decay - BasExc_rise))
invpeak_BasInh = (BasInh_decay / BasInh_rise) ** (BasInh_rise / (BasInh_decay - BasInh_rise))

# synaptic delays: hand tuned by Bence #TODO: update these!
delay_PyrExc = 4.2 * ms
delay_PyrInh = 1.1 * ms
delay_BasExc = 0.5 * ms
delay_BasInh = 0.6 * ms

# input parameters: hand tuned by Bence #TODO: update these!
p_rate_mf = 25.0 * Hz
J_PyrMF = 30.0


eqs_Pyr = '''
dvm/dt = (-gL_Pyr*(vm-Vrest_Pyr) + gL_Pyr*delta_T_Pyr*exp((vm- theta_Pyr)/delta_T_Pyr)-w - (g_ampa*z*(vm-E_Exc) + g_gaba*z*(vm-E_Inh)))/Cm_Pyr : volt (unless refractory)
dw/dt = (a_Pyr*(vm- Vrest_Pyr )-w)/tau_w_Pyr : amp
dg_ampa/dt = (invpeak_PyrExc * x_ampa - g_ampa) / PyrExc_rise : 1
dx_ampa/dt = -x_ampa / PyrExc_decay : 1
dg_gaba/dt = (invpeak_PyrInh * x_gaba - g_gaba) / PyrInh_rise : 1
dx_gaba/dt = -x_gaba/PyrInh_decay : 1
'''

eqs_Bas = '''
dvm/dt = (-gL_Bas*(vm-Vrest_Bas) + gL_Bas*delta_T_Bas*exp((vm- theta_Bas)/delta_T_Bas)-w - (g_ampa*z*(vm-E_Exc) + g_gaba*z*(vm-E_Inh)))/Cm_Bas : volt (unless refractory)
dw/dt = (a_Bas*(vm- Vrest_Bas )-w)/tau_w_Bas : amp
dg_ampa/dt = (invpeak_BasExc * x_ampa - g_ampa) / BasExc_rise : 1
dx_ampa/dt = -x_ampa/BasExc_decay : 1
dg_gaba/dt = (invpeak_BasInh * x_gaba - g_gaba) / BasInh_rise : 1
dx_gaba/dt = -x_gaba/BasInh_decay : 1
'''

# ====================================== end of parameters ======================================


# load in Wee only once ...
def load_Wee(fName):  # this way the file will closed and memory will cleaned
    """dummy function, just to make python clear the memory"""
    Wee = np.genfromtxt(fName) * 1e9 * 5 # *5 hand tuned by Bence #TODO: update these!
    np.fill_diagonal(Wee, 0)  # just to make sure

    print "weight matrix loaded"
    return Wee

fName = os.path.join(SWBasePath, 'files', fIn)
Wee = load_Wee(fName)


PE = NeuronGroup(NE, model=eqs_Pyr, threshold="vm>v_spike_Pyr",
                 reset="vm=reset_Pyr; w+=b_Pyr", refractory=tref_Pyr, method="exponential_euler")
PI = NeuronGroup(NI, model=eqs_Bas, threshold="vm>v_spike_Bas",
                 reset="vm=reset_Bas; w+=b_Bas", refractory=tref_Bas, method="exponential_euler")

PE.vm = Vrest_Pyr
PE.g_ampa = 0
PE.g_gaba = 0

PI.vm  = Vrest_Bas
PI.g_ampa = 0
PI.g_gaba = 0

MF = PoissonGroup(NE, p_rate_mf)

print 'Connecting the network'

Cext = Synapses(MF, PE, on_pre="x_ampa+=J_PyrMF")
Cext.connect(j='i')

# weight matrix used here:
Cee = Synapses(PE, PE, 'w_exc:1', on_pre='x_ampa+=w_exc')
Cee.connect()
Cee.w_exc = Wee.flatten()
Cee.delay = delay_PyrExc

Cei = Synapses(PE, PI, on_pre='x_ampa+=J_BasExc')
Cei.connect(p=eps_pyr)
Cei.delay = delay_BasExc

Cie = Synapses(PI, PE, on_pre='x_gaba+=J_PyrInh')
Cie.connect(p=eps_bas)
Cie.delay = delay_PyrInh

Cii = Synapses(PI, PI, on_pre='x_gaba+=J_BasInh')
Cii.connect(p=eps_bas)
Cii.delay = delay_BasInh

print 'Connections done'
del Wee  # cleary memory

# Monitors
sme = SpikeMonitor(PE)
smi = SpikeMonitor(PI)
popre = PopulationRateMonitor(PE)
popri = PopulationRateMonitor(PI)             


run(10000*ms, report='text')


if sme.num_spikes > 0 and smi.num_spikes > 0:  # check if there is any activity

    spikeTimesE, spikingNeuronsE, poprE, ISIhist, bin_edges = preprocess_monitors(sme, popre)
    spikeTimesI, spikingNeuronsI, poprI = preprocess_monitors(smi, popri, calc_ISI=False)

    # calling detect_oscillation functions:
    avgReplayInterval = replay(ISIhist[3:16])  # bins from 150 to 850 (range of interest)

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
    plot_raster_ISI(spikeTimesE, spikingNeuronsE, [ISIhist, bin_edges], 'blue', multiplier_=1)
    plot_PSD(poprE, rEAC, fE, PxxE, "Pyr_population", 'b-', multiplier_=1)
    plot_PSD(poprI, rIAC, fI, PxxI, "Bas_population", 'g-', multiplier_=1)

    _, _ = plot_zoomed(spikeTimesE, spikingNeuronsE, poprE, "Pyr_population", "blue", multiplier_=1)
    plot_zoomed(spikeTimesI, spikingNeuronsI, poprI, "Bas_population", "green", multiplier_=1, Pyr_pop=False)

else:  # if there is no activity the auto-correlation function will throw an error!

    print "No activity !"
    print "--------------------------------------------------"

plt.show()            
                 
                 
