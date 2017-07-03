#!/usr/bin/python
# -*- coding: utf8 -*-
'''
analyse pure BC network with Poisson input
author: András Ecker last update: 06.2017
'''

import os
from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
SWBasePath = os.path.sep.join(os.path.abspath('__file__').split(os.path.sep)[:-2])
# add the 'scripts' directory to the path (import the modules)
sys.path.insert(0, os.path.sep.join([SWBasePath, 'scripts']))
from detect_oscillations import *
from plots import *
from detect_oscillations import *
from plots import *

np.random.seed(12345)

NE = 4000
NI = 1000
eps_pyr = 0.16
eps_bas = 0.4

# synaptic parameters
BasExc_rise = 1. * ms  # Lee 2014 (data from CA1) 
BasInh_rise = 0.25 * ms  # Bartos 2002 (20-80%)
BasExc_decay = 4.1 * ms  # Lee 2014 (data from CA1)
BasInh_decay = 1.2 * ms  # Bartos 2002
invpeak_BasExc = (BasExc_decay / BasExc_rise) ** (BasExc_rise / (BasExc_decay - BasExc_rise))
invpeak_BasInh = (BasInh_decay / BasInh_rise) ** (BasInh_rise / (BasInh_decay - BasInh_rise))
delay_BasExc = 0.9 * ms  # Geiger 1997 (data from DG)
delay_BasInh = 0.6 * ms  # Bartos 2002
E_Exc = 0.0 * mV
E_Inh = -70.0 * mV
z = 1 * nS

# parameters for bas cells (optimized by Bence)
gL_Bas = 7.0102757369e-3 * uS
tauMem_Bas = 37.7598232668 * ms
Cm_Bas = tauMem_Bas * gL_Bas
Vrest_Bas = -58.9682231705 * mV
reset_Bas = -39.1229822301 * mV
theta_Bas = -39.5972788689 * mV
tref_Bas = 1.06976577195 * ms
delta_T_Bas = 2.21103724225 * mV
v_spike_Bas = theta_Bas + 10 * delta_T_Bas

eqs_Bas = '''
dvm/dt = (-gL_Bas*(vm-Vrest_Bas) + gL_Bas*delta_T_Bas*exp((vm- theta_Bas)/delta_T_Bas) - (g_ampa*z*(vm-E_Exc) + g_gaba*z*(vm-E_Inh)))/Cm_Bas : volt (unless refractory)
dg_ampa/dt = (invpeak_BasExc * x_ampa - g_ampa) / BasExc_rise : 1
dx_ampa/dt = -x_ampa/BasExc_decay : 1
dg_gaba/dt = (invpeak_BasInh * x_gaba - g_gaba) / BasInh_rise : 1
dx_gaba/dt = -x_gaba/BasInh_decay : 1
'''

J_BasExc = 5.
J_BasInh = 0.4
exc_rates = np.linspace(2, 4, 5)


for exc_rate in exc_rates:

    rate_ = NE * eps_pyr * exc_rate * Hz  # calc incoming rate

    PI = NeuronGroup(NI, model=eqs_Bas, threshold="vm>v_spike_Bas",
                     reset="vm=reset_Bas", refractory=tref_Bas, method="exponential_euler")
    PI.vm  = Vrest_Bas
    PI.g_ampa = 0
    PI.g_gaba = 0

    outer_input = PoissonGroup(NI, rate_)

    Cext = Synapses(outer_input, PI, on_pre="x_ampa+=J_BasExc")
    Cext.connect(j='i')

    Cii = Synapses(PI, PI, on_pre='x_gaba+=J_BasInh')
    Cii.connect(p=eps_bas)
    Cii.delay = delay_BasInh

    smi = SpikeMonitor(PI)
    popri = PopulationRateMonitor(PI)             

    run(10000*ms, report="text")


    if smi.num_spikes > 0:  # check if there is any activity

        spikeTimesI, spikingNeuronsI, poprI = preprocess_monitors(smi, popri, calc_ISI=False)
        meanIr, rIAC, maxIAC, tMaxIAC, maxIACR, tMaxIACR, fI, PxxI, avgRippleFI, ripplePI = ripple(poprI, 1000)
        avgGammaFI, gammaPI = gamma(fI, PxxI)
        
        # Print out some info
        print 'Mean inhibitory rate: ', meanIr
        print 'Average inh. ripple freq:', avgRippleFI
        print 'Inh. ripple power:', ripplePI
        print 'Average inh. gamma freq:', avgGammaFI
        print 'Inh. gamma power:', gammaPI
        print "--------------------------------------------------"
        
        # Plots
        plot_PSD(poprI, rIAC, fI, PxxI, "Bas_population", 'g-', multiplier_=exc_rate)
        plot_zoomed(spikeTimesI, spikingNeuronsI, poprI, "Bas_population", "green", multiplier_=exc_rate, Pyr_pop=False)

    else:  # if there is no activity the auto-correlation function will throw an error!

        print "No activity !"
        print "--------------------------------------------------"

plt.show()
