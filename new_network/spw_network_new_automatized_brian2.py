#!/usr/bin/python
# -*- coding: utf8 -*-
'''
looped version of spw_network_new_brian2.py -> checks the dynamics for different multipliers of the learned weight matrix
authors: Bence Bagi, András Ecker last update: 06.2017
'''

import os
import gc
import sys
from brian2 import *
import numpy as np
import random as pyrandom
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore") # ignore scipy 0.18 sparse matrix warning...
SWBasePath = os.path.sep.join(os.path.abspath('__file__').split(os.path.sep)[:-2])
# add the 'scripts' directory to the path (import the modules)
sys.path.insert(0, os.path.sep.join([SWBasePath, 'scripts']))
from detect_oscillations import *
from plots import *


STDP_mode = "sym"
fIn = "wmxR_%s.txt"%STDP_mode
fOut = "%s_baseline_v1.txt"%STDP_mode

first = 0.7
last = 1.5
dataPoints = 9
multipliers = np.linspace(first, last, dataPoints)

# synaptic weights

if STDP_mode == "asym":
    J_PyrInh = 0.02
    J_BasExc = 5
    J_BasInh = 0.4
elif STDP_mode == "sym":
    J_PyrInh = 0.016
    J_BasExc = 4.5
    J_BasInh = 0.75
# wmx scale factor already introduced in the stdp* script!
   
# mossy fiber input
J_PyrMF = 24.25
rate_MF = 20 * Hz

# size of populations
NE = 4000
NI = 1000
# sparseness
eps_pyr = 0.16
eps_bas = 0.4

# synaptic time constants:
# rise time constants
PyrExc_rise = 1.3 * ms  # Gupta 2016 (only from Fig.1 H - 20-80%)
PyrExcMF_rise = 0.65 * ms  # Vyleta ... Jonas 2016 (20-80%)
PyrInh_rise = 0.3 * ms  # Bartos 2002 (20-80%)
BasExc_rise = 1. * ms  # Lee 2014 (data from CA1) 
BasInh_rise = 0.25 * ms  # Bartos 2002 (20-80%)
# decay time constants
PyrExc_decay = 9.5 * ms  # Gupta 2016 ("needed for temporal summation of EPSPs") 
PyrExcMF_decay = 5.4 * ms  # Vyleta ... Jonas 2016
PyrInh_decay = 3.3 * ms  # Bartos 2002
BasExc_decay = 4.1 * ms  # Lee 2014 (data from CA1)
BasInh_decay = 1.2 * ms  # Bartos 2002
# Normalization factors (normalize the peak of the PSC curve to 1)
invpeak_PyrExc = (PyrExc_decay / PyrExc_rise) ** (PyrExc_rise / (PyrExc_decay - PyrExc_rise))
invpeak_PyrExcMF = (PyrExcMF_decay / PyrExcMF_rise) ** (PyrExcMF_rise / (PyrExcMF_decay - PyrExcMF_rise))
invpeak_PyrInh = (PyrInh_decay / PyrInh_rise) ** (PyrInh_rise / (PyrInh_decay - PyrInh_rise))
invpeak_BasExc = (BasExc_decay / BasExc_rise) ** (BasExc_rise / (BasExc_decay - BasExc_rise))
invpeak_BasInh = (BasInh_decay / BasInh_rise) ** (BasInh_rise / (BasInh_decay - BasInh_rise))

# synaptic delays:
delay_PyrExc = 2.2 * ms  # Gupta 2016
delay_PyrInh = 1.1 * ms  # Bartos 2002
delay_BasExc = 0.9 * ms  # Geiger 1997 (data from DG)
delay_BasInh = 0.6 * ms  # Bartos 2002
        
# synaptic reversal potentials
E_Exc = 0.0 * mV
E_Inh = -70.0 * mV

z = 1 * nS
# parameters for pyr cells (optimized by Bence)
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


eqs_Pyr = '''
dvm/dt = (-gL_Pyr*(vm-Vrest_Pyr) + gL_Pyr*delta_T_Pyr*exp((vm- theta_Pyr)/delta_T_Pyr) - w - ((g_ampa+g_ampaMF)*z*(vm-E_Exc) + g_gaba*z*(vm-E_Inh)))/Cm_Pyr : volt (unless refractory)
dw/dt = (a_Pyr*(vm- Vrest_Pyr )-w)/tau_w_Pyr : amp
dg_ampa/dt = (invpeak_PyrExc * x_ampa - g_ampa) / PyrExc_rise : 1
dx_ampa/dt = -x_ampa / PyrExc_decay : 1
dg_ampaMF/dt = (invpeak_PyrExcMF * x_ampaMF - g_ampaMF) / PyrExcMF_rise : 1
dx_ampaMF/dt = -x_ampaMF / PyrExcMF_decay : 1
dg_gaba/dt = (invpeak_PyrInh * x_gaba - g_gaba) / PyrInh_rise : 1
dx_gaba/dt = -x_gaba/PyrInh_decay : 1
'''

eqs_Bas = '''
dvm/dt = (-gL_Bas*(vm-Vrest_Bas) + gL_Bas*delta_T_Bas*exp((vm- theta_Bas)/delta_T_Bas) - (g_ampa*z*(vm-E_Exc) + g_gaba*z*(vm-E_Inh)))/Cm_Bas : volt (unless refractory)
dg_ampa/dt = (invpeak_BasExc * x_ampa - g_ampa) / BasExc_rise : 1
dx_ampa/dt = -x_ampa/BasExc_decay : 1
dg_gaba/dt = (invpeak_BasInh * x_gaba - g_gaba) / BasInh_rise : 1
dx_gaba/dt = -x_gaba/BasInh_decay : 1
'''


# ====================================== end of parameters ======================================

# load weight matrix
fName = os.path.join(SWBasePath, "files", fIn)
Wee = load_Wee(fName)

# plot matrix (and some selected weights)
plot_wmx_avg(Wee, 100, "wmx")
selection = np.array([500, 1500, 2500, 3500])  # some random neuron IDs to save weigths
dWee = save_selected_w(Wee, selection)
plot_weights(dWee, "sel_weights")
plt.close("all")


X = np.zeros((20, dataPoints))  # init. container to store results

# ====================================== iterates over diff. multipliers ======================================

for k, multiplier in enumerate(multipliers):

    print "multiplier: %.2f"%multiplier
    Wee_tmp = Wee * multiplier  # Wee matrix loaded before the for loop

    np.random.seed(12345)
    pyrandom.seed(12345)

    # recreate the neurons in every iteration (just to make sure!)
    PE = NeuronGroup(NE, model=eqs_Pyr, threshold="vm>v_spike_Pyr",
                     reset="vm=reset_Pyr; w+=b_Pyr", refractory=tref_Pyr, method="exponential_euler")
    PI = NeuronGroup(NI, model=eqs_Bas, threshold="vm>v_spike_Bas",
                     reset="vm=reset_Bas", refractory=tref_Bas, method="exponential_euler")

    PE.vm = Vrest_Pyr
    PE.g_ampa = 0
    PE.g_ampaMF = 0
    PE.g_gaba = 0
    PI.vm  = Vrest_Bas
    PI.g_ampa = 0
    PI.g_gaba = 0

    np.random.seed(1234)
    pyrandom.seed(1234)

    MF = PoissonGroup(NE, rate_MF)

    Cext = Synapses(MF, PE, on_pre="x_ampaMF+=J_PyrMF")
    Cext.connect(j='i')

    # weight matrix used here:
    Cee = Synapses(PE, PE, 'w_exc:1', on_pre='x_ampa+=w_exc')
    Cee.connect()
    Cee.w_exc = Wee_tmp.flatten()
    Cee.delay = delay_PyrExc
    del Wee_tmp  # clear memory

    Cei = Synapses(PE, PI, on_pre='x_ampa+=J_BasExc')
    Cei.connect(p=eps_pyr)
    Cei.delay = delay_BasExc

    Cie = Synapses(PI, PE, on_pre='x_gaba+=J_PyrInh')
    Cie.connect(p=eps_bas)
    Cie.delay = delay_PyrInh

    Cii = Synapses(PI, PI, on_pre='x_gaba+=J_BasInh')
    Cii.connect(p=eps_bas)
    Cii.delay = delay_BasInh

    # Monitors
    sme = SpikeMonitor(PE)
    smi = SpikeMonitor(PI)
    popre = PopulationRateMonitor(PE)
    popri = PopulationRateMonitor(PI)
    selection = np.arange(0, 4000, 50)  # subset of neurons for recoring variables
    msMe = StateMonitor(PE, ["vm", "w", "g_ampa", "g_ampaMF","g_gaba"], record=selection.tolist())  # comment this out later (takes memory!)        


    run(10000*ms, report="text")


    if sme.num_spikes > 0 and smi.num_spikes > 0:  # check if there is any activity

        spikeTimesE, spikingNeuronsE, poprE, ISIhist, bin_edges = preprocess_monitors(sme, popre)
        spikeTimesI, spikingNeuronsI, poprI = preprocess_monitors(smi, popri, calc_ISI=False)

        # calling detect_oscillation functions:
        avgReplayInterval = replay(ISIhist[3:16])  # bins from 150 to 850 (range of interest)

        meanEr, rEAC, maxEAC, tMaxEAC, maxEACR, tMaxEACR, fE, PxxE, avgRippleFE, ripplePE = ripple(poprE, 1000)
        avgGammaFE, gammaPE = gamma(fE, PxxE)
        meanIr, rIAC, maxIAC, tMaxIAC, maxIACR, tMaxIACR, fI, PxxI, avgRippleFI, ripplePI = ripple(poprI, 1000)
        avgGammaFI, gammaPI = gamma(fI, PxxI)

        print "Avg. exc. ripple freq:%s, Avg. inh. ripple freq:%s"%(avgRippleFE, avgRippleFI)
        print "--------------------------------------------------"
        
        # store results for this multiplier
        X[:, k] = [multiplier,
                   meanEr, maxEAC, tMaxEAC, maxEACR, tMaxEACR,
                   meanIr, maxIAC, tMaxIAC, maxIACR, tMaxIACR,
                   avgReplayInterval,
                   avgRippleFE, ripplePE, avgGammaFE, gammaPE,
                   avgRippleFI, ripplePI, avgGammaFI, gammaPI]

        # Plots
        plot_raster_ISI(spikeTimesE, spikingNeuronsE, [ISIhist, bin_edges], "blue", multiplier)
        plot_PSD(poprE, rEAC, fE, PxxE, "Pyr_population", "b-", multiplier)
        plot_PSD(poprI, rIAC, fI, PxxI, "Bas_population", "g-", multiplier)

        ymin, ymax= plot_zoomed(spikeTimesE, spikingNeuronsE, poprE, "Pyr_population", "blue", multiplier)
        plot_zoomed(spikeTimesI, spikingNeuronsI, poprI, "Bas_population", "green", multiplier, Pyr_pop=False)
        subset = select_subset(selection, ymin, ymax)
        plot_detailed(msMe, subset, multiplier, new_network=True)
        
        plt.close("all")

    else:  # if there is no activity the auto-correlation function will throw an error!

        print "No activity !"
        print "--------------------------------------------------"

        # store results for this multiplier
        X[:, k] = [multiplier, 0, np.nan, np.nan, np.nan, np.nan, 0, np.nan, np.nan, np.nan, np.nan,
		           np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]

    gc.collect()           


# ====================================== summary plots and saving ======================================

plot_summary_replay(multipliers, replay_interval=X[11, :], rateE=X[1, :], rateI=X[6, :])
plot_summary_AC(multipliers, maxACE=X[2, :], maxACI=X[7, :], maxRACE=X[4, :], maxRACI=X[9, :])
plot_summary_ripple(multipliers, rippleFE=X[12, :], rippleFI=X[16, :], ripplePE=X[13, :], ripplePI=X[17, :])
plot_summary_gamma(multipliers, gammaFE=X[14, :], gammaFI=X[18, :], gammaPE=X[15, :], gammaPI=X[19, :])

plt.close("all")


# Save result array (X)
fName= os.path.join(SWBasePath, "files", fOut)
header = 'Multiplier, Mean_exc.rate, Max.exc.AC., at[ms], Max.exc.AC.in_ripple_range, at[ms],' \
         'Mean_inh.rate, Max.inh.AC., at[ms], Max.inh.AC.in_ripple_range, at[ms],' \
         'avg. replay interval,' \
         'avgRippleFE, ripplePE, avgGammaFE, ripplePE,' \
         'avgRippleFI, ripplePI, avgGammaFI, ripplePI'
np.savetxt(fName, X, fmt='%.6f', delimiter='\t', header=header)                
                 