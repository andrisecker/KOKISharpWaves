#!/usr/bin/python
# -*- coding: utf8 -*-
'''
looped version of spw_network4a_1.py -> checks the dynamics for different multipliers of the learned weight matrix
see more: https://drive.google.com/file/d/0B089tpx89mdXZk55dm0xZm5adUE/view
authors: András Ecker, Eszter Vértes, Szabolcs Káli last update: 05.2017
'''

import os
import gc
from brian import *
import numpy as np
import matplotlib.pyplot as plt
from detect_oscillations import *
from plots import *


fIn = 'wmxR_asym.txt'
fOut = 'results_asym_v02.txt'

SWBasePath = os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-2])

first = 0.5
last = 2.5
dataPoints = 21

multipliers = np.linspace(first, last, dataPoints)

NE = 4000
NI = 1000

eps_pyr = 0.16
eps_bas = 0.4

z = 1*nS
gL_Pyr = 4.333e-3 * uS  # 3.3333e-3
tauMem_Pyr = 60.0 * ms
Cm_Pyr = tauMem_Pyr * gL_Pyr
Vrest_Pyr = -70.0 * mV
reset_Pyr = -53.0*mV
theta_Pyr = -50.0*mV
tref_Pyr = 5*ms
# Adaptation parameters for pyr cells
a_Pyr = -0.8*nS    # nS subthreshold adaptation conductance
# moves threshold up
b_Pyr = 0.04*nA     # nA    Spike-triggered adaptation
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

J_PyrInh = 0.15
J_BasExc = 4.5  # 5.2083
J_BasInh = 0.25

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
delay_PyrExc = 3.0*ms
delay_PyrInh = 1.5*ms
delay_BasExc = 3.0*ms
delay_BasInh = 1.5*ms

# outer input
J_PyrMF = 5.0  # synapse strength
p_rate_mf = 5.0*Hz


# Creating populations
eqs_adexp = '''
dvm/dt=(gL_Pyr*(Vrest_Pyr-vm)+gL_Pyr*delta_T_Pyr*exp((vm- theta_Pyr)/delta_T_Pyr)-w -(g_ampa*z*(vm-E_Exc)+g_gaba*z*(vm-E_Inh)))/Cm_Pyr : volt
dw/dt=(a_Pyr*(vm- Vrest_Pyr )-w)/tau_w_Pyr : amp
dg_ampa/dt = -g_ampa/tauSyn_PyrExc : 1
dg_gaba/dt = -g_gaba/tauSyn_PyrInh : 1
'''

def myresetfunc(P, spikes):
    P.vm_[spikes] = reset_Pyr   # reset voltage
    P.w_[spikes] += b_Pyr  # low pass filter of spikes (adaptation mechanism)

eqs_bas = '''
dvm/dt = (-gL_Bas*(vm-Vrest_Bas)-(g_ampa*z*(vm-E_Exc)+g_gaba*z*(vm-E_Inh)))/Cm_Bas : volt
dg_ampa/dt = -g_ampa/tauSyn_BasExc : 1
dg_gaba/dt = -g_gaba/tauSyn_BasInh : 1
'''


# ====================================== end of parameters ======================================

# load in Wee only once ...
def load_Wee(fName):  # this way the file will closed and memory will cleaned
    """dummy function, just to make python clear the memory"""
    Wee = np.genfromtxt(fName) * 1e9
    np.fill_diagonal(Wee, 0)  # just to make sure

    print "weight matrix loaded"
    return Wee

fName = os.path.join(SWBasePath, 'files', fIn)
Wee = load_Wee(fName)

X = np.zeros((20, dataPoints))  # init. container to store results

# ====================================== iterates over diff. multipliers ======================================

seed = 12345
for k in range(0, dataPoints):

    multiplier = multipliers[k]
    print "multiplier=%s"%multiplier

    np.random.seed(seed)

    # recreate the neurons in every iteration (just to make sure!)
    SCR = SimpleCustomRefractoriness(myresetfunc, tref_Pyr, state='vm')
    PE = NeuronGroup(NE, model=eqs_adexp, threshold=v_spike_Pyr, reset=SCR)
    PI = NeuronGroup(NI, model=eqs_bas, threshold=theta_Bas, reset=reset_Bas, refractory=tref_Bas)
    PE.vm = Vrest_Pyr
    PE.g_ampa = 0
    PE.g_gaba = 0
    PI.vm = Vrest_Bas
    PI.g_ampa = 0
    PI.g_gaba = 0

    MF = PoissonGroup(NE, p_rate_mf)

    print "Connecting the network"

    Cext = IdentityConnection(MF, PE, 'g_ampa', weight=J_PyrMF)
    Cee = Connection(PE, PE, 'g_ampa', delay=delay_PyrExc)
    Wee_tmp = Wee * multiplier  # Wee matrix loaded before the for loop
    Cee.connect(PE, PE, Wee_tmp)
    Cei = Connection(PE, PI, 'g_ampa', weight=J_BasExc, sparseness=eps_pyr, delay=delay_BasExc)
    Cie = Connection(PI, PE, 'g_gaba', weight=J_PyrInh, sparseness=eps_bas, delay=delay_PyrInh)
    Cii = Connection(PI, PI, 'g_gaba', weight=J_BasInh, sparseness=eps_bas, delay=delay_BasInh)

    print "Connections done"
    del Wee_tmp  # cleary memory

    # init monitors
    sme = SpikeMonitor(PE)
    smi = SpikeMonitor(PI)
    popre = PopulationRateMonitor(PE)
    popri = PopulationRateMonitor(PI)
    # other monitors factored out to speed up simulation and make the process compatible with Brian2
    selection = np.arange(0, 4000, 100) # subset of neurons for recoring variables
    msMe = MultiStateMonitor(PE, vars=['vm', 'w', 'g_ampa', 'g_gaba'], record=selection.tolist())  # comment this out later (takes a lot of memory!)


    run(10000*ms, report='text')  # run the simulation!


    if sme.nspikes > 0 and smi.nspikes > 0:  # check if there is any activity
    
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
        plot_PSD(poprE, rEAC, fE, PxxE, "Pyr_population", 'b-', multiplier)
        plot_PSD(poprI, rIAC, fI, PxxI, "Bas_population", 'g-', multiplier)

        ymin, ymax = plot_zoomed(spikeTimesE, spikingNeuronsE, poprE, "Pyr_population", "blue", multiplier)
        plot_zoomed(spikeTimesI, spikingNeuronsI, poprI, "Bas_population", "green", multiplier, Pyr_pop=False)
        subset = select_subset(selection, ymin, ymax)
        plot_detailed(msMe, subset, multiplier)
        #plot_adaptation(msMe, selection, multiplier)

        plt.close('all')

    else:  # if there is no activity the auto-correlation function will throw an error!

        print "No activity !"
        print "--------------------------------------------------"

        # store results for this multiplier
        X[:, k] = [multiplier, 0, np.nan, np.nan, np.nan, np.nan, 0, np.nan, np.nan, np.nan, np.nan,
		           np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]

    # Reinitialize variables
    reinit(states=True)  # population is recreated so this might be useless
    reinit_default_clock()  # population is recreated so this might be useless
    clear(True)
    gc.collect()
    seed += 1


# ====================================== summary plots and saving ======================================


fig = plt.figure(figsize=(10, 8))

ax = fig.add_subplot(2, 1, 1)
ax.plot(multipliers, X[11, :], linewidth=2, marker='|')
ax.set_title('Average replay interval')
ax.set_xlim(first, last)
ax.set_ylabel('Time (ms)')

ax2 = fig.add_subplot(2, 1, 2)
ax3 = ax2.twinx()
ax2.plot(multipliers, X[1, :], 'b-', linewidth=2, marker='|')
ax2.set_ylabel(ylabel='PC (exc.) rate', color='blue')
ax3.plot(multipliers, X[6, :], 'g-', linewidth=2, marker='|')
ax3.set_ylabel(ylabel='BC (inh.) rate', color='green')
ax2.set_xlabel('scale factors')
ax2.set_xlim(first, last)
ax2.set_title('Mean firing rates')

fig.tight_layout()
fig.savefig(os.path.join(SWBasePath, 'figures', 'replay_and_firing_rates.png'))


fig2 = plt.figure(figsize=(10, 8))

ax = fig2.add_subplot(2, 1, 1)
ax.plot(multipliers, X[2, :], label='PC (exc.)', color='blue', linewidth=2, marker='|')
ax.plot(multipliers, X[7, :], label='BC (inh.)', color='green', linewidth=2, marker='|')
ax.set_xlim(first, last)
ax.set_title('Maximum autocerrelations')
ax.legend()

ax2 = fig2.add_subplot(2, 1, 2)
ax2.plot(multipliers, X[4, :], label='PC (exc.)', color='blue', linewidth=2, marker='|')
ax2.plot(multipliers, X[9, :], label='BC (inh.)', color='green', linewidth=2, marker='|')
ax2.set_xlim(first, last)
ax2.set_title('Maximum autocerrelations in ripple range')
ax2.set_xlabel('scale factors')
ax2.legend()

fig2.tight_layout()
fig2.savefig(os.path.join(SWBasePath, 'figures', 'autocorrelations.png'))


fig3 = plt.figure(figsize=(10, 8))

ax = fig3.add_subplot(2, 1, 1)
ax.plot(multipliers, X[12, :], label='Ripple freq (exc.)', color='blue', linewidth=2, marker='o')
ax2 = ax.twinx()
ax2.plot(multipliers, X[13, :], label='Ripple power (exc.)', color='red', linewidth=2, marker='|')
ax.set_xlim(first, last)
ax.set_ylabel(ylabel='freq [Hz]', color='blue')
ax.set_ylim([np.nanmin(X[12, :])-5, np.nanmax(X[12, :])+8])
ax2.set_ylabel(ylabel='power %', color='red')
ax2.set_ylim([0, 100])
ax.set_title('Ripple oscillation')
ax.legend(loc=2)
ax2.legend()

ax3 = fig3.add_subplot(2, 1, 2)
ax3.plot(multipliers, X[16, :], label='Ripple freq (inh.)', color='green', linewidth=2, marker='o')
ax4 = ax3.twinx()
ax4.plot(multipliers, X[17, :], label='Ripple power (inh.)', color='red', linewidth=2, marker='|')
ax3.set_xlim(first, last)
ax3.set_ylabel(ylabel='freq [Hz]', color='green')
ax3.set_ylim([np.nanmin(X[16, :])-5, np.nanmax(X[16, :])+8])
ax4.set_ylabel(ylabel='power %', color='red')
ax4.set_ylim([0, 100])
ax3.set_xlabel('scale factors')
ax3.legend(loc=2)
ax4.legend()

fig3.tight_layout()
fig3.savefig(os.path.join(SWBasePath, 'figures', 'ripple.png'))


fig4 = plt.figure(figsize=(10, 8))

ax = fig4.add_subplot(2, 1, 1)
ax.plot(multipliers, X[14, :], label='Gamma freq (exc.)', color='blue', linewidth=2, marker='o')
ax2 = ax.twinx()
ax2.plot(multipliers, X[15, :], label='Gamma power (exc.)', color='red', linewidth=2, marker='|')
ax.set_xlim(first, last)
ax.set_ylabel(ylabel='freq [Hz]', color='blue')
ax.set_ylim([np.nanmin(X[14, :])-5, np.nanmax(X[14, :])+8])
ax2.set_ylabel(ylabel='power %', color='red')
ax2.set_ylim([0, 100])
ax.set_title('Gamma oscillation')
ax.legend(loc=2)
ax2.legend()

ax3 = fig4.add_subplot(2, 1, 2)
ax3.plot(multipliers, X[18, :], label='Gamma freq (inh.)', color='green', linewidth=2, marker='o')
ax4 = ax3.twinx()
ax4.plot(multipliers, X[19, :], label='Gamma power (inh.)', color='red', linewidth=2, marker='|')
ax3.set_xlim(first, last)
ax3.set_ylabel(ylabel='freq [Hz]', color='green')
ax3.set_ylim([np.nanmin(X[18, :])-5, np.nanmax(X[18, :])+8])
ax4.set_ylabel(ylabel='power %', color='red')
ax4.set_ylim([0, 100])
ax3.set_xlabel('scale factors')
ax3.legend(loc=2)
ax4.legend()

fig4.tight_layout()
fig4.savefig(os.path.join(SWBasePath, 'figures', 'gamma.png'))

plt.close('all')

if len(fIn) > 8:  # if not the original matrix (wmxR.txt) is used - save the figure of the matrix...

    fName = os.path.join(SWBasePath, 'files', fIn)
    wmxM = load_Wee(fName)

    plot_wmx_avg(wmxM, 100, "wmx")
    plt.close()


# Save result array (X)
fName= os.path.join(SWBasePath, 'files', fOut)
header = 'Multiplier, Mean_exc.rate, Max.exc.AC., at[ms], Max.exc.AC.in_ripple_range, at[ms],' \
         'Mean_inh.rate, Max.inh.AC., at[ms], Max.inh.AC.in_ripple_range, at[ms],' \
         'avg. replay interval,' \
         'avgRippleFE, ripplePE, avgGammaFE, ripplePE,' \
         'avgRippleFI, ripplePI, avgGammaFI, ripplePI'
np.savetxt(fName, X, fmt='%.6f', delimiter='\t', header=header)
