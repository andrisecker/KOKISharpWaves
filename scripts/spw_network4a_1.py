#!/usr/bin/python
# -*- coding: utf8 -*-
'''
crates PC (adExp IF) and BC (IF) population in Brian, loads in recurrent connection matrix for PC population 
runs simulation and checks the dynamics
see more: https://drive.google.com/file/d/0B089tpx89mdXZk55dm0xZm5adUE/view
authors: András Ecker, Eszter Vértes, Szabolcs Káli last update: 11.2015 (+ some minor checks for symmetric STDP in 03.2017)
'''

import os
from brian import *
import numpy as np
import matplotlib.pyplot as plt
from detect_oscillations import replay, ripple, gamma
from plots import *

fIn = 'wmxR_asym_gauss.txt'

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
dvm/dt = (-gL_Pyr*(vm-Vrest_Pyr) + gL_Pyr*delta_T_Pyr*exp((vm- theta_Pyr)/delta_T_Pyr)-w - (g_ampa*z*(vm-E_Exc) + g_gaba*z*(vm-E_Inh)))/Cm_Pyr : volt
dw/dt = (a_Pyr*(vm - Vrest_Pyr )-w)/tau_w_Pyr : amp
dg_ampa/dt = -g_ampa/tauSyn_PyrExc : 1
dg_gaba/dt = -g_gaba/tauSyn_PyrInh : 1
'''

def myresetfunc(P, spikes):
    P.vm[spikes] = reset_Pyr   # reset voltage
    P.w[spikes] += b_Pyr  # low pass filter of spikes (adaptation mechanism)

SCR = SimpleCustomRefractoriness(myresetfunc, tref_Pyr, state='vm')

eqs_bas = '''
dvm/dt = (-gL_Bas*(vm-Vrest_Bas) - (g_ampa*z*(vm-E_Exc) + g_gaba*z*(vm-E_Inh)))/Cm_Bas :volt
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
#poprext = PopulationRateMonitor(MF, bin=1*ms)
selection = np.arange(0, 4000, 100) # subset of neurons for recoring variables
msMe = MultiStateMonitor(PE, vars=['vm', 'w', 'g_ampa'], record=selection.tolist())  # comment this out later (takes a lot of memory!)
bins = [0*ms, 50*ms, 100*ms, 150*ms, 200*ms, 250*ms, 300*ms, 350*ms, 400*ms, 450*ms, 500*ms,
        550*ms, 600*ms, 650*ms, 700*ms, 750*ms, 800*ms, 850*ms, 900*ms, 950*ms, 1000*ms]
isi = ISIHistogramMonitor(PE, bins)
dWee = save_selected_w(Wee, selection)
del Wee  # cleary memory


run(10000*ms, report='text')


# Brian's raster + ISI plot
fig = plt.figure(figsize=(10, 8))

subplot(2, 1, 1)
raster_plot(sme, spacebetweengroups=1, title="Raster plot", newfigure=False)
xlim([0, 10000])
ylim([0, 4000])

subplot(2, 1, 2)
hist_plot(isi, title="ISI histogram", newfigure=False)
xlim([0, 1000])

fig.tight_layout()
figName = os.path.join(SWBasePath, "figures", "1.png")
fig.savefig(figName)

if np.max(popre.rate > 0):  # check if there is any activity

    avgReplayInterval = replay(isi.count[3:17])  # bins from 150 to 850 (range of interest)

    meanEr, rEAC, maxEAC, tMaxEAC, maxEACR, tMaxEACR, fE, PxxE, avgRippleFE, ripplePE = ripple(popre.rate, 1000)
    avgGammaFE, gammaPE = gamma(fE, PxxE)
    meanIr, rIAC, maxIAC, tMaxIAC, maxIACR, tMaxIACR, fI, PxxI, avgRippleFI, ripplePI = ripple(popri.rate, 1000)
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
    plot_PSD(popre.rate, rEAC, fE, PxxE, "Pyr_population", 'b-', multiplier_=1)
    plot_PSD(popri.rate, rIAC, fI, PxxI, "Bas_population", 'g-', multiplier_=1)

    ymin, ymax = plot_zoomed(popre.rate, sme.spikes, "Pyr_population", "blue", 'b-', multiplier_=1)
    _, _ = plot_zoomed(popri.rate, smi.spikes, "Bas_population", "green", 'g-', multiplier_=1)
    subset = select_subset(selection, ymin, ymax)
    plot_detailed(msMe, subset, dWee, multiplier_=1)
    plot_adaptation(msMe, selection, multiplier_=1)

else:  # if there is no activity the auto-correlation function will throw an error!
    
    print "No activity !"
    print "--------------------------------------------------"

plt.show()



