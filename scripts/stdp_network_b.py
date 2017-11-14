#!/usr/bin/python
# -*- coding: utf8 -*-
'''
loads in hippocampal like spike train (produced by generate_spike_train.py) and runs STD learning rule in a recurrent spiking neuron population
-> creates learned weight matrix for PC population, used by spw_network* scripts
see more: https://drive.google.com/file/d/0B089tpx89mdXZk55dm0xZm5adUE/view
authors: András Ecker, Eszter Vértes, Szabolcs Káli last update: 09.2015 (+ some minor checks for symmetric STDP in 03.2017)
'''

import brian_no_units
from brian import *
import os
import numpy as np
import matplotlib.pyplot as plt
from plots import *

fIn = "spikeTrainsR.npz"
fOut = "wmxR_asym_old.txt"

SWBasePath = os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-2])

np.random.seed(12345)

N = 4000  # #{neurons}

# importing spike times from file
fName = os.path.join(SWBasePath, "files", fIn)
npzFile = np.load(fName)
spikeTrains = npzFile["spikeTrains"]

spiketimes = []

for neuron in range(N):
    nrn = neuron * np.ones_like(spikeTrains[neuron])
    z = zip(nrn, spikeTrains[neuron])
    spiketimes.append(z)

spiketimes = [item for sublist in spiketimes for item in sublist]
print "spike times loaded"


PC = SpikeGeneratorGroup(N, spiketimes)


# STDP parameters
taup = taum = 20  # ms  # 20 - baseline
Ap = 0.01  # : 1 # asymmetric STDP rule
Am = -Ap  # : 1 # asymmetric STDP rule
#Ap = Am = 0.01  # : 1 # symmetric STDP rule
wmax = 40e-9  # S # asymmetric STDP rule
#wmax=7.5e-9  # S # symmetric STDP rule (orig taus)


def learning(spikingNeuronGroup, taup, taum, Ap, Am, wmax):
    """
    Takes a spiking group of neurons, connects the neurons sparsely with each other,
    and learns the weight 'pattern' via STDP:
    exponential STDP: f(s) = A_p * exp(-s/tau_p) (if s > 0), where s=tpost_{spike}-tpre_{spike}
    see more: https://brian.readthedocs.org/en/1.4.1/reference-plasticity.html#brian.ExponentialSTDP
    :param spikingNeuronGroup: Brian class of spiking neurons
    :param taup, taum: time constant of weight change
    :param Ap, Am: max amplitude of weight change
    :param wmax: maximum weight
    :return weightmx: numpy ndarray with the learned synaptic weights
            spikeM: SpikeMonitor of the network (for plotting and further analysis)
            mode_: ['asym', 'sym'] just for saving conventions (see saved wmx figures) 
    """
    
    mode_ = plot_STDP_rule(taup, taum, Ap*wmax*1e9, Am*wmax*1e9, "STDP_rule")

    Conn = Connection(spikingNeuronGroup, spikingNeuronGroup, weight=0.1e-9, sparseness=0.16)
    
    # symmetric STDP rule Ap = Am (see: Mishra et al. 2016 - 10.1038/ncomms11552)
    STDP = ExponentialSTDP(Conn, taup=taup*0.001, taum=taum*0.001,  # *0.001 - ms convertion
                           Ap=Ap, Am=Am, wmax=wmax,
                           interactions='all', update='additive')

    # run simulation
    spikeM = SpikeMonitor(spikingNeuronGroup, record=True)
    run(400, report='text')  # the generated spike train is 500 sec long...

    # weight matrix
    weightmx = [[Conn[i, j] for j in range(N)] for i in range(N)]
    tmp = np.asarray(weightmx)
    weightmx = np.reshape(tmp, (4000, 4000))
    np.fill_diagonal(weightmx, 0)

    return weightmx, spikeM, mode_


weightmx, spikeM, mode_ = learning(PC, taup, taum, Ap, Am, wmax)


# Plots: raster
figure(figsize=(10, 8))
raster_plot(spikeM, spacebetweengroups=1, title='Raster plot', newfigure=False)
#plt.show()

plot_wmx(weightmx, "wmx_%s"%mode_)
plot_wmx_avg(weightmx, 100, "wmx_avg_%s"%mode_)
plot_w_distr(weightmx, "w_distr_%s"%mode_)

selection = np.array([500, 1500, 2500, 3500])  # some random neuron IDs to save weigths
dWee = save_selected_w(weightmx, selection)
plot_weights(dWee, "sel_weights_%s"%mode_)


# save weightmatrix
fName = os.path.join(SWBasePath, 'files', fOut)
np.savetxt(fName, weightmx)
