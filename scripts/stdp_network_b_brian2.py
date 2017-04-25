#!/usr/bin/python
# -*- coding: utf8 -*-
'''
loads in hippocampal like spike train (produced by generate_spike_train.py) and runs STD learning rule in a recurrent spiking neuron population
-> creates learned weight matrix for PC population, used by spw_network* scripts
see more: https://drive.google.com/file/d/0B089tpx89mdXZk55dm0xZm5adUE/view
!!! updated to produce sym stdp curve as reported in Mishra et al. 2016 - 10.1038/ncomms11552
author: András Ecker last update: 03.2017
'''

from brian2 import *
set_device('cpp_standalone')  # speed up the simulation with generated C++ code
from brian2tools import *  # just for spec. plotting
import os
import numpy as np
import matplotlib.pyplot as plt
from plots import plot_wmx, plot_wmx_avg, plot_w_distr, plot_STDP_rule

fIn = "spikeTrainsR.npz"
fOut = "wmxR_sym.txt"

SWBasePath =  '/'.join(os.path.abspath(__file__).split('/')[:-2])

N = 4000  # #{neurons}

# importing spike times from file
fName = os.path.join(SWBasePath, "files", fIn)
npzFile = np.load(fName)
spikeTrains = npzFile["spikeTrains"]

# create 2 numpy arrays for Brian2's SpikeGeneratorGroup
spikingNrns = 0 * np.ones(len(spikeTrains[0]))
spikeTimes = np.asarray(spikeTrains[0])
for neuron in range(1, N):
    nrn = neuron * np.ones(len(spikeTrains[neuron]))
    spikingNrns = np.concatenate((spikingNrns, nrn), axis=0)
    tmp = np.asarray(spikeTrains[neuron])
    spikeTimes = np.concatenate((spikeTimes, tmp), axis=0)

print "spike times loaded"


# with default dt=100*us there are neurons whose emit more then 1 spike per timestep
#TODO: check why! (generated spikes are is sec...)
PC = SpikeGeneratorGroup(N, spikingNrns, spikeTimes*second, dt=35*us)  


# STDP parameters
taup = taum = 20*ms
Ap = Am = 0.01  # : 1
wmax = 7.5e-9 # S (w is dimensionless in the equations)
Ap *= wmax
Am *= wmax


def learning(spikingNeuronGroup, taup, taum, Ap, Am, wmax):
    """
    Takes a spiking group of neurons, connects the neurons sparsely with each other,
    and learns the weight 'pattern' via STDP:
    exponential STDP: f(s) = A_p * exp(-s/tau_p) (if s > 0), where s=tpost_{spike}-tpre_{spike}
    :param spikingNeuronGroup: Brian class of spiking neurons
    :param taup, taum: time constant of weight change
    :param Ap, Am: max amplitude of weight change
    :param wmax: maximum weight
    :return weightmx: numpy ndarray with the learned synaptic weights
            spikeM: SpikeMonitor of the network (for plotting and further analysis)
            mode_: ['asym', 'sym'] just for saving conventions (see saved wmx figures) 
    """

    mode_ = plot_STDP_rule(taup/ms, taum/ms, Ap/1e-9, Am/1e-9, "STDP_rule")

    # mimics Brian1's exponentialSTPD class, with interactions='all', update='additive'
    # see more on conversion: http://brian2.readthedocs.io/en/stable/introduction/brian1_to_2/synapses.html
    STDP = Synapses(spikingNeuronGroup, spikingNeuronGroup,
             """
             w : 1
             dA_pre/dt = -A_pre/taup : 1 (event-driven)
             dA_post/dt = -A_post/taum : 1 (event-driven)
             """,
             on_pre="""
             A_pre += Ap
             w = clip(w + A_post, 0, wmax)
             """,
             on_post="""
             A_post += Am
             w = clip(w + A_pre, 0, wmax)
             """)#,
             #dt=35*us)  # small dt is only to match the dt of SpikeGeneratorGroup
             
    STDP.connect(condition="i!=j", p=0.16)
    STDP.w = 0.1e-9 # S


    # run simulation
    spikeM = SpikeMonitor(spikingNeuronGroup, record=True)
    run(400*second, report='text')  # the generated spike train is 500 sec long...
    
    # weight matrix
    weightmx = np.zeros((4000, 4000))
    weightmx[STDP.i[:], STDP.j[:]] = STDP.w[:]

    return weightmx, spikeM, mode_


weightmx, spikeM, mode_ = learning(PC, taup, taum, Ap, Am, wmax)

# Plots: raster
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(1, 1, 1)
brian_plot(spikeM, axes=ax)
ax.set_title("Raster plot")
#plt.show()

plot_wmx(weightmx, "wmx_%s"%mode_)
plot_wmx_avg(weightmx, 100, "wmx_avg_%s"%mode_)
plot_w_distr(weightmx, "w_distr_%s"%mode_)

selection = np.array([500, 1500, 2500, 3500])  # some random neuron IDs to save weigths
dWee = save_selected_w(Wee, selection)
plot_weights(dWee, "sel_weights_%s"%mode_)


# save weightmatrix
fName = os.path.join(SWBasePath, 'files', fOut)
np.savetxt(fName, weightmx)
