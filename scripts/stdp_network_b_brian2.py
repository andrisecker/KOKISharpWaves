#!/usr/bin/python
# -*- coding: utf8 -*-
'''
loads in hippocampal like spike train (produced by generate_spike_train.py) and runs STDP learning rule in a recurrent spiking neuron population
-> creates learned weight matrix for PC population, used by spw_network* scripts
see more: https://drive.google.com/file/d/0B089tpx89mdXZk55dm0xZm5adUE/view
!!! updated to produce sym stdp curve as reported in Mishra et al. 2016 - 10.1038/ncomms11552
author: András Ecker last update: 03.2017
'''

from brian2 import *
import os
import numpy as np
import matplotlib.pyplot as plt

fIn = 'spikeTrainsR.npz'
fOut = 'wmxR_sym.txt'

SWBasePath =  '/'.join(os.path.abspath(__file__).split('/')[:-2])

N = 4000  # #{neurons}

# importing spike times from file
fName = os.path.join(SWBasePath, 'files', fIn)
npzFile = np.load(fName)
spikeTrains = npzFile['spikeTrains']

# create 2 numpy arrays for Brian2's SpikeGeneratorGroup
spikingNrns = 0 * np.ones(len(spikeTrains[0]))
spikeTimes = np.asarray(spikeTrains[0])*1000  # *1000 for ms convertion
for neuron in range(1, N):
    nrn = neuron * np.ones(len(spikeTrains[neuron]))
    spikingNrns = np.concatenate((spikingNrns, nrn), axis=0)
    tmp = np.asarray(spikeTrains[neuron])*1000  # *1000 for ms convertion
    spikeTimes = np.concatenate((spikeTimes, tmp), axis=0)

print "spike times loaded"

# with default dt=100*us there are neurons whose emit more then 1 spike per timestep
PC = SpikeGeneratorGroup(N, spikingNrns, spikeTimes*ms, dt=10*us)  

def learning(spikingNeuronGroup):
    '''
    Takes a spiking group of neurons, connects the neurons sparsely with each other,
    and learns the weight 'pattern' via STDP
    :param spikingNeuronGroup: Brian class of spiking neurons
    :return weightmx: numpy ndarray with the learned synaptic weights
            SpikeM: SpikeMonitor of the network (for plotting and further analysis)
            STDP: Brian Synapse class (for plotting and further analysis)
    '''

    # exponential STDP: f(s) = A_p * exp(-s/tau_p) (if s > 0), where A_p = 0.01, tau_p = 20*ms and s=tpost_{spike}-tpre_{spike}
    # see more: http://brian2.readthedocs.io/en/stable/resources/tutorials/2-intro-to-brian-synapses.html   
    taup = taum = 20*ms
    Ap = Am = 0.01
    wmax = 7.5  # nS
    Ap *= wmax
    Am *= wmax
    # mimics Brian1's exponentialSTPD class, with interactions='all', update='additive'
    # see more on conversion: http://brian2.readthedocs.io/en/stable/introduction/brian1_to_2/synapses.html
    STDP = Synapses(spikingNeuronGroup, spikingNeuronGroup,
             '''
             w : 1
             dA_pre/dt = -Ap/taup : 1 (event-driven)
             dA_post/dt = -Am/taum : 1 (event-driven)
             ''',
             on_pre='''
             A_pre += Ap
             w = w + A_post
             w = clip(w + A_post, 0, wmax)
             ''',
             on_post='''
             A_post += Am
             w = w + A_pre
             w = clip(w + A_pre, 0, wmax)
             ''',
             dt=10*us)  # small dt is only to match the dt of SpikeGeneratorGroup
             
    STDP.connect(condition="i!=j", p=0.16)
    STDP.w = 0.1 # nS

    # run simulation
    SpikeM = SpikeMonitor(spikingNeuronGroup, record=True)
    #SM = StateMonitor(STDP, 'w', record=True)  # TODO: this throws segmentation error...
    run(100*second, report='text')  # the generated spike train is 500 sec long...
    
    # weight matrix
    weightmx = STDP.w[:, :, :]
    #weightmx = np.reshape(weightmx, (4000, 4000))
    # np.fill_diagonal(weightmx, 0)  # avoided by condition...

    return weightmx, SpikeM, STDP


weightmx, SpikeM, STDP = learning(PC)


fig0 = plt.figure(figsize=(10,8))

ax = fig0.add_subplot(1, 1, 1)
ax.plot(SP.t/ms, SP.i, '.b')
ax.set_title("raster plot")

plt.show()

"""
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(1, 1, 1)
i = ax.imshow(weightmx, interpolation='None')
fig.colorbar(i)
ax.set_title('Learned weight matrix')


# averaged figure! (better view as a whole...)
nPop = 100
popSize = 4000.0 / nPop
wmxM = np.zeros((100, 100))
for i in range(nPop):
    for j in range(nPop):
        tmp = weightmx[i*popSize:(i+1)*popSize, j*popSize:(j+1)*popSize]
        wmxM[i, j] = np.mean(tmp)


fig2 = plt.figure(figsize=(10, 8))
ax = fig2.add_subplot(1, 1, 1)
i = ax.imshow(wmxM, interpolation='None')
fig2.colorbar(i)
ax.set_title('Learned weight matrix (averaged)')


# deleting nulls from wmx to plot the distribution of the weights
wmx = weightmx.tolist()
wmx = [val for sublist in wmx for val in sublist]
wmx = filter(lambda i: i != 0, wmx)
wmx = np.array(wmx)
log10wmx = np.log10(wmx)
print("mean:", np.mean(wmx))


fig3 = plt.figure(figsize=(10, 8))

ax = fig3.add_subplot(2, 1, 1)
ax.hist(wmx, bins=150)
ax.set_title('Distriboution of synaptic weights')
ax.set_xlabel('pyr-pyr synaptic weight strength [nS]')
#ax.set_xlim([0, 28])
ax.set_ylabel('# of synapses (on logarithmic scale)')
plt.yscale('log')

ax2 = fig3.add_subplot(2, 1, 2)
ax2.hist(log10wmx, bins=150, color='red')
ax2.set_title('Distribution of synaptic weights')
ax2.set_xlabel('log10(pyr-pyr synaptic weight strength) [nS]')
ax2.set_ylabel('# of synapses (on logarithmic scale)')
#ax2.set_xlim(-200, 2)
plt.yscale('log')

fig3.tight_layout()

plt.show()

# save weightmatrix
fName = os.path.join(SWBasePath, 'files', fOut)
np.savetxt(fName, weightmx)
"""
