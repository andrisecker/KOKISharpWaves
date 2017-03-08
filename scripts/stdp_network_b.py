#!/usr/bin/python
# -*- coding: utf8 -*-

import brian_no_units
from brian import *
import numpy as np
import matplotlib.pyplot as plt
import os
from detect_oscillations import ripple, gamma

fIn = 'spikeTrainsR.npz'
fOut = 'wmxR_asym.txt'

SWBasePath =  '/home/bandi/workspace/KOKISharpWaves'  # os.path.split(os.path.split(__file__)[0])[0]

N = 4000  # #{neurons}

# importing spike times from file
fName = os.path.join(SWBasePath, 'files', fIn)
npzFile = np.load(fName)
spikeTrains = npzFile['spikeTrains']

spiketimes = []

for neuron in range(N):
    nrn = neuron * np.ones(len(spikeTrains[neuron]))
    z = zip(nrn, spikeTrains[neuron])
    spiketimes.append(z)

spiketimes = [item for sublist in spiketimes for item in sublist]

PC = SpikeGeneratorGroup(N, spiketimes)


def learning(spikingNeuronGroup):
    '''
    Takes a spiking group of neurons, connects the neurons sparsely with each other,
    and learns the weight 'pattern' via STDP
    :param spikingNeuronGroup: Brian class of spiking neurons
    :return weightmx: numpy ndarray? with the learned synaptic weights
            sp: SpikeMonitor of the network (for plotting and further analysis)
    '''

    Conn = Connection(spikingNeuronGroup, spikingNeuronGroup, weight=0.1e-9, sparseness=0.16)

    # f(s) = A_p * exp(-s/tau_p) (if s > 0)
    # A_p = 0.01, tau_p = 20e-3
    # see more: https://brian.readthedocs.org/en/1.4.1/reference-plasticity.html#brian.ExponentialSTDP
    stdp = ExponentialSTDP(Conn, 20e-3, 20e-3, 0.01, -0.01, wmax=40e-9, interactions='all', update='additive')  # asymmetric STDP
    
    # symmetric STDP rule (see: Mishra et al. 2016 - 10.1038/ncomms11552)
    #stdp = ExponentialSTDP(Conn, 20e-3, 20e-3, 0.01, 0.01, wmax=7.5e-9, interactions='all', update='additive')

    # run simulation
    sp = SpikeMonitor(spikingNeuronGroup, record=True)
    run(400, report='text')

    # weight matrix
    weightmx = [[Conn[i, j] for j in range(N)] for i in range(N)]

    return weightmx, sp


weightmx, sp = learning(PC)

tmp = np.asarray(weightmx)
weightmx = np.reshape(tmp, (4000, 4000))
np.fill_diagonal(weightmx, 0)

# Plots (cutted out drom diff. places, it might be ugly...)

#figure(figsize=(10, 8))
#raster_plot(sp, spacebetweengroups=1, title='Raster plot', newfigure=False)

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
