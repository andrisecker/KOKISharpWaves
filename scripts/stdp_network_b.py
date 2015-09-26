#!/usr/bin/python
# -*- coding: utf8 -*-

import brian_no_units
from brian import *
import numpy as np
import matplotlib.pyplot as plt
import os
import random

# mode = 'blocks'
# mode = 'continuous'
mode = 'random'
assert mode in ['blocks', 'continuous', 'random']

SWBasePath = os.path.split(os.path.split(__file__)[0])[0]

N = 4000

Npop = 50
Nn = N/Npop


#importing spike times from file
fName = os.path.join(SWBasePath, 'files', 'spikeTrain.txt')
f = file(fName, 'r')
data = [line.split() for line in f]
f.close()

spiketimes = []
for neuron in range(N):
    data[neuron] = [float(x) for x in data[neuron]]
    nrn = neuron * np.ones(len(data[neuron]))
    z = zip(nrn, data[neuron])
    spiketimes.append(z)

spiketimes = [item for sublist in spiketimes for item in sublist]


def learning(spikingNeuronGroup):
    '''
    Takes a spiking group of neurons, connects the neurons sparsely with each other,
    and learns a 'pattern' via STDP
    :param spikingNeuronGroup: Brian class of spiking neurons
    :return weightmx: numpy ndarray with the learned synaptic weights
    :return sp: SpikeMonitor of the network (for plotting and frther analysis)
    '''

    Conn = Connection(spikingNeuronGroup, spikingNeuronGroup, weight=0.1e-9, sparseness=0.16)
    # f(s) = A_p * exp(-s/tau_p) (if s > 0)
    # A_p = 0.01, tau_p = 20e-3
    # see more: https://brian.readthedocs.org/en/1.4.1/reference-plasticity.html#brian.ExponentialSTDP
    stdp = ExponentialSTDP(Conn, 20e-3, 20e-3, 0.01, -0.01, wmax=40e-9, interactions='all', update='additive')

    # run simulation
    sp = SpikeMonitor(spikingNeuronGroup, record=True)
    run(400, report='text')

    # weight matrix
    weightmx = [[Conn[i,j] for j in range(N)] for i in range(N)]

    return weightmx, sp


PC = SpikeGeneratorGroup(N, spiketimes)

if mode == 'blocks':
    for i in range(Npop):
        PC.subgroup(Nn)

    weightmx, sp = learning(PC)

    fOut = 'wmxB.txt'

elif mode == 'continuous':
    for i in range(N):
        PC.subgroup(1)

    weightmx, sp = learning(PC)

    fOut = 'wmxC.txt'

elif mode == 'random':
    rndList = list(range(N))
    random.shuffle(rndList)
    for i in rndList:
        PC.subgroup(1)

    weightmx, sp = learning(PC)

    fOut = 'wmxR.txt'


np.savetxt(os.path.join(SWBasePath, 'files', fOut), weightmx)

#plots
figure()
raster_plot(sp, spacebetweengroups=1, title='Raster plot', newfigure=False)

# fig2 = plt.figure(figsize=(10, 8))
# ax = fig2.add_subplot(1, 1, 1)
# ax.hist(sp.spiketimes[0], np.linspace(0, 0.5, 100))
# ax.set_title('Spike times of the 1st neuron')
# ax.set_xlabel('bins [ms]')
# ax.set_ylabel('number of spikes')


fig3 = plt.figure(figsize=(10, 8))
ax = fig3.add_subplot(1, 1, 1)
i = ax.imshow(weightmx, interpolation='None')
fig3.colorbar(i)
ax.set_title('Weight matrix')

plt.show()
