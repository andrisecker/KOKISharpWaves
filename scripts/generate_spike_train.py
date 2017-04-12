#!/usr/bin/python
# -*- coding: utf8 -*-

import numpy as np
import os
from poisson_proc import inhomPoisson

fOut = 'spikeTrainsR.npz'

nPop = 4000  # #{populations}
nNeuron = 1  # #{neurons in one population}
# mode = 'continuous'
mode = 'random'
# if nNeuron != 1 mode -> 'block'

SWBasePath = '/'.join(os.path.abspath(__file__).split('/')[:-2])

seed = 0

spikeTrains = []

if nNeuron != 1:  # mode = 'blocks'
    for pop in range(0, nPop):
        for neuron in range(0, nNeuron):
            spikeTrains.append(inhomPoisson(nPop, pop, 0.0, seed))
            print str(pop + 1), '/', str(neuron + 1)
            seed += 1

elif nNeuron == 1 and mode == 'continuous':
    neurons = np.linspace(0, nPop - 1, nPop)
    for neuron in neurons:
        spikeTrains.append(inhomPoisson(nPop, neuron, 0.0, seed))
        print str(neuron + 1)
        seed += 1

elif nNeuron == 1 and mode == 'random':
    phiStarts = {}
    for neuron in range(0, nPop):
        spikeTrains.append(inhomPoisson(nPop, neuron, 0.0, seed, phiStarts=phiStarts))
        print str(neuron + 1)
        seed += 1

    assert len(phiStarts.keys()) == nPop

    tmpList = phiStarts.values()
    indexing = [i[0] for i in sorted(enumerate(tmpList), key=lambda x:x[1])]
    tmpArray = np.asarray(spikeTrains)
    spikeTrains = tmpArray[indexing].tolist()

    # save place fields for further analysis
    tmpArray = np.asarray(tmpList)
    pfStarts = tmpArray[indexing].tolist()
    fName = os.path.join(SWBasePath, 'files', 'PFstarts.npz')
    np.savez(fName, pfStarts=pfStarts)

assert len(spikeTrains) == nPop * nNeuron

# save results to .npz
fName = os.path.join(SWBasePath, 'files', fOut)
np.savez(fName, spikeTrains=spikeTrains)

