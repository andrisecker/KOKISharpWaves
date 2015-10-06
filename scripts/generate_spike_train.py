#!/usr/bin/python
# -*- coding: utf8 -*-

import numpy as np
import os
import random
from poisson_proc import inhomPoisson

fOut = 'spikeTrainsR.npz'

SWBasePath = os.path.split(os.path.split(__file__)[0])[0]

nPop = 4000  # of populations
nNeuron = 1  # of neurons in one population

seed = 0

spikeTrains = []

if nNeuron != 1:

    for pop in range(0, nPop):
        for neuron in range(0, nNeuron):
            spikeTrains.append(inhomPoisson(float(nPop), 0.0, float(pop), seed))
            print str(pop + 1), '/', str(neuron + 1), 'done from', str(nPop), '/', str(nNeuron)
            seed += 1
else:

    neurons = np.linspace(0, nPop - 1, nPop)
    random.shuffle(neurons)
    for i, neuron in enumerate(neurons):
        spikeTrains.append(inhomPoisson(float(nPop), 0.0, float(neuron), seed))
        print str(i + 1), 'done from', str(nPop)
        seed += 1

assert len(spikeTrains) == nPop * nNeuron


# save results to .npz
fName = os.path.join(SWBasePath, 'files', fOut)
np.savez(fName, spikeTrains=spikeTrains)
