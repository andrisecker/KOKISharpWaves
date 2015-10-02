#!/usr/bin/python
# -*- coding: utf8 -*-

import numpy as np
import os
from poisson_proc import inhomPoisson

SWBasePath = os.path.split(os.path.split(__file__)[0])[0]


theta = 7.0  # theta frequence [Hz]
avgRateInField = 20.0  # avg. in-field firing rate [ms]
vMice = 32.43567842  # velocity of the mice [cm/s]
lPlaceField = 30.0  # length of the place field [cm]
lRoute = 300.0  # circumference [cm]

nPop = 50  # # of populations
nNeuron = 80 # # of neurons in one population


spikeTrains = []

for pop in range(0, nPop):
    for neuron in range(0, nNeuron):
        tmp = inhomPoisson(0, pop, theta, avgRateInField, vMice, lPlaceField,lRoute)
        spikeTrains.append(tmp)
        print pop + 1, '/', neuron + 1, 'done from', nPop, '/', nNeuron


assert len(spikeTrains) == nPop * nNeuron

fOut = os.path.join(SWBasePath, 'files', 'spikeTrains.npz')
np.savez(fOut, spikeTrains=spikeTrains)
