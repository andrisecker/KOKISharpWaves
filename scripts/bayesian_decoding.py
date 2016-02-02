#!/usr/bin/python
# -*- coding: utf8 -*-

# This code is based on: T.Davidson, F.Kloosterman, M.Wilson "Hippocampal replay of extended experience",
# in Neuron, vol. 63, pp. 497-507, 2009
# difference: \tau_i(x) (rate parameters) are known (from poisson_proc.py and generate_spike_train.py)

import numpy as np
from scipy.misc import factorial
import matplotlib.pyplot as plt
import os

fInSpikes = 'spikes.npz'
fInPF = 'PFstarts.npz'
fOut = 'route.npz'

tempRes = 0.05  # [s]
spaRes = 2*np.pi / 360.0  # [rad] ( == 1 degree)
N = 4000

SWBasePath = os.path.split(os.path.split(__file__)[0])[0]

spatialPoints = np.linspace(0, 2*np.pi, int(2*np.pi / spaRes))
samplingTimes = np.linspace(0, 10, int(10.0 / tempRes)+1)

# (constants from poisson_proc.py:)
lRoute = 300  # circumference [cm]
lPlaceField = 30  # [cm]
r = lRoute / (2 * np.pi)  # radius [cm]
phiPFRad = lPlaceField / r  # (angle of) place field [rad]
avgRateInField = 20.0  # avg. in-field firing rate [Hz]


# calculate firing rates (\tau_i(x))  !!! calculate not estimate
fName = os.path.join(SWBasePath, 'files', fInPF)
npzFile = np.load(fName)
pfStarts = npzFile['pfStarts']

rates = []

for i in range(0, N):
    tau = np.zeros((1, int(2*np.pi / spaRes)))

    pfEnd = np.mod(pfStarts[i] + phiPFRad, 2*np.pi)
    mPF = pfStarts[i] + phiPFRad / 2

    for ind, phi in enumerate(spatialPoints):
        if pfStarts[i] < pfEnd:
            if pfStarts[i] <= phi and phi < pfEnd:
                tau[0][ind] = np.cos((2*np.pi) / (2 * phiPFRad) * (phi - mPF)) * avgRateInField
        else:
            if pfStarts[i] <= phi or phi < pfEnd:
                tau[0][ind] = np.cos((2*np.pi) / (2 * phiPFRad) * (phi - mPF)) * avgRateInField

    rates.append(tau)

print 'rates calculated'


# read spike times
fName = os.path.join(SWBasePath, 'files', fInSpikes)
npzFile = np.load(fName)
spikes = npzFile['spikes']  # only for the raster plot
spiketimes = npzFile['spiketimes']


# log(likelihood): log(Pr(spikes|x)) = \sum_{i=1}^N n_ilog(\frac{\Delta t \tau_i(x)}{n_i!}) - \Delta t \sum_{i=1}^N \tau_i(x)
delta_t = tempRes  # in s

route = []
ML = []
for t1, t2 in zip(samplingTimes[:-1], samplingTimes[1:]):
    likelihoods = []
    for indPhi in range(0, len(spatialPoints)):
        likelihood1 = 0
        likelihood2 = 0

        for i in range(0, N):
            tmp = 0

            n_i = ((t1 < spiketimes[i]) & (spiketimes[i] < t2)).sum()  # #{spikes of the i-th cell in the bin}
            tau_i_phi = rates[i][0, indPhi]  # firing rate of the i-th cell in a given position (on the circle)
            if tau_i_phi != 0 and n_i != 0:  # because log() can't take 0
                tmp = n_i * np.log(delta_t * tau_i_phi / factorial(n_i).item())
                # .item() is needed because factorial gives 0-d array

            likelihood1 += tmp
            likelihood2 += tau_i_phi

        likelihood = likelihood1 - delta_t * likelihood2
        likelihoods.append(likelihood)

    # search for the maximum of the likelihoods in a given sampling time
    id = np.argmax(likelihoods)
    maxLikelihood = likelihoods[id]
    place = spatialPoints[id]
    route.append(place)
    ML.append(maxLikelihood)
    print 'sampling time:', str(t2 * 1000), '[ms]:', str(place), '[rad] ML:', maxLikelihood

fName = os.path.join(SWBasePath, 'files', fOut)
np.savez(fName, route=route, ML=ML)


# raster plot
spikingNeurons = []
spikeTimes = []

for i in spikes:
    spikingNeurons.append(i[0])
    spikeTimes.append(i[1] * 1000)  # ms scale

fig = plt.figure(figsize=(10, 8))

ax = fig.add_subplot(1, 1, 1)
ax.scatter(spikeTimes, spikingNeurons, c='blue', marker='.', lw=0)
ax.set_title('Raster plot')
ax.set_xlim([0, 10000])
ax.set_xlabel('Time [ms]')
ax.set_ylim([0, 4000])
ax.set_ylabel('Neuron number')

plt.show()