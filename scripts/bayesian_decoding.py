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
fOut = 'route_0.01.npz'
tempRes = 0.01  # [s]
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


# list of overlapping place fields
fName = os.path.join(SWBasePath, 'files', fInPF)
npzFile = np.load(fName)
pfStarts = npzFile['pfStarts'].tolist()

overlappingPFs = []
for pfStart in pfStarts:
    overlap = []
    pfEnd = np.mod(pfStart + phiPFRad, 2*np.pi)
    if pfStart < (2*np.pi - phiPFRad):
        overlap = [i for i, val in enumerate(pfStarts) if pfStart <= val and val < pfEnd]
    else:
        overlap = [i for i, val in enumerate(pfStarts) if pfStart <= val or val < pfEnd]

    overlappingPFs.append(overlap)


# calculate firing rates (\tau_i(x))  !!! calculate not estimate
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

# taking only the cells into account, whose have overlapping place fields with a cell, that fired in the bin
cellROI = []
binSpikes = []

for t1, t2 in zip(samplingTimes[:-1], samplingTimes[1:]):
    count = 0
    tmp = []  # will be a list of list (cells that have to be taken into account)
    for i in range(0, N):
        n_i = ((t1 < spiketimes[i]) & (spiketimes[i] < t2)).sum()  # #{spikes of the i-th cell in the bin}
        if n_i != 0:
            tmp.append(overlappingPFs[i])
            count += n_i
    tmp2 = list(set(sorted([item for sublist in tmp for item in sublist])))
    cellROI.append(tmp2)
    binSpikes.append(count)

print 'average spikes/bin:', np.mean(binSpikes)


delta_t = tempRes  # in s
route = []
ML = []

bin = 0
for t1, t2 in zip(samplingTimes[:-1], samplingTimes[1:]):
    likelihoods = []
    for indPhi in range(0, len(spatialPoints)):
        likelihood1 = 0
        likelihood2 = 0

        for i in cellROI[bin]:  # instead of "for i in range(0, N):"
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
        likelihoods = [np.nan if x == 0 else x for x in likelihoods]  # change 0s to np.nan
        if np.isnan(likelihoods).all():
            likelihoods[0] = 0

    # search for the maximum of the likelihoods in a given sampling time
    id = np.nanargmax(likelihoods)
    maxLikelihood = likelihoods[id]
    place = spatialPoints[id]
    route.append(place)
    ML.append(maxLikelihood)
    print 'sampling time:', str(t2 * 1000), '[ms]:', str(place), '[rad] ML:', maxLikelihood
    bin += 1


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
ax.scatter(spikeTimes, spikingNeurons, color='blue', marker='.', lw=0)
for t in samplingTimes:
    if t*1000 < 1000:
        ax.plot((t*1000, t*1000), (0, 4000), c='0.25', lw=0.5)
ax.set_ylim([0, 4000])
ax.set_ylabel(ylabel='Neuron number', color='blue')

ax2 = ax.twinx()
ax2.plot(samplingTimes[1:]*1000, route, color='red', marker='o', lw=0)
ax2.set_ylim([0, 2*np.pi])
ax2.set_ylabel(ylabel='place', color='red')

ax.set_title('Place decoding from spike train')
ax.set_xlim([0, 10000])
ax.set_xlabel('Time [ms]')

fName = os.path.join(SWBasePath, 'figures', 'place_decoding_'+str(tempRes)+'.jpg')
fig.savefig(fName)
