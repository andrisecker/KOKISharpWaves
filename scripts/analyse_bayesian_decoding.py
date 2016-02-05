#!/usr/bin/python
# -*- coding: utf8 -*-


import numpy as np
import matplotlib.pyplot as plt
import os

fInSpikes = 'spikes.npz'
fIn = 'route_0.05.npz'

tempRes = 0.05  # [s]
spaRes = 2*np.pi / 360.0  # [rad] ( == 1 degree)
N = 4000

SWBasePath = os.path.split(os.path.split(__file__)[0])[0]

mplPars = { #'text.usetex'       :    True,
            'axes.labelsize'    :   'large',
            'font.family'       :   'serif',
            'font.sans-serif'   :   'computer modern roman',
            'font.size'         :    16,
            'xtick.labelsize'   :    14,
            'ytick.labelsize'   :    14
            }
for key, val in mplPars.items():
    plt.rcParams[key] = val


spatialPoints = np.linspace(0, 2*np.pi, int(2*np.pi / spaRes))
samplingTimes = np.linspace(0, 10, int(10.0 / tempRes)+1)


fName = os.path.join(SWBasePath, 'files', fInSpikes)
npzFile = np.load(fName)
spikes = npzFile['spikes']  # only for the raster plot

fName = os.path.join(SWBasePath, 'files', fIn)
npzFile = np.load(fName)
route = npzFile['route'].tolist()
ML = npzFile['ML']


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
ax2.set_ylabel(ylabel='Place', color='red')

ax.set_title('Place decoding from spike train')
ax.set_xlim([0, 10000])
ax.set_xlabel('Time [ms]')


# delete 0-s from route (whose appeared because of the lack of spikes)
correctedRoute = []
for ind, val in enumerate(route):
    if val != 0:
        correctedRoute.append(val)
    elif val == 0 and ML[ind] != 0:
        correctedRoute.append(val)

# w = dphi/dt
w = []
dt = tempRes
for phi1, phi2 in zip(correctedRoute[:-1], correctedRoute[1:]):
    if phi1 < phi2:
        w.append((phi2 - phi1) / dt)
    elif phi2 < phi1 and int(phi2) == 0:
        w.append((2*np.pi - phi1 + phi2) / dt)

np.asarray(w)

mu = 2*np.pi / 0.484  # this shouldn't be hard coded !!!
sigma = 1

fig2 = plt.figure(figsize=(12, 8))

ax = fig2.add_subplot(1, 1, 1)
n, bins, patches = ax.hist(w, 200, facecolor='red', normed=1)
ax.set_ylabel(ylabel='probability', color='red')

tmp = np.linspace(0, np.max(w), 5000)
ax2 = ax.twinx()
ax2.plot(tmp, 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-(tmp - mu)**2 / (2 * sigma**2)), color='green', lw=1.5)
ax2.set_ylabel(ylabel='predicted velocity', color='green')

ax.set_title('Distribution of the angular velocity (during replay)')
ax.set_xlabel('angular velocity (w) [rad/s]')
ax.set_xlim([0, np.max(w)])

plt.show()


