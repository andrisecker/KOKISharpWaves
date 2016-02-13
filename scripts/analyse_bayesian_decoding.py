#!/usr/bin/python
# -*- coding: utf8 -*-


import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import os
from detect_oscillations import autocorrelation, ripple, gamma

fInSpikes = 'spikes.npz'
fIn = 'route_0.005.npz'
figOut = 'baseline_'

tempRes = 0.005  # [s]
spaRes = 2*np.pi / 360.0  # [rad] ( == 1 degree)
N = 4000

spatialPoints = np.linspace(0, 2*np.pi, int(2*np.pi / spaRes))
samplingTimes = np.linspace(0, 10, int(10.0 / tempRes)+1)

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


fName = os.path.join(SWBasePath, 'files', fInSpikes)
npzFile = np.load(fName)
spikes = npzFile['spikes']  # only for the raster plot

fName = os.path.join(SWBasePath, 'files', fIn)
npzFile = np.load(fName)
route = npzFile['route'].tolist()
ML = npzFile['ML']

# data for raster plot, firing rate plot
popre = {}
spikingNeurons = []
spikeTimes = []

for i in spikes:
    if np.floor(i[1] * 1000) not in popre:  # ms scale
        popre[np.floor(i[1] * 1000)] = 1
    elif np.floor(i[1] * 1000) in popre:
        popre[np.floor(i[1] * 1000)] += 1

    spikingNeurons.append(i[0])
    spikeTimes.append(i[1] * 1000)  # ms scale

# rate correction
for i in range(0, 10000):
    if i not in popre:
        popre[i] = 0


# change 0-s to np.nan in route (whose appeared because of the lack of spikes)
# this should be somehow solved in bayesian_decoding.py ...
correctedRoute = []
for ind, val in enumerate(route):
    if val != 0:
        correctedRoute.append(val)
    elif val == 0 and ML[ind] != 0:
        correctedRoute.append(val)
    else:
        correctedRoute.append(np.nan)


# w = dphi/dt
dphi = []
for phi1, phi2 in zip(correctedRoute[:-1], correctedRoute[1:]):
    if phi1 < phi2:
        dphi.append(phi2 - phi1)
    elif phi2 < phi1 and int(phi2) == 0 and int(phi1) != 0:
        dphi.append(2*np.pi - phi1 + phi2)
    elif np.isnan(phi1) or np.isnan(phi2):
        dphi.append(np.nan)
    else:
        dphi.append(phi1 - phi2)
        # dphi.append(np.nan)

w = np.asarray(dphi) / tempRes


# plots

fig = plt.figure(figsize=(10, 8))

# Raster
ax = fig.add_subplot(1, 1, 1)
ax.scatter(spikeTimes, spikingNeurons, color='blue', marker='.', lw=0)
for t in samplingTimes:
    if t*1000 < 200:
        ax.plot((t*1000, t*1000), (0, 4000), c='0.25', lw=0.5)
ax.set_ylim([0, 4000])
ax.set_ylabel(ylabel='Neuron number', color='blue')
ax.set_title('Place decoding from spike train')
ax.set_xlim([0, 10000])
ax.set_xlabel('Time [ms]')

# infered place
ax2 = ax.twinx()
ax2.plot(samplingTimes[1:]*1000, route, color='red', marker='o', lw=0)
ax2.set_ylim([0, 2*np.pi])
ax2.set_ylabel(ylabel='Place', color='red')

figName = os.path.join(SWBasePath, 'figures', str(figOut)+str(tempRes)+'_place_decoding.jpg')
fig.savefig(figName)


# Time series
fig2 = plt.figure(figsize=(10, 8))

ax = fig2.add_subplot(3, 1, 1)
ax.plot(np.linspace(0, len(w)-1, len(w)), w, 'r-', lw=1.5)
ax.set_title('angular velocity (during replay)')
ax.set_ylabel('w = dphi/dt [rad/s]')
ax.set_xlabel('#{sampling time}')


# AutoCorrelation
tmp = np.nan_to_num(w)
wAC = autocorrelation(tmp)

ax2 = fig2.add_subplot(3, 1, 2)
ax2.plot(np.linspace(0, len(wAC)-1, len(wAC)), wAC, 'r-', lw=1.5)
ax2.set_title('Autocorrelogram of w')
ax2.set_ylabel('Autocorrelation')
ax2.set_ylim([-0.2, 0.4])
ax2.set_xlabel('#{sampling time}')


# PSD
fs = 1 / tempRes
# see more: http://docs.scipy.org/doc/scipy-dev/reference/generated/scipy.signal.welch.html
f, Pxx = signal.welch(np.nan_to_num(w), fs, nperseg=128, scaling='spectrum')
dbPxx = 10 * np.log10(Pxx / max(Pxx))

ax3 = fig2.add_subplot(3, 1, 3)
ax3.plot(f, dbPxx, 'r-', marker='o', lw=1.5)
ax3.set_title('Power Spectrum of w')
ax3.set_ylabel('PSD [dB]')
ax3.set_xlabel('Frequency [Hz]')

fig2.tight_layout()

figName = os.path.join(SWBasePath, 'figures', str(figOut)+str(tempRes)+'_angular_velocity.jpg')
fig2.savefig(figName)

# Histogram
wfiltered = filter(lambda x: x == x, w)  # removes np.nan-s from the array (to make a histogram)

fig3 = plt.figure(figsize=(10, 8))

ax = fig3.add_subplot(1, 1, 1)
n, bins, patches = ax.hist(wfiltered, 30, facecolor='red', normed=1)
ax.set_title('Distribution of angular velocity (during replay)')
ax.set_ylabel('Probability')
ax.set_xlabel('w = dphi/dt [rad/s]')

figName = os.path.join(SWBasePath, 'figures', str(figOut)+str(tempRes)+'_dist(w).jpg')
fig3.savefig(figName)

'''
# average speed
mu = 2*np.pi / 0.484  # this shouldn't be hard coded !!!
sigma = 1
'''

# -------------------------------------------------
# Analysis of firing rates (from spw_network....py)
# -------------------------------------------------

excRate = popre.values()

meanEr, rEAC, maxEAC, tMaxEAC, maxEACR, tMaxEACR, fE, PxxE, avgRippleFE, ripplePE = ripple(excRate, 1000)
avgGammaFE, gammaPE = gamma(fE, PxxE)


fig4 = plt.figure(figsize=(10, 8))

ax = fig4.add_subplot(3, 1, 1)
ax.plot(np.linspace(0, 9999, 10000), excRate, 'b-')
ax.set_title('Exc. population rate')
ax.set_xlabel('Time [ms]')

rEACPlot = rEAC[2:201]  # 500 - 5 Hz interval

ax2 = fig4.add_subplot(3, 1, 2)
ax2.plot(np.linspace(2, 200, len(rEACPlot)), rEACPlot, 'b-', lw=1.5)
ax2.set_title('Autocorrelogram (2-200 ms)')
ax2.set_xlabel('Time [ms]')
ax2.set_xlim([2, 200])
ax2.set_ylabel('Autocorrelation')

PxxEPlot = 10 * np.log10(PxxE / max(PxxE))

fE = np.asarray(fE)
rippleS = np.where(145 < fE)[0][0]
rippleE = np.where(fE < 250)[0][-1]
gammaS = np.where(30 < fE)[0][0]
gammaE = np.where(fE < 80)[0][-1]
fE.tolist()

PxxRipple = PxxE[rippleS:rippleE]
PxxGamma = PxxE[gammaS:gammaE]

fRipple = fE[rippleS:rippleE]
fGamma = fE[gammaS:gammaE]

PxxRipplePlot = 10 * np.log10(PxxRipple / max(PxxE))
PxxGammaPlot = 10 * np.log10(PxxGamma / max(PxxE))

ax3 = fig4.add_subplot(3, 1, 3)
ax3.plot(fE, PxxEPlot, 'b-', marker='o', linewidth=1.5)
ax3.plot(fRipple, PxxRipplePlot, 'r-', marker='o', linewidth=2)
ax3.plot(fGamma, PxxGammaPlot, 'k-', marker='o', linewidth=2)
ax3.set_title('Power Spectrum of exc. rate')
ax3.set_xlim([0, 500])
ax3.set_xlabel('Frequency [Hz]')
ax3.set_ylabel('PSD [dB]')

fig4.tight_layout()

figName = os.path.join(SWBasePath, 'figures', str(figOut)+str(tempRes)+'_firing_rate.jpg')
fig4.savefig(figName)

plt.show()


