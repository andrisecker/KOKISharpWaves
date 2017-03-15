#!/usr/bin/python
# -*- coding: utf8 -*-
'''
helper file to plot dynamics
author: András Ecker, last update: 03.2017
'''

import os
import numpy as np
import matplotlib.pyplot as plt

SWBasePath = '/'.join(os.path.abspath(__file__).split('/')[:-2])
figFolder = os.path.join(SWBasePath, "figures")


def plot_PSD(rate, rippleAC, f, Pxx, title, linespec_, multiplier):
    '''
    saves figure with rate, auto-correlation plot, and PSD
    :param rate: population rate (produced by Brian population rate monitor)
    :param rippleAC: auto-correlation function of the rate (returned by ripple())
    :param f, Pxx (returned by PSD analysis) see more: http://docs.scipy.org/doc/scipy-dev/reference/generated/scipy.signal.welch.html
    :param title, linespec_, multiplier: outline and naming parameters
    '''

    fig = plt.figure(figsize=(10, 8))

    ax = fig.add_subplot(3, 1, 1)
    ax.plot(np.linspace(0, 10000, len(rate)), rate, linespec_)
    ax.set_title("%s rate"%title)
    ax.set_xlabel('Time [ms]')


    rEACPlot = rippleAC[2:201] # 500 - 5 Hz interval

    ax2 = fig.add_subplot(3, 1, 2)
    ax2.plot(np.linspace(2, 200, len(rEACPlot)), rEACPlot, linespec_)
    ax2.set_title('Autocorrelogram 2-200 ms')
    ax2.set_xlabel('Time [ms]')
    ax2.set_xlim([2, 200])
    ax2.set_ylabel('AutoCorrelation')


    f = np.asarray(f)
    rippleS = np.where(145 < f)[0][0]
    rippleE = np.where(f < 250)[0][-1]
    gammaS = np.where(30 < f)[0][0]
    gammaE = np.where(f < 80)[0][-1]
    f.tolist()
    # ripple range
    PxxRipple = Pxx[rippleS:rippleE]
    PxxGamma = Pxx[gammaS:gammaE]
    # gamma range
    fRipple = f[rippleS:rippleE]
    fGamma = f[gammaS:gammaE]
    
    PxxPlot = 10 * np.log10(Pxx / max(Pxx))
    PxxRipplePlot = 10 * np.log10(PxxRipple / max(Pxx))
    PxxGammaPlot = 10 * np.log10(PxxGamma / max(Pxx))

    ax3 = fig.add_subplot(3, 1, 3)
    ax3.plot(f, PxxPlot, linespec_, marker='o', linewidth=1.5)
    ax3.plot(fRipple, PxxRipplePlot, 'r-', marker='o', linewidth=2)
    ax3.plot(fGamma, PxxGammaPlot, 'k-', marker='o', linewidth=2)
    ax3.set_title('Power Spectrum Density')
    ax3.set_xlim([0, 500])
    ax3.set_xlabel('Frequency [Hz]')
    ax3.set_ylabel('PSD [dB]')

    fig.tight_layout()

    figName = os.path.join(figFolder, "%s_%s.png"%(multiplier, title))
    fig.savefig(figName)
    
    
def plot_zoomed(rate, spikes, title, color_, linespec_, multiplier):
    '''
    saves figure with zoomed in raster and rate (last 100ms)
    :param rate: population rate (produced by Brian population rate monitor)
    :param spikes: tuple with neuron IDs and spike times (produced by Brian spike monitor)
    :param title, color_, linespec_, multiplier: outline and naming parameters
    '''
    
    fig = plt.figure(figsize=(10, 8))

    spikingNeurons = [i[0] for i in spikes]
    spikeTimes = [i[1] for i in spikes]

    tmp = np.asarray(spikeTimes)
    ROI = np.where(tmp > 9.9)[0].tolist()
    rasterX = np.asarray(spikeTimes)[ROI] * 1000
    rasterY = np.asarray(spikingNeurons)[ROI]

    # boundaries 
    if rasterY.min()-50 > 0:
        ymin = rasterY.min()-50
    else:
        ymin = 0
    if rasterY.max()+50 < 4000:
        ymax = rasterY.max()+50
    else:
        ymax = 4000

    ax = fig.add_subplot(2, 1, 1)
    ax.scatter(rasterX, rasterY, c=color_, marker='.', lw=0)
    ax.set_title("%s raster (last 100 ms)"%title)
    ax.set_xlim([9900, 10000])
    ax.set_xlabel("Time [ms]")
    ax.set_ylim([ymin, ymax])
    ax.set_ylabel("Neuron number")

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(np.linspace(9900, 10000, len(rate[9900:10000])), rate[9900:10000], linespec_, linewidth=1.5)
    ax2.set_title("Rate (last 100 ms)")
    ax2.set_xlabel("Time [ms]")

    fig.tight_layout()

    figName = os.path.join(figFolder, "%s_%s_zoomed.png"%(multiplier, title))
    fig.savefig(figName)
    
    
