#!/usr/bin/python
# -*- coding: utf8 -*-
'''
helper file to plot dynamics (and the weight matrix)
author: András Ecker, last update: 03.2017
'''

import os
import numpy as np
import matplotlib.pyplot as plt

SWBasePath = '/'.join(os.path.abspath(__file__).split('/')[:-2])
figFolder = os.path.join(SWBasePath, "figures")


def plot_PSD(rate, rippleAC, f, Pxx, title_, linespec_, multiplier_):
    """
    saves figure with rate, auto-correlation plot, and PSD
    :param rate: population rate (produced by Brian population rate monitor)
    :param rippleAC: auto-correlation function of the rate (returned by ripple())
    :param f, Pxx (returned by PSD analysis) see more: http://docs.scipy.org/doc/scipy-dev/reference/generated/scipy.signal.welch.html
    :param title_, linespec_, multiplier: outline and naming parameters
    """

    fig = plt.figure(figsize=(10, 8))

    ax = fig.add_subplot(3, 1, 1)
    ax.plot(np.linspace(0, 10000, len(rate)), rate, linespec_)
    ax.set_title("%s rate"%title_)
    ax.set_xlabel("Time (ms)")


    rEACPlot = rippleAC[2:201] # 500 - 5 Hz interval

    ax2 = fig.add_subplot(3, 1, 2)
    ax2.plot(np.linspace(2, 200, len(rEACPlot)), rEACPlot, linespec_)
    ax2.set_title("Autocorrelogram 2-200 ms")
    ax2.set_xlabel("Time (ms)")
    ax2.set_xlim([2, 200])
    ax2.set_ylabel("AutoCorrelation")


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
    ax3.set_title("Power Spectrum Density")
    ax3.set_xlim([0, 500])
    ax3.set_xlabel("Frequency (Hz)")
    ax3.set_ylabel("PSD (dB)")

    fig.tight_layout()

    figName = os.path.join(figFolder, "%s_%s.png"%(multiplier_, title_))
    fig.savefig(figName)
    
    
def plot_zoomed(rate, spikes, title_, color_, linespec_, multiplier_):
    """
    saves figure with zoomed in raster and rate (last 100ms)
    :param rate: population rate (produced by Brian population rate monitor)
    :param spikes: tuple with neuron IDs and spike times (produced by Brian spike monitor)
    :param title_, color_, linespec_, multiplier_: outline and naming parameters
    """
    
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
    ax.set_title("%s raster (last 100 ms)"%title_)
    ax.set_xlim([9900, 10000])
    ax.set_xlabel("Time (ms)")
    ax.set_ylim([ymin, ymax])
    ax.set_ylabel("Neuron number")

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(np.linspace(9900, 10000, len(rate[9900:10000])), rate[9900:10000], linespec_, linewidth=1.5)
    ax2.set_title("Rate (last 100 ms)")
    ax2.set_xlabel("Time (ms)")

    fig.tight_layout()

    figName = os.path.join(figFolder, "%s_%s_zoomed.png"%(multiplier_, title_))
    fig.savefig(figName)


def plot_wmx(wmx, saveName_):
    """
    Plots the weight matrix
    :param wmx: ndarray representing the weight matrix
    :param saveName_: name of saved img
    """
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)
    i = ax.imshow(wmx, cmap=plt.get_cmap("jet"))
    i.set_interpolation("nearest")  # set to "None" to less pixels and smooth, nicer figure
    fig.colorbar(i)
    ax.set_title("Learned synaptic weights")
    
    figName = os.path.join(figFolder, "%s.png"%saveName_)
    fig.savefig(figName)
  
  
def plot_wmx_avg(wmx, nPop, saveName_):
    """
    Plots the averaged weight matrix (better view as a whole)
    :param wmx: ndarray representing the weight matrix
    :param nPop: number of populations
    :param saveName_: name of saved img
    """ 
    
    assert 4000 % nPop == 0

    popSize = int(4000.0 / nPop)
    wmxM = np.zeros((100, 100))
    for i in range(nPop):
        for j in range(nPop):
            tmp = wmx[int(i*popSize):int((i+1)*popSize), int(j*popSize):int((j+1)*popSize)]
            wmxM[i, j] = np.mean(tmp)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)
    i = ax.imshow(wmxM, cmap=plt.get_cmap("jet"))
    i.set_interpolation("nearest")  # set to "None" to less pixels and smooth, nicer figure
    fig.colorbar(i)
    ax.set_title("Learned synaptic weights (avg.)")
    
    figName = os.path.join(figFolder, "%s.png"%saveName_)
    fig.savefig(figName)
    

def plot_w_distr(wmx, saveName_):
    """
    Plots the distribution of the weights
    :param wmx: ndarray representing the weight matrix
    :param saveName_: name of saved img
    """  
    
    # deleting nulls from wmx to plot the distribution of the weights
    tmp = wmx.tolist()
    wmx = [val for sublist in tmp for val in sublist]
    wmx = filter(lambda i: i != 0, wmx)
    wmx = np.array(wmx)
    log10wmx = np.log10(wmx)
    print "mean(nonzero weights):", np.mean(wmx)

    fig = plt.figure(figsize=(10, 8))

    ax = fig.add_subplot(2, 1, 1)
    ax.hist(wmx, bins=150)
    ax.set_title('Distriboution of synaptic weights')
    ax.set_xlabel('pyr-pyr synaptic weight strength [nS]')
    ax.set_ylabel('# of synapses (on logarithmic scale)')
    plt.yscale('log')

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.hist(log10wmx, bins=150, color='red')
    ax2.set_title('Distribution of synaptic weights')
    ax2.set_xlabel('log10(pyr-pyr synaptic weight strength) [nS]')
    ax2.set_ylabel('# of synapses (on logarithmic scale)')
    plt.yscale('log')

    fig.tight_layout()
    
    figName = os.path.join(figFolder, "%s.png"%saveName_)
    fig.savefig(figName)
    
    
def plot_STDP_rule(taup, taum, Ap, Am, saveName_):
    """
    Plots the STDP rule used for learning
    exponential STDP: f(s) = A_p * exp(-s/tau_p) (if s > 0), where s=tpost_{spike}-tpre_{spike}
    :param taup, taum: time constant of weight change
    :param Ap, Am: max amplitude of weight change
    """

    delta_t = np.linspace(-100, 100, 1000)
    delta_w = np.where(delta_t>0, Ap*np.exp(-delta_t/taup), Am*np.exp(delta_t/taum))
    
    fig = plt.figure(figsize=(10, 8))

    ax = fig.add_subplot(1, 1, 1)
    ax.plot(delta_t, delta_w)
    ax.set_title("STDP curve")
    ax.set_xlabel("delta_t /post-pre/ (ms)")
    ax.set_ylabel("delta_w (nS)")
    ax.set_ylim(-Ap*1.05, Ap*1.05)
    ax.set_xlim([-70, 70])
    ax.axhline(0, ls='-', c='k')
    
    figName = os.path.join(figFolder, "%s.png"%saveName_)
    fig.savefig(figName)
    plt.close()
     
