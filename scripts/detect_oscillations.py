#!/usr/bin/python
# -*- coding: utf8 -*-

import numpy as np
from scipy import signal, misc

def replay(isi):
    '''
    Decides if there is a replay or not:
    searches for the max # of spikes (and plus one bin one left- and right side)
    if the 90% of the spikes are in that 3 bins then it's periodic activity: replay
    :param isi: Inter Spike Intervals of the pyr. pop.
    :return avgReplayInterval: counted average replay interval
    '''

    binsROI = isi
    binMeans = np.linspace(175, 825, 14)
    maxInd = np.argmax(binsROI)

    if 1 <= maxInd <= len(binsROI) - 2:
        bins3 = binsROI[maxInd-1:maxInd+2]
        tmp = binsROI[maxInd-1]*binMeans[maxInd-1] + binsROI[maxInd]*binMeans[maxInd] + binsROI[maxInd+1]*binMeans[maxInd+1]
        avgReplayInterval = tmp / (binsROI[maxInd-1] + binsROI[maxInd] + binsROI[maxInd+1])
    else:
        bins3 = []

    if sum(int(i) for i in binsROI) * 0.9 < sum(int(i) for i in bins3):
        print 'Replay, avg. replay interval:', avgReplayInterval, '[ms]'
    else:
        avgReplayInterval = np.nan
        print 'Not replay'

    return avgReplayInterval


def ripple(rate):
    '''
    Decides if there is a high freq. ripple oscillation or not
    calculates the autocorrelation and the power spectrum of the activity
    and applies a Fisher g-test (on the spectrum) and if p value is smaller than 0.001 it's ripple
    :param rate: firing rate of the neuron population
    :return: meanr, rAC: mean rate, autocorrelation of the rate
             maxAC, tMaxAC: maximum autocorrelation, time interval of maxAC
             maxACR, tMaxAC: maximum autocorrelation in ripple range, time interval of maxACR
             f, Pxx: sample frequencies and power spectral density (results of PSD analysis)
             avgRippleF, rippleP: average frequency and power of the oscillation
    '''

    meanr = np.mean(rate)
    rUb = rate - meanr
    rVar = np.sum(rUb**2)
    rAC = np.correlate(rUb, rUb, mode='same') / rVar  # cross correlation of reub and reub -> autocorrelation
    rAC = rAC[len(rAC)/2:]

    maxAC = rAC[1:].max()
    tMaxAC = rAC[1:].argmax()+1
    maxACR = rAC[3:9].max()
    tMaxACR = rAC[3:9].argmax()+3

    # see more: http://docs.scipy.org/doc/scipy-dev/reference/generated/scipy.signal.welch.html
    fs = 1000
    f, Pxx = signal.welch(rate, fs, nperseg=512, scaling='spectrum')

    f = np.asarray(f)
    rippleS = np.where(145 < f)[0][0]
    rippleE = np.where(f < 250)[0][-1]
    f.tolist()
    PxxRipple = Pxx[rippleS:rippleE]

    # Fisher g-test
    fisherG = PxxRipple.max() / np.sum(PxxRipple)

    N = len(PxxRipple)
    upper = int(np.floor(1 / fisherG))
    I = []
    for i in range(1, upper):
        Nchoosei = misc.comb(N, i)
        I.append(np.power(-1, i-1) * Nchoosei * np.power((1-i*fisherG), N-1))
    pVal = np.sum(I)

    if pVal < 0.01:
        avgRippleF = f[PxxRipple.argmax() + rippleS]
    else:
        avgRippleF = np.nan

    power = sum(Pxx)
    tmp = sum(PxxRipple)
    rippleP = (tmp / power) * 100



    return meanr, rAC, maxAC, tMaxAC, maxACR, tMaxACR, f, Pxx, avgRippleF, rippleP

def gamma(f, Pxx):
    '''
    Decides if there is a gamma oscillation or not
    and applies a Fisher g-test (on the spectrum) and if p value is smaller than 0.001 it's gamma
    :param f: calculated frequecies of the power spectrum
    :param Pxx: power spectrum of the neural activity
    :return: avgGammaF, gammaP: average frequency and power of the oscillation
    '''

    f = np.asarray(f)
    gammaS = np.where(30 < f)[0][0]
    gammaE = np.where(f < 145)[0][-1]
    f.tolist()
    PxxGamma = Pxx[gammaS:gammaE]

    # Fisher g-test
    fisherG = PxxGamma.max() / np.sum(PxxGamma)

    N = len(PxxGamma)
    upper = int(np.floor(1 / fisherG))
    I = []
    for i in range(1, upper):
        Nchoosei = misc.comb(N, i)
        I.append(np.power(-1, i-1) * Nchoosei * np.power((1-i*fisherG), N-1))
    pVal = np.sum(I)

    if pVal < 0.01:
        avgGammaF = f[PxxGamma.argmax() + gammaS]
    else:
        avgGammaF = np.nan

    power = sum(Pxx)
    tmp = sum(PxxGamma)
    gammaP = (tmp / power) * 100

    return avgGammaF, gammaP