#!/usr/bin/python
# -*- coding: utf8 -*-
"""
helper functions for generating hippocampal like spike trains
authors: András Ecker, Eszter Vértes, Szabolcs Káli last update: 05.2017
"""

import numpy as np


def generateFiringRate(nPop, pop, phase0, t, **kwargs):
    '''
    Calculates the lambda parameter of the Poisson process, that represent the firing rate in CA3 (pyr. cells)
    (takes preferred place and phase precession into account)
    :param nPop: # of neuron populations
    :param t: time
    :param pop: current population
    :param phase0: initial phase (used for modeling phase precession)
    :return: lambda: calculated lambda parameter of the Poisson process
    '''

    theta = 7.0  # theta frequence [Hz]
    avgRateInField = 20.0  # avg. in-field firing rate [Hz]
    vMice = 32.43567842  # velocity of the mice [cm/s]
    lRoute = 300.0  # circumference [cm]
    lPlaceField = 30.0  # length of the place field [cm]

    r = lRoute / (2 * np.pi)  # radius [cm]
    phiPFRad = lPlaceField / r  # (angle of) place field [rad]

    tRoute = lRoute / vMice  # [s]
    wMice = 2 * np.pi / tRoute  # angular velocity
    x = np.mod(wMice * t, 2 * np.pi)  # position of the mice [rad]

    if "phiStarts" not in kwargs.keys():
        phiStart = (float(pop) / float(nPop)) * 2 * np.pi
    else:
        phiStarts = kwargs["phiStarts"]
        if pop not in phiStarts.keys():
            phiStart = np.random.rand(1)[0] * 2 * np.pi
            phiStarts[pop] = phiStart
        else:
            phiStart = phiStarts[pop]

    phiEnd = np.mod(phiStart + phiPFRad, 2 * np.pi)

    # phase precession
    y = phase0 + 2 * np.pi * theta * t
    mPF = phiStart + phiPFRad / 2
    m = - (x - phiStart) * 2 * np.pi / phiPFRad  # prefered phase: f(current position within the place field)

    sig = 0.5  # deviation of phase-locking (von Misses distribution -> see lambda2)
    s = 1.0 / sig

    if phiStart < phiEnd:
        if phiStart <= x and x < phiEnd:  # if the mice is in the place field
            lambda1 = np.cos((2 * np.pi) / (2 * phiPFRad) * (x - mPF)) * avgRateInField
            lambda2 = np.exp(s * np.cos(y - m)) / np.exp(s)
        else:
            lambda1 = 0
            lambda2 = 1

        lambdaP = lambda1 * lambda2

    else:
        if phiStart <= x or x < phiEnd:  # if the mice is in the place field
            lambda1 = np.cos((2 * np.pi) / (2 * phiPFRad) * (x - mPF)) * avgRateInField
            lambda2 = np.exp(s * np.cos(y - m)) / np.exp(s)
        else:
            lambda1 = 0
            lambda2 = 1

        lambdaP = lambda1 * lambda2

    return lambdaP


def inhomPoisson(nPop, pop, phase0, seed, **kwargs):
    '''
    Makes a homogenous Poisson process and transfer it to inhomogenous via deleting spikes
    (based on an other Poisson process made by generateFiringRates)
    :param nPop: # of neuron populations (see generateFiringRate)
    :param pop: current population (see generateFiringRate)
    :param phase0: initial phase (see generateFiringRate)
    :param seed: seed for random number generation
    :return: inhP: python list which represent an inhomogenos Poisson process
    '''

    lambdaE = 20.0
    mu = 1.0 / lambdaE
    tMax = 500.0  # [s]

    homP = []  # homogeneous Poisson process
    np.random.seed(seed)
    exprnd = -mu * np.log(np.random.rand(1))[0]  # MATLAB's random exponential number
    homP.append(exprnd)
    i = 0
    while homP[i] < tMax:
        np.random.seed(seed + i + 1)
        exprnd = -mu * np.log(np.random.rand(1))[0]  # # MATLAB's random exponential number
        tmp = homP[-1] + exprnd
        homP.append(tmp)
        i += 1

    del homP[-1]  # delete the last element which is higher than tMax

    inhP = []  # inhomogeneous Poisson process
    for i, t in enumerate(homP):
        np.random.seed(seed + i + 1)
        if generateFiringRate(nPop, pop, phase0, t, **kwargs) / lambdaE >= np.random.rand(1):
            inhP.append(t)

    return inhP
    
def refractoriness(spikeTrains, ref_per=5e-3):  # added only in 05.2017
    '''
    Delete spikes which are too close to each other
    :param spikeTrains: list of (4000) lists representing individual spike trains
    :param ref_per: refractory period (in sec)
    :return spikeTrains: same structure, but with some spikes deleted
    '''
    
    spikeTrains_updated = []
    count = 0
    for spikes in spikeTrains:  # iterates over single spike trains (from indiv. neurons)
        tmp = np.diff(spikes)  # calculate ISIs
        idx = np.where(tmp < ref_per)[0] + 1 # 2ms refractoriness
        if idx.size:
            count += idx.size
            spikes_updated = np.delete(spikes, idx).tolist()  # delete spikes which are too close
        else:
            spikes_updated = spikes
        spikeTrains_updated.append(spikes_updated)
    
    print "%i spikes deleted becuse of too short refractory period"%count
      
    return spikeTrains_updated
     
