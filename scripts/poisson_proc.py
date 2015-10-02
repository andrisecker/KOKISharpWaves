#!/usr/bin/python
# -*- coding: utf8 -*-

import numpy as np

def generateFiringRate(t, pop, phase0, theta, avgRateInField, vMice, lPlaceField,lRoute):

    r = lRoute / (2 * np.pi)  #radius [cm]
    tRoute = lRoute / vMice  # [s]
    wMice=2 * np.pi / tRoute  # angular velocity

    phiPFRad = lPlaceField / r  # angle of place field [rad]
    # phiPF = 180 * (phiPFRad / np.pi)  # angle of place field [deg]

    nPop = 50

    phiStart = pop * 2 * np.pi * 1/nPop
    phiEnd = np.mod(phiStart + phiPFRad, 2 * np.pi)  # ???

    x = np.mod(wMice * t, 2 * np.pi)  # position of the mice [rad]
    y = phase0 + 2 * np.pi * theta * t  # ??? 't=0 -nál 0 foktól indul'

    shift = phiStart + phiPFRad / 2

    # phase precession
    m = - (x - phiStart) * 2 * np.pi / phiPFRad  # prefered phase: f(actuall position within the place field)

    sig = 0.5  # deviation of phase-locking (von Misses distribution)
    s = 1.0 / sig

    if phiStart < phiEnd:
        if phiStart <= x and x < phiEnd:

            lambda1 = np.cos((2 * np.pi) / (2 * phiPFRad) * (x - shift)) * avgRateInField
            lambda2 = np.exp(s * np.cos(y - m)) / np.exp(s)

        else:

            lambda1 = 0
            lambda2 = 1

    else:
        if phiStart <= x or x < phiEnd:

            lambda1 = np.cos((2 * np.pi) / (2 * phiPFRad) * (x - shift)) * avgRateInField
            lambda2 = np.exp(s * np.cos(y - m)) / np.exp(s)

        else:

            lambda1 = 0
            lambda2 = 1

    lambdaP = lambda1 * lambda2

    return lambdaP


def inhomPoisson(phase0, pop, theta, avgRateInField, vMice, lPlaceField,lRoute):

    lambdaE = 20.0
    tMax = 500.0  # [ms]

    homP = []  # homogeneous Poisson process
    homP.append(np.random.exponential(1.0 / lambdaE))
    i = 0
    while homP[i] < tMax:
        tmp = homP[-1] + np.random.exponential(1.0 / lambdaE)
        homP.append(tmp)
        i += 1

    del homP[-1]

    inhP = []  # inhomogeneous Poisson process
    for spike in homP:
        tmp = generateFiringRate(spike, pop, phase0, theta, avgRateInField, vMice, lPlaceField,lRoute)
        if  tmp / lambdaE >= np.random.rand(1):
            inhP.append(spike)

    return inhP
