#!/usr/bin/python
# -*- coding: utf8 -*-

import numpy as np


def gauss(wmxO):
    '''
    Replace the whole weight matrix with random numbers (gaussian distribution, same mean and deviation)
    :param wmxO: original weight matrix (4000 * 4000 ndarray)
    :return: wmxM: modified weight matrix (4000 * 4000 ndarray)
    '''

    mu = np.mean(wmxO)
    print 'mu:', mu
    sigma = np.std(wmxO)
    print 'sigma:', sigma

    np.random.seed(0)
    wmxM = np.random.normal(mu, sigma, (4000, 4000))

    return wmxM


def gauss_rectangle(wmxO):
    '''
    Replace an 1000*1000 rectangle with random numbers
    (gaussian distribution, same mean and deviation as the original 1/8)
    :param wmxO: original weight matrix (4000 * 4000 ndarray)
    :return: wmxM: modified weight matrix (4000 * 4000 ndarray)
    '''
    x1, x2, x3, tmp = np.vsplit(wmxO, 4)
    tmp1, tmp2, tmp3, tmp4 = np.hsplit(tmp, 4)

    mu = np.mean(tmp4)
    print 'mu:', mu
    sigma = np.std(tmp4)
    print 'sigma:', sigma

    np.random.seed(0)
    wmxGauss = np.random.normal(mu, sigma, (1000, 1000))

    wmxU = np.concatenate((x1, x2, x3), axis=0)
    wmxD = np.concatenate((tmp1, tmp2, tmp3, wmxGauss), axis=1)
    wmxM = np.concatenate((wmxU, wmxD), axis=0)

    return wmxM

def mean_rectangle(wmxO):
    '''
    Replace an 1000*1000 rectangle with the mean of the distribution
    :param wmxO: original weight matrix (4000 * 4000 ndarray)
    :return: wmxM: modified weight matrix (4000 * 4000 ndarray)
    '''

    x1, x2, x3, tmp = np.vsplit(wmxO, 4)
    tmp1, tmp2, tmp3, tmp4 = np.hsplit(tmp, 4)

    mu = np.mean(tmp4)
    print 'mu:', mu

    np.random.seed(0)
    wmxMean = mu * np.ones((1000, 1000))

    wmxU = np.concatenate((x1, x2, x3), axis=0)
    wmxD = np.concatenate((tmp1, tmp2, tmp3, wmxMean), axis=1)
    wmxM = np.concatenate((wmxU, wmxD), axis=0)

    return wmxM


def mix_rows_cols(wmxO):
    '''
    Perturbes the rows and columns of the weight matrix
    :param wmxO: original weight matrix (4000 * 4000 ndarray)
    :return: wmxM: modified weight matrix (4000 * 4000 ndarray)
    '''
    rowPert = np.linspace(0, 3999, 4000)
    np.random.seed(0)
    np.random.shuffle(rowPert)
    tmp = wmxO[rowPert.tolist()]

    tmp = np.transpose(tmp)
    colPert = np.linspace(0, 3999, 4000)
    np.random.seed(1)
    np.random.shuffle(colPert)
    tmp = tmp[colPert.tolist()]

    wmxM = np.transpose(tmp)
    print wmxM.shape

    return wmxM


def mix_block_rows_cols(wmxO):
    '''
    Perturbes the rows and columns in small blocks of the weight matrix
    :param wmxO: original weight matrix (4000 * 4000 ndarray)
    :return: wmxM: modified weight matrix (4000 * 4000 ndarray)
    '''

    popSize = 50
    nPop = 4000 / popSize

    wmxM = np.zeros((4000, 4000))

    for i in range(nPop):
        for j in range(nPop):
            tmp = wmxO[i*popSize:(i+1)*popSize, j*popSize:(j+1)*popSize]

            rowPert = np.linspace(0, popSize-1, popSize)
            np.random.seed(i)
            np.random.shuffle(rowPert)

            tmp = tmp[rowPert.tolist()]

            tmp = np.transpose(tmp)
            colPert = np.linspace(0, popSize-1, popSize)
            np.random.seed(j)
            np.random.shuffle(colPert)
            tmp = tmp[colPert.tolist()]
            tmp = np.transpose(tmp)

            wmxM[i*popSize:(i+1)*popSize, j*popSize:(j+1)*popSize] = tmp

    return wmxM




