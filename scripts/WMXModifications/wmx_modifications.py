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
    Replace to weights of neurons 3500 - 4000, with random numbers
    (gaussian distribution, same mean and deviation as the original 15/64 of the matrix)
    :param wmxO: original weight matrix (4000 * 4000 ndarray)
    :return: wmxM: modified weight matrix (4000 * 4000 ndarray)
    '''
    x1, x2, x3, x4, x5, x6, x7, wmxDown = np.vsplit(wmxO, 8)
    wmx_tmp = np.concatenate((x1, x2, x3, x4, x5, x6, x7), axis=0)
    y1, y2, y3, y4, y5, y6, y7, wmxRight = np.hsplit(wmx_tmp, 8)

    muR = np.mean(wmxRight)
    sigmaR = np.std(wmxRight)
    muD = np.mean(wmxDown)
    sigmaD = np.std(wmxDown)
    mu = (muR + muD) / 2.0
    print 'mu:', mu
    sigma = (sigmaR + sigmaD) / 2.0
    print 'sigma:', sigma

    np.random.seed(0)
    wmxRGauss = np.random.normal(mu, sigma, (3500, 500))
    np.random.seed(1)
    wmxDGauss = np.random.normal(mu, sigma, (500, 4000))

    wmx_tmp = np.concatenate((y1, y2, y3, y4, y5, y6, y7), axis=1)
    wmxUp = np.concatenate((wmx_tmp, wmxRGauss), axis=1)
    wmxM = np.concatenate((wmxUp, wmxDGauss), axis=0)

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


def shuffle_rows_cols(wmxO):
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

    return wmxM


def shuffle_block_rows_cols(wmxO):
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


def avg_weak_weights(wmxO):
    '''
    Holds only the strongest weight in every row and average the other ones
    :param wmxO: original weight matrix (4000 * 4000 ndarray)
    :return: wmxM: modified weight matrix (4000 * 4000 ndarray)
    '''

    wmxM = np.zeros((4000, 4000))

    for i in range(0, 4000):
        row = wmxO[i, :]
        max = np.max(row)
        maxj = np.argmax(row)
        mu = np.mean(row)
        wmxM[i, :] = mu
        wmxM[i, maxj] = max

    return wmxM


def avg_x_weak_weights(wmxO, x):
    '''
    Holds only the 4000-x strongest weight in every row and average the other ones
    :param wmxO: original weight matrix (4000 * 4000 ndarray)
    :return: wmxM: modified weight matrix (4000 * 4000 ndarray)
    '''

    nHolded = 4000 - x
    wmxM = np.zeros((4000, 4000))

    for i in range(0, 4000):
        row = wmxO[i, :]

        tmp = np.partition(-row, nHolded)
        max = -tmp[:nHolded]  # values of first 4000-x elements
        rest = -tmp[nHolded:]
        mu = np.mean(rest)  # mean of the x weights
        tmp = np.argpartition(-row, nHolded)
        maxj = tmp[:nHolded]  # indexes of first 4000-x elements

        rowM = mu * np.ones((1, 4000))
        for j, val in zip(maxj, max):
           rowM[0, j] = val

        wmxM[i, :] = rowM

    return wmxM


def disconnected(wmxO):
    '''
    splits the matrix and reconnet the upper and the lower part invertedly
    :param wmxO: original weight matrix (4000 * 4000 ndarray)
    :return: wmxM: modified weight matrix (4000 * 4000 ndarray)
    '''

    tmp1, tmp2 = np.vsplit(wmxO, 2)
    wmxM = np.concatenate((tmp2, tmp1), axis=0)

    return wmxM


def binary_weights(wmxO, x):
    '''
    Makes the matrix binary by averaging the first x and the other 100-x percent of the matrix
    :param wmxO: original weight matrix (4000 * 4000 ndarray)
    :return: wmxM: modified weight matrix (4000 * 4000 ndarray)
    '''

    nHolded = int((4000**2) * (x / 100.0))
    wmxM = np.zeros((4000, 4000))

    tmp = wmxO.tolist()
    wmxL = [val for sublist in tmp for val in sublist]
    wmxL.sort()
    wmxL = wmxL[::-1]
    max = np.mean(wmxL[:nHolded])
    min = np.mean(wmxL[nHolded:])

    print 'max:', max
    print 'min:', min

    nHolded = int(4000 * (x / 100.0))

    for i in range(0, 4000):
        row = wmxO[i, :]

        tmp = np.argpartition(-row, nHolded)
        maxj = tmp[:nHolded]

        rowM = min * np.ones((1, 4000))
        for j in maxj:
           rowM[0, j] = max

        wmxM[i, :] = rowM

    return wmxM


def shuffle_blocks(wmxO, popSize):
    '''
    shuffles popSize*popSize blocks within the martrix
    :param wmxO: original weight matrix (4000 * 4000 ndarray)
    :param popSize: size of the blocks
    :return: wmxM: modified weight matrix (4000 * 4000 ndarray)
    '''

    nPop = 4000 / popSize

    d = {}
    wmxM = np.zeros((4000, 4000))

    for i in range(nPop):
        for j in range(nPop):
            tmp = wmxO[i*popSize:(i+1)*popSize, j*popSize:(j+1)*popSize]
            d[i, j] = tmp

    x = np.linspace(0, nPop-1, nPop)
    y = np.linspace(0, nPop-1, nPop)

    np.random.seed(0)
    np.random.shuffle(x)
    np.random.seed(1)
    np.random.shuffle(y)

    for i, vali in enumerate(x):
        for j, valj in enumerate(y):
            wmxM[i*popSize:(i+1)*popSize, j*popSize:(j+1)*popSize] = d[vali, valj]

    return wmxM

def mirror(wmxO):

    wmxM = np.fliplr(wmxO)

    return wmxM