#!/usr/bin/python
# -*- coding: utf8 -*-
"""
helper functions used for weight matrix modifications
author: András Ecker last update: 06.2017
"""

import numpy as np
np.random.seed(12345)


def shuffle(wmxO):
    """
    Randomly shuffles the weight matrix (keeps weight distribution, but no spatial pattern)
    :param wmxO: original weight matrix (4000 * 4000 ndarray)
    :return: wmxM: modified weight matrix (4000 * 4000 ndarray)
    """
    
    wmxM = wmxO  # stupid numpy...
    np.random.shuffle(wmxM)  # shuffle's only rows (keeps output weights)
    np.random.shuffle(wmxM.T)  # transpose and shuffle rows -> shuffle columns

    return wmxM


def shuffle_blocks(wmxO, popSize=50):
    '''
    shuffles popSize*popSize blocks within the martrix
    :param wmxO: original weight matrix (4000 * 4000 ndarray)
    :param popSize: size of the blocks
    :return: wmxM: modified weight matrix (4000 * 4000 ndarray)
    '''

    nPop = 4000 / popSize
    wmxM = np.zeros((4000, 4000))
    d = {}
    # get subpops
    for i in range(nPop):
        for j in range(nPop):
            tmp = wmxO[i*popSize:(i+1)*popSize, j*popSize:(j+1)*popSize]
            d[i, j] = tmp
    # shuffle idx
    x = np.linspace(0, nPop-1, nPop)
    y = np.linspace(0, nPop-1, nPop)
    np.random.shuffle(x)
    np.random.shuffle(y)
    # recreate matrix
    for i, vali in enumerate(x):
        for j, valj in enumerate(y):
            wmxM[i*popSize:(i+1)*popSize, j*popSize:(j+1)*popSize] = d[vali, valj]

    return wmxM


def shuffle_block_rows_cols(wmxO, popSize=50):
    '''
    Perturbes the rows and columns in small blocks of the weight matrix
    :param wmxO: original weight matrix (4000 * 4000 ndarray)
    :param popSize: size of the block shuffled inside
    :return: wmxM: modified weight matrix (4000 * 4000 ndarray)
    '''

    wmxM = np.zeros((4000, 4000))
    nPop = 4000 / popSize  
    # get subpops
    for i in range(nPop):
        for j in range(nPop):
            tmp = wmxO[i*popSize:(i+1)*popSize, j*popSize:(j+1)*popSize]
            # shuffle rows
            rowPert = np.linspace(0, popSize-1, popSize)
            np.random.shuffle(rowPert)
            tmp = tmp[rowPert.tolist()]
            # shuffle cols
            tmp = np.transpose(tmp)
            colPert = np.linspace(0, popSize-1, popSize)
            np.random.shuffle(colPert)
            tmp = tmp[colPert.tolist()]
            tmp = np.transpose(tmp)
            # put back to matrix
            wmxM[i*popSize:(i+1)*popSize, j*popSize:(j+1)*popSize] = tmp

    return wmxM


def shuffle_subpop_input_weights(wmxO, shuffle_size=200):  # added only in 06.2017
    """
    shuffles the input weight (within one column) of a small pop. at the end of the matrix (4000-shuffle_size)
    :param wmxO: original weight matrix (4000 * 4000 ndarray)
    :param shuffle_size: size of the subpop. to modify
    :return: wmxM: modified weight matrix (4000 * 4000 ndarray)
    """
    
    # split matrix
    id_ = 4000-shuffle_size
    subwmxO_keep = wmxO[:, 0:id_]
    subwmxO_mod = wmxO[:, id_:]
    
    # shuffle subpop
    shuffled = subwmxO_mod[:, 0]  # stupid numpy...
    np.random.shuffle(shuffled)
    for i in range(1, shuffle_size):  # iterates over colums
        tmp = subwmxO_mod[:, i]  # stupid numpy...
        np.random.shuffle(tmp)
        shuffled = np.vstack([shuffled, tmp])  # stacks as rows and will be transposed later  
               
    # connect to non-shuffled part
    shuffled = np.transpose(shuffled)  # get back the original dimensions
    wmxM = np.hstack([subwmxO_keep, shuffled])
    
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
        max_ = np.max(row)
        maxj = np.argmax(row)
        mu = np.mean(row)
        wmxM[i, :] = mu
        wmxM[i, maxj] = max_

    return wmxM


def avg_x_weak_weights(wmxO, x=3975):
    '''
    Holds only the 4000-x strongest weight in every row and average the other ones
    :param wmxO: original weight matrix (4000 * 4000 ndarray)
    :return: wmxM: modified weight matrix (4000 * 4000 ndarray)
    '''

    nHeld = 4000 - x
    wmxM = np.zeros((4000, 4000))

    for i in range(0, 4000):
        row = wmxO[i, :]

        tmp = np.partition(-row, nHeld)
        max_ = -tmp[:nHeld]  # values of first 4000-x elements
        rest = -tmp[nHeld:]
        mu = np.mean(rest)  # mean of the x weights
        tmp = np.argpartition(-row, nHeld)
        maxj = tmp[:nHeld]  # indexes of first 4000-x elements

        rowM = mu * np.ones((1, 4000))
        for j, val in zip(maxj, max_):
           rowM[0, j] = val

        wmxM[i, :] = rowM

    return wmxM


def binary_weights(wmxO, x=1):
    '''
    Makes the matrix binary by averaging the first x and the other 100-x percent of the matrix
    :param wmxO: original weight matrix (4000 * 4000 ndarray)
    :param x: percent of strongest weights to keep
    :return: wmxM: modified weight matrix (4000 * 4000 ndarray)
    '''

    nHeld = int((4000**2) * (x / 100.0))
    wmxM = np.zeros((4000, 4000))

    # sort values, get min and max weights
    tmp = wmxO.tolist()
    wmxL = [val for sublist in tmp for val in sublist]
    wmxL.sort()
    wmxL = wmxL[::-1]
    max_ = np.mean(wmxL[:nHeld])
    min_ = np.mean(wmxL[nHeld:])

    # recreate matrix
    nHeld = int(4000 * (x / 100.0))   
    for i in range(0, 4000):
        row = wmxO[i, :]
        tmp = np.argpartition(-row, nHeld)
        maxj = tmp[:nHeld]
        rowM = min_ * np.ones((1, 4000))
        for j in maxj:
           rowM[0, j] = max_
        wmxM[i, :] = rowM

    return wmxM

