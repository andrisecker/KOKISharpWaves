#!/usr/bin/python
# -*- coding: utf8 -*-
"""
analyse the "effect" of an STDP rule (with the given cell model and synapse parameters)
protocol based on Mishra et al. 2016 - 10.1038/ncomms11552
author: András Ecker last update: 11.2017
"""

import os
#import sys
from brian2 import *
import numpy as np
import random as pyrandom
#import matplotlib.pyplot as plt
from analyse_EPS import sym_paired_recording
SWBasePath = os.path.sep.join(os.path.abspath('__file__').split(os.path.sep)[:-2])
sys.path.insert(0, os.path.sep.join([SWBasePath, 'scripts']))
from plots import plot_STDP_rule


def sim_pairing_prot(delta_ts, taup, taum, Ap, Am, wmax, w_init):  # STDP_synapse
    """
    Aims to mimic spike pairing induced LTP protocol from Mishra et al. 2016 (300 repetition with different $delta$t-s at 1Hz)
    (Simulated for all different $delta$t-s in the same time, since it's way more effective than one-by-one)
    exponential STDP: f(s) = A_p * exp(-s/tau_p) (if s > 0), where s=tpost_{spike}-tpre_{spike}
    :param delta_ts: list of $delta$t intervals between pre and post spikes (in ms)
    :param taup, taum: time constant of weight change
    :param Ap, Am: max amplitude of weight change
    :param wmax: maximum weight
    :param w_init: initial weights (in S)
    :return: np.array with learned weights (same order as delta_ts)
    """
    
    np.random.seed(12345)
    pyrandom.seed(12345)
          
    pre_spikeTrain = np.arange(5, 305)
    # create 2 numpy arrays for Brian2's SpikeGeneratorGroup
    spikeTimes = pre_spikeTrain
    spikingNrns = 0 * np.ones_like(pre_spikeTrain)
    for i in range(0, len(delta_ts)):     
        post_spikeTrain = pre_spikeTrain + delta_ts[i]/1000.  # /1000 ms conversion
        spikingNrns = np.concatenate((spikingNrns, (i+1)*np.ones_like(post_spikeTrain)), axis=0)        
        spikeTimes = np.concatenate((spikeTimes, post_spikeTrain), axis=0)

    PC = SpikeGeneratorGroup(1+len(delta_ts), spikingNrns, spikeTimes*second)

    # mimics Brian1's exponentialSTPD class, with interactions='all', update='additive'
    # see more on conversion: http://brian2.readthedocs.io/en/stable/introduction/brian1_to_2/synapses.html
    STDP = Synapses(PC, PC,
             """
             w : 1
             dA_pre/dt = -A_pre/taup : 1 (event-driven)
             dA_post/dt = -A_post/taum : 1 (event-driven)
             """,
             on_pre="""
             A_pre += Ap
             w = clip(w + A_post, 0, wmax)
             """,
             on_post="""
             A_post += Am
             w = clip(w + A_pre, 0, wmax)
             """)
             
    STDP.connect(i=0, j=np.arange(1, len(delta_ts)+1).tolist())  # connect pre, to every post
    STDP.w = w_init

    # run simulation
    sm = SpikeMonitor(PC, record=True)
    run(310*second, report='text')
    
    return STDP.w[:]


if __name__ == "__main__":
    
    delta_ts = [-100., -50., -10., 10., 50., 100.]  # ms
    
    # STDP parameters
    taup = taum = 20 * ms
    Ap = Am = 0.01  # 0.0035
    wmax = 8e-9  # S (w is dimensionless in the equations)
    Ap *= wmax  # needed to reproduce Brian1 results
    Am *= wmax  # needed to reproduce Brian1 results
    w_init = 1e-9  # S (w is dimensionless in the equations)
    
    mode_ = plot_STDP_rule(taup/ms, taum/ms, Ap/1e-9, Am/1e-9, "STDP_rule")
    
    i_hold = -43.638  # pA (calculated by clamp_cell.py)
    
    # baseline EPSP
    t_, EPSP, _ = sym_paired_recording(w_init, i_hold)
    dEPSPs = {"t":t_, "baseline":EPSP}
    
    # apply spike pairing LTP protocol     
    weights = sim_pairing_prot(delta_ts, taup, taum, Ap, Am, wmax, w_init)
    
    # EPSPs after learning
    for delta_t, weight in zip(delta_ts, weights):
        _, EPSP, _ = sym_paired_recording(weight, i_hold)
        dEPSPs[delta_t] = EPSP


    
