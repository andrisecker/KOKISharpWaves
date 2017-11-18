#!/usr/bin/python
# -*- coding: utf8 -*-
"""
looped version of spw_network_new_brian2.py -> checks the dynamics for different multipliers of the learned weight matrix
authors: Bence Bagi, András Ecker last update: 11.2017
"""

import os
import sys
#from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore") # ignore scipy 0.18 sparse matrix warning...
from spw_network_new_brian2 import run_simulation
SWBasePath = os.path.sep.join(os.path.abspath('__file__').split(os.path.sep)[:-2])
sys.path.insert(0, os.path.sep.join([SWBasePath, 'scripts']))
from detect_oscillations import *
from plots import *


def run_simulation_analyse_results(Wee, multiplier, X, STDP_mode="asym", detailed=True, verbose=True):
    """
    Runs simulation for a given Wee multiplier and analyses, plots and stores the results
    :param Wee, STDP_mode, detailed, verbose: see `run_simulation` in `spw_network_new_brian2.py`
    :param multiplier: weight matrix multiplier
    :param X: 20*len(multipliers) np.array for storing the results
    """

    print "multiplier: %.2f"%multiplier
    Wee_tmp = Wee * multiplier  # Wee matrix loaded before the for loop

    sme, smi, popre, popri, selection, mSME, sMI = run_simulation(Wee_tmp, STDP_mode, detailed, verbose)  # neuronal parameters are in `spw_network_new_brian2.py`

    if sme.num_spikes > 0 and smi.num_spikes > 0:  # check if there is any activity

        spikeTimesE, spikingNeuronsE, poprE, ISIhist, bin_edges = preprocess_monitors(sme, popre)
        spikeTimesI, spikingNeuronsI, poprI = preprocess_monitors(smi, popri, calc_ISI=False)

        # calling detect_oscillation functions:
        avgReplayInterval = replay(ISIhist[3:16])  # bins from 150 to 850 (range of interest)

        meanEr, rEAC, maxEAC, tMaxEAC, maxEACR, tMaxEACR, fE, PxxE, avgRippleFE, ripplePE = ripple(poprE, 1000)
        avgGammaFE, gammaPE = gamma(fE, PxxE)
        meanIr, rIAC, maxIAC, tMaxIAC, maxIACR, tMaxIACR, fI, PxxI, avgRippleFI, ripplePI = ripple(poprI, 1000)
        avgGammaFI, gammaPI = gamma(fI, PxxI)

        print "Avg. exc. ripple freq:%s, Avg. inh. ripple freq:%s"%(avgRippleFE, avgRippleFI)
        print "--------------------------------------------------"
        
        # store results for this multiplier
        X[:, k] = [multiplier,
                   meanEr, maxEAC, tMaxEAC, maxEACR, tMaxEACR,
                   meanIr, maxIAC, tMaxIAC, maxIACR, tMaxIACR,
                   avgReplayInterval,
                   avgRippleFE, ripplePE, avgGammaFE, gammaPE,
                   avgRippleFI, ripplePI, avgGammaFI, gammaPI]

        # Plots
        plot_raster_ISI(spikeTimesE, spikingNeuronsE, poprE, [ISIhist, bin_edges], "blue", multiplier)
        plot_PSD(poprE, rEAC, fE, PxxE, "Pyr_population", "b-", multiplier)
        plot_PSD(poprI, rIAC, fI, PxxI, "Bas_population", "g-", multiplier)

        subset = plot_zoomed(spikeTimesE, spikingNeuronsE, poprE, "Pyr_population", "blue", multiplier, sm=mSME, selection=selection)
        plot_zoomed(spikeTimesI, spikingNeuronsI, poprI, "Bas_population", "green", multiplier, Pyr_pop=False, sm=sMI)        
        plot_detailed(mSME, subset, multiplier, new_network=True)
        
        plt.close("all")

    else:  # if there is no activity the auto-correlation function will throw an error!

        print "No activity !"
        print "--------------------------------------------------"

        # store results for this multiplier
        X[:, k] = [multiplier, 0, np.nan, np.nan, np.nan, np.nan, 0, np.nan, np.nan, np.nan, np.nan,
		           np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]        


if __name__ == "__main__":  

    try:
        STDP_mode = sys.argv[1]       
    except:
        STDP_mode = "asym"
    assert(STDP_mode in ["sym", "asym"])
    fIn = "wmxR_%s_binary.txt"%STDP_mode    
    fName = os.path.join(SWBasePath, "files", fIn)
    Wee = load_Wee(fName)
    
    fOut = "%s_binary_v2.txt"%STDP_mode
    
    # range of Wee multipliers tested:
    first = 0.6
    last = 1.5
    dataPoints = 10
    multipliers = np.linspace(first, last, dataPoints)
    
    # plot matrix (and some selected weights)
    plot_wmx_avg(Wee, 100, "wmx")
    dWee = save_selected_w(Wee, selection=np.array([500, 1500, 2500, 3500]))
    plot_weights(dWee, "sel_weights")
    
    # run simulations for different multipliers
    X = np.zeros((20, dataPoints))
    for k, multiplier in enumerate(multipliers):
            
        run_simulation_analyse_results(Wee, multiplier, X, STDP_mode)
    
    # summary plots   
    plot_summary_replay(multipliers, replay_interval=X[11, :], rateE=X[1, :], rateI=X[6, :])
    plot_summary_AC(multipliers, maxACE=X[2, :], maxACI=X[7, :], maxRACE=X[4, :], maxRACI=X[9, :])
    plot_summary_ripple(multipliers, rippleFE=X[12, :], rippleFI=X[16, :], ripplePE=X[13, :], ripplePI=X[17, :])
    plot_summary_gamma(multipliers, gammaFE=X[14, :], gammaFI=X[18, :], gammaPE=X[15, :], gammaPI=X[19, :])
    
    # save result array (X)
    fName= os.path.join(SWBasePath, "files", fOut)
    header = 'Multiplier, Mean_exc.rate, Max.exc.AC., at[ms], Max.exc.AC.in_ripple_range, at[ms],' \
         'Mean_inh.rate, Max.inh.AC., at[ms], Max.inh.AC.in_ripple_range, at[ms],' \
         'avg. replay interval,' \
         'avgRippleFE, ripplePE, avgGammaFE, ripplePE,' \
         'avgRippleFI, ripplePI, avgGammaFI, ripplePI'
    np.savetxt(fName, X, fmt='%.6f', delimiter='\t', header=header)
    
 
