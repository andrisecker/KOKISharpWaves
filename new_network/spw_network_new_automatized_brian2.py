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


def run_simulation_analyse_results(Wee, multiplier, X, STDP_mode="asym", detailed=True, TFR=False, verbose=True):
    """
    Runs simulation for a given Wee multiplier and analyses, plots and stores the results
    :param Wee, STDP_mode: see `run_simulation` in `spw_network_new_brian2.py`
    :param multiplier: weight matrix multiplier
    :param X: 20*len(multipliers) np.array for storing the results
    :param detailed, verbose: see `spw_network_new_brian2/run_sim()`
    :param TFR: bool - to analyse and plot time frequency representation (by wavelet analysis)
    """

    print "multiplier: %.2f"%multiplier
    Wee_tmp = Wee * multiplier  # Wee matrix loaded before the for loop

    # run simulation (neuronal parameters are in `spw_network_new_brian2.py`)
    if detailed:   
        sme, smi, popre, popri, selection, mSME, sMI = run_simulation(Wee, STDP_mode, detailed=True, verbose=verbose)
    else:
        sme, smi, popre, popri = run_simulation(Wee, STDP_mode, detailed=False, verbose=verbose)

    if sme.num_spikes > 0 and smi.num_spikes > 0:  # check if there is any activity

        # analyse spikes
        spikeTimesE, spikingNeuronsE, poprE, ISIhist, bin_edges = preprocess_monitors(sme, popre)
        spikeTimesI, spikingNeuronsI, poprI = preprocess_monitors(smi, popri, calc_ISI=False)
        # detect replay
        avgReplayInterval = replay(ISIhist[3:16])  # bins from 150 to 850 (range of interest)

        # analyse rates
        if TFR:
            meanEr, rEAC, maxEAC, tMaxEAC, fE, PxxE, trfE, tE, freqsE = analyse_rate(poprE, TFR=True)
            meanIr, rIAC, maxIAC, tMaxIAC, fI, PxxI, trfI, tI, freqsI = analyse_rate(poprI, TFR=True)
        else:
            meanEr, rEAC, maxEAC, tMaxEAC, fE, PxxE = analyse_rate(poprE)
            meanIr, rIAC, maxIAC, tMaxIAC, fI, PxxI = analyse_rate(poprI)
        maxEACR, tMaxEACR, avgRippleFE, ripplePE = ripple(rEAC, fE, PxxE)
        maxIACR, tMaxIACR, avgRippleFI, ripplePI = ripple(rIAC, fI, PxxI)
        avgGammaFE, gammaPE = gamma(fE, PxxE)       
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
        plot_raster_ISI(spikeTimesE, spikingNeuronsE, poprE, [ISIhist, bin_edges], "blue", multiplier_=1)
        if TFR:
            plot_PSD(poprE, rEAC, fE, PxxE, "Pyr_population", "blue", multiplier_=1,
                     TFR=True, tfr=trfE, t=tE, freqs=freqsE, fs=1000)
            plot_PSD(poprI, rIAC, fI, PxxI, "Bas_population", "green", multiplier_=1,
                     TFR=True, tfr=trfI, t=tI, freqs=freqsI, fs=1000)
        else:
            plot_PSD(poprE, rEAC, fE, PxxE, "Pyr_population", "blue", multiplier_=1)
            plot_PSD(poprI, rIAC, fI, PxxI, "Bas_population", "green", multiplier_=1)
   
        if detailed:
            subset = plot_zoomed(spikeTimesE, spikingNeuronsE, poprE, "Pyr_population", "blue", multiplier_=1,
                                 sm=mSME, selection=selection)
            plot_zoomed(spikeTimesI, spikingNeuronsI, poprI, "Bas_population", "green", multiplier_=1,
                        Pyr_pop=False, sm=sMI)
            plot_detailed(mSME, subset, multiplier_=1, new_network=True)
        else:
            plot_zoomed(spikeTimesE, spikingNeuronsE, poprE, "Pyr_population", "blue", multiplier_=1)
            plot_zoomed(spikeTimesI, spikingNeuronsI, poprI, "Bas_population", "green", multiplier_=1, Pyr_pop=False)
        
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
    
    detailed = True; TFR = True
    
    fIn = "wmxR_%s_shuf_subpop_inp.txt"%STDP_mode    
    fName = os.path.join(SWBasePath, "files", fIn)
    Wee = load_Wee(fName)
    
    fOut = "%s_shuf_subpop_inp_v3.txt"%STDP_mode
    
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
            
        run_simulation_analyse_results(Wee, multiplier, X, STDP_mode, detailed=detailed, TFR=TFR)
    
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
    
 
