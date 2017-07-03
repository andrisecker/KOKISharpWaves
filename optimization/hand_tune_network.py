#!/usr/bin/python
# -*- coding: utf8 -*-
'''
looped version of spw* scripts to fine tune weights
author: András Ecker last update: 06.2017
'''

import os
import gc
import numpy as np
from brian2 import *
from run_sim import *
import matplotlib.pyplot as plt
SWBasePath = os.path.sep.join(os.path.abspath('__file__').split(os.path.sep)[:-2])
# add the 'scripts' directory to the path (import the modules)
sys.path.insert(0, os.path.sep.join([SWBasePath, 'scripts']))
from detect_oscillations import *
from plots import *


fIn = "wmxR_asym.txt"

fName = os.path.join(SWBasePath, "files", fIn)
Wee = load_Wee(fName)
    
J_BasExc = 5.
J_BasInh = 0.45
J_PyrInhs = [0.0175, 0.02, 0.025]
Wee_mults = [2.8, 2.9, 3.]
J_PyrMFs = [24, 25, 26]
rate_MFs = [18, 19, 20]

for mult in Wee_mults:
    for J_PyrMF in J_PyrMFs:
        for rate_MF in rate_MFs:
            for J_PyrInh in J_PyrInhs:
            
                tmp = "%s_%s_%s_%s"%(mult, J_PyrMF, rate_MF, J_PyrInh)
                print tmp
        
                sme, smi, popre, popri  = run_simulation(Wee, J_PyrInh, J_BasExc, J_BasInh, mult, J_PyrMF, rate_MF)
                
                if sme.num_spikes > 0 and smi.num_spikes > 0:  # check if there is any activity

                    spikeTimesE, spikingNeuronsE, poprE, ISIhist, bin_edges = preprocess_monitors(sme, popre)
                    spikeTimesI, spikingNeuronsI, poprI = preprocess_monitors(smi, popri, calc_ISI=False)
                    plot_raster_ISI(spikeTimesE, spikingNeuronsE, [ISIhist, bin_edges], 'blue', multiplier_=tmp)

                    # call detect_oscillation functions:
                    avgReplayInterval = replay(ISIhist[3:16])  # bins from 150 to 850 (range of interest)
                    
                    if not np.isnan(avgReplayInterval):  # evaluate only if there's sequence replay!

                        meanEr, rEAC, maxEAC, tMaxEAC, maxEACR, tMaxEACR, fE, PxxE, avgRippleFE, ripplePE = ripple(poprE, 1000)
                        avgGammaFE, gammaPE = gamma(fE, PxxE)
                        meanIr, rIAC, maxIAC, tMaxIAC, maxIACR, tMaxIACR, fI, PxxI, avgRippleFI, ripplePI = ripple(poprI, 1000)
                        avgGammaFI, gammaPI = gamma(fI, PxxI)
                
                        plot_PSD(poprE, rEAC, fE, PxxE, "Pyr_population", 'b-', multiplier_=tmp)
                        plot_PSD(poprI, rIAC, fI, PxxI, "Bas_population", 'g-', multiplier_=tmp)

                        _, _ = plot_zoomed(spikeTimesE, spikingNeuronsE, poprE, "Pyr_population", "blue", multiplier_=tmp)
                        plot_zoomed(spikeTimesI, spikingNeuronsI, poprI, "Bas_population", "green", multiplier_=tmp, Pyr_pop=False)
                        plt.close("all")
                        
                    print "--------------------------------------------------"

                else:  # if there is no activity the auto-correlation function will throw an error!

                    print "No activity !"
                    print "--------------------------------------------------"
                    
                gc.collect()
        
    
    
