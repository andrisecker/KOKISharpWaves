#!/usr/bin/python
# -*- coding: utf8 -*-
'''
BluePyOpt evaluator for optimization
authors: Bence Bagi, András Ecker last update: 06.2017
'''

import os
import sys
import numpy as np
import bluepyopt as bpop
import run_sim as sim
SWBasePath = os.path.sep.join(os.path.abspath('__file__').split(os.path.sep)[:-2])
# add the 'scripts' directory to the path (import the modules)
sys.path.insert(0, os.path.sep.join([SWBasePath, 'scripts']))
from detect_oscillations import *


class Brian2Evaluator(bpop.evaluators.Evaluator):

    def __init__(self, Wee, params):
        """
        :param params: list of parameters to fit - every entry must be a tuple: (name, lower bound, upper bound). 
        """
        super(Brian2Evaluator, self).__init__()
        # passing Wee with cPickle to the slaves (as BluPyOpt does) is still the fastest solution (that's why it's in self)
        self.Wee = Wee
        self.params = params
        # Parameters to be optimized
        self.params = [bpop.parameters.Parameter(name, bounds=(minval, maxval))
                       for name, minval, maxval in self.params]
        self.objectives = ["fitness_score"]  # random name for BluePyOpt
                       
    def generate_model(self, individual):
        """runs simulation (run_sim.py) and returns monitors"""
        sme, smi, popre, popri = sim.run_simulation(self.Wee, *individual)
        return sme, smi, popre, popri
        
    def evaluate_with_lists(self, individual):
        """fitness error used by BluePyOpt for the optimization"""
        sme, smi, popre, popri = self.generate_model(individual)
        fitness = 0
        if sme.num_spikes > 0 and smi.num_spikes > 0:  # check if there is any activity

            spikeTimesE, spikingNeuronsE, poprE, ISIhist, bin_edges = preprocess_monitors(sme, popre)
            spikeTimesI, spikingNeuronsI, poprI = preprocess_monitors(smi, popri, calc_ISI=False)
            # call replay detection functions:
            avgReplayInterval = replay(ISIhist[3:16])  # bins from 150 to 850 (range of interest)
            
            if not np.isnan(avgReplayInterval):  # evaluate only if there's sequence replay!
            
                # call detect_oscillation functions:
                meanEr, rEAC, maxEAC, tMaxEAC, maxEACR, tMaxEACR, fE, PxxE, avgRippleFE, ripplePE = ripple(poprE, 1000)
                avgGammaFE, gammaPE = gamma(fE, PxxE)
                meanIr, rIAC, maxIAC, tMaxIAC, maxIACR, tMaxIACR, fI, PxxI, avgRippleFI, ripplePI = ripple(poprI, 1000)
                avgGammaFI, gammaPI = gamma(fI, PxxI)
            
                # look for significant ripple peak close to 180 Hz
                ripple_peakE = 0
                if not np.isnan(avgRippleFE):
                    ripple_peakE = 5 - (np.abs(avgRippleFE - 180) / 180)
                ripple_peakI = 0
                if not np.isnan(avgRippleFI):
                    ripple_peakI = 7 - (np.abs(avgRippleFI - 180) / 180)
                # look for the absence of significant gamma peak
                bool_gammaE = int(np.isnan(avgGammaFE))
                bool_gammaI = int(np.isnan(avgGammaFI))
                # look for "low" population rate (gauss around 3Hz for exc. pop.)
                rateE = np.exp(-1/2*(meanEr-3)**2/1.5**2)  # peak normalized to 1
                
                fitness = -1 * (ripple_peakE + ripple_peakI + ripplePE/gammaPE + ripplePI/gammaPI + 3*bool_gammaE + 2*bool_gammaI + 4*rateE)
            else:
                fitness = 0
        
        return [fitness]  # single score but has to be a list for BluePyOpt
        
 
