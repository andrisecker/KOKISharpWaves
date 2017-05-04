#!/usr/bin/python
# -*- coding: utf8 -*-
'''
optimize connection parameters (synaptic weights, time constants, delays)
authors: Bagi Bence, András Ecker last update: 05.2017
'''


import os
import sys
import logging
import numpy as np
import sim_evaluator
import bluepyopt as bpop
import multiprocessing as mp
import matplotlib.pyplot as plt
SWBasePath = os.path.sep.join(os.path.abspath('__file__').split(os.path.sep)[:-2])
# add the 'scripts' directory to the path (import the modules)
sys.path.insert(0, os.path.sep.join([SWBasePath, 'scripts']))
from detect_oscillations import *
from plots import *

# print info into console
logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

fIn = "wmxR_asym.txt"
fName = os.path.join(SWBasePath, "files", fIn)
Wee = load_Wee(fName)

# Parameters to be fitted as a list of: (name, lower bound, upper bound)
optconf = [("J_PyrInh_", 0.01, 1.),  # !!! the values are now around the ones Bence found with hand tuning, just to check the algo ... 
           ("J_BasExc_", 5., 35.),
           ("J_BasInh_", 2., 15.),
           ("WeeMult_", 2., 7.)]
           # the order matters! if you want to add more parameters - update run_sim.py too 
pnames = [name for name, _, _ in optconf]

# Create multiprocessing pool for parallel evaluation of fitness function
pool = mp.Pool(processes=mp.cpu_count())

# Create BluePyOpt optimization and run 
evaluator = sim_evaluator.Brian2Evaluator(Wee, optconf)
opt = bpop.optimisations.DEAPOptimisation(evaluator, offspring_size=3, map_function=pool.map,
                                          eta=20, mutpb=0.3, cxpb=0.7)
                                          
pop, hof, log, history = opt.run(max_ngen=3, cp_filename="checkpoints/checkpoint.pkl")

# Get best individual
best = hof[0]
for pname, value in zip(pnames, best):
    print '%s = %.2f' % (pname, value)
print 'Fitness value: ', best.fitness.values

# summary figure (about optimization)
plot_evolution(log.select('gen'), np.array(log.select('min')), np.array(log.select('avg')),
               np.array(log.select('std')), "fittnes_evolution")


# ====================================== end of optimization ======================================

print " ===== Rerun simulation with the best parameters ===== "
sme, smi, popre, popri = evaluator.generate_model(best)

# analyze dynamics
spikeTimesE, spikingNeuronsE, poprE, ISIhist, bin_edges = preprocess_monitors(sme, popre)
spikeTimesI, spikingNeuronsI, poprI = preprocess_monitors(smi, popri, calc_ISI=False)
# call detect_oscillation functions:
avgReplayInterval = replay(ISIhist[3:16])  # bins from 150 to 850 (range of interest)
print "replay:", avgReplayInterval
meanEr, rEAC, maxEAC, tMaxEAC, maxEACR, tMaxEACR, fE, PxxE, avgRippleFE, ripplePE = ripple(poprE, 1000)
avgGammaFE, gammaPE = gamma(fE, PxxE)
meanIr, rIAC, maxIAC, tMaxIAC, maxIACR, tMaxIACR, fI, PxxI, avgRippleFI, ripplePI = ripple(poprI, 1000)
avgGammaFI, gammaPI = gamma(fI, PxxI)
# plot results
plot_raster_ISI(spikeTimesE, spikingNeuronsE, [ISIhist, bin_edges], "blue", multiplier_=1)
plot_PSD(poprE, rEAC, fE, PxxE, "Pyr_population", 'b-', multiplier_=1)
plot_PSD(poprI, rIAC, fI, PxxI, "Bas_population", 'g-', multiplier_=1)
_, _ = plot_zoomed(spikeTimesE, spikingNeuronsE, poprE, "Pyr_population", "blue", multiplier_=1)
plot_zoomed(spikeTimesI, spikingNeuronsI, poprI, "Bas_population", "green", multiplier_=1, Pyr_pop=False)

