## Optimize network parameters with new cell models

This folder contains scripts to tune the synaptic weights of the network (mainly using BluePyOpt)

------------------------------------------------------

Scripts:

* clamp_cell.py - finds clampint current for specified clamping voltage
* analyse_EPS.py - analyses EPSP and EPSC based on the learned synaptic weights
* analyse_STDP.py - analyses STDP rule and it's effects on EPSP changes
* analyse_BasNetwork.py - creates a purely BC network driven by external input (calculated from mean PC rates) -> should result ripple oscillation
* run_sim.py - runs single simulation with the updated cells and synapses
* sim_evaluator.py - BluePyOpt evaluator to score single runs
* optimize_network.py - optimizes parameters in a given range using BluePyOpt's evolutionary algorithm (DEAP)
* analyse_checkpoint.py - analyses results, saved during (and after) optimization
* hand_tune_network.py - script for additional exact weight specification for final hand tuning

To run the scripts [install Brian2](http://brian2.readthedocs.io/en/stable/introduction/install.html) and run:

    python clamp_cell.py  # finds I_hold
    python analyse_EPS.py  # saves EPS* plots
    python analyse_STDP.py  # saves STDP plots
    python analyse_BasNetwork.py  # saves plots from purely inh. network (driven externally)

To run the big optimization script [install BluePyOpt](https://github.com/BlueBrain/BluePyOpt) too (+ change the parameters in optimize_network.py) and run:

	python optimize_network.py  # runs optimization (use a cluster for big ones...)
	python analyse_checkpoint.py  # reloads results
