## Optimize network parameters with new cell models

This folder contains scripts to tune the synaptic weights of the network (mainly using BluePyOpt)

------------------------------------------------------

Scripts:

* analyse_EPSC.py - analyses EPSP and EPSC based on the learned synaptic weights
* analyse_BasNetwork.py - creates a purely BC network driven by external input (calculated from mean PC rates) -> should result ripple oscillation
* run_sim.py - runs single simulation with the updated cells and synapses
* sim_evaluator.py - BluePyOpt evaluator to score single runs
* optimize_network.py - optimizes parameters in a given range using BluePyOpt's evolutionary algorithm (DEAP)
* hand_tune_network.py - script for additional exact weight specification for final hand tuning

To run the scripts, install [Brian2](http://brian2.readthedocs.io/en/stable/introduction/install.html), and [BluePyOpt](https://github.com/BlueBrain/BluePyOpt) (+ change the parameters in optimize_network.py) and run:

	python optimize_network.py