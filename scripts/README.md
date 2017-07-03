Reference: [BSc thesis](https://drive.google.com/file/d/0B089tpx89mdXZk55dm0xZm5adUE/view) *Modeling the network dynamics underlying hippocampal sharp waves and sequence replay.*

------------------------------------------------------

The foder contains all the python scripts needed to reproduce the reference.

Teaching the recurrent weights:

* poisson_proc.py (helper functions used by generate_spike_trains.py)
* generate_spike_train.py
* stdp_network_b.py (Brian script: STDP during exploration)

Analyse network dynamics:

* plots.py (helper file used by the following scripts)
* detect_oscillations.py (helper functions used by the following scripts)
* spw_network4a_1.py (Brian script: runs the simulation, extracts dynamic features)
* spw_network4a_1_brian2.py (same Brian2 script - bit faster)
* spw_network4_automatized.py (Brian script: investigates into network dynamics with varios scaling factor (of the weight matrix))

------------------------------------------------------

Bayesian decoding:

* bayesian_decoding.py (infers to place from spikes recorded from place cells)
* analyse_bayesian_decoding.py (analyse the resulted (from the inference) angular velocity)

> note: these are not included in the reference (and do not work properly yet) analyse the resulted (from the inference) angular velocity))analyse the resulted (from the inference) angular velocity)elocity))analyse the resulted (from the inference) angular velocity)
