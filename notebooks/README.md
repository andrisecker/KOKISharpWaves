## New network model

This folder contains an improved version of the model (still under development -
see [optimization](https://github.com/andrisecker/KOKISharpWaves/tree/master/optimization) folder)

------------------------------------------------------

Improvements:

* new cell models, optimized with experimental traces
* double exponential synapses
* added a different recurrent weight distribution (based on a symetric STDP rule)
* simulation transfered to Brian2 (to be paralellized...)

To run the scripts, [Brian2](http://brian2.readthedocs.io/en/stable/introduction/install.html) and run:

	python stdp_network_new_brian2.py (recreate correctly scaled weight matrix - can be symetric or asymetric STDP rule)
	python spw_network_new_brian2.py

or install [jupyter](http://jupyter.org/install.html) and run:

	jupyter notebooks
