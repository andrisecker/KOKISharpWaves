## New network model

This folder contains an improved version of the model (still under development)

------------------------------------------------------

Improvements:

* new cell models, optimized with experimental traces
* double exponential synapses
* added a different recurrent weight distribution (based on a symetric STDP rule)
* simulation transfered to Brian2 (to be paralellized...)

To run the notebooks, install [Brian2](http://brian2.readthedocs.io/en/stable/introduction/install.html), and [jupyter](http://jupyter.org/install.html) and run:

	jupyter notebook

or just run:

	python spw_network_new_brian2.py