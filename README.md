## Sharp waves (Computational neuroscience, KOKI-HAS)

Reference: [BSc thesis](https://drive.google.com/file/d/0B089tpx89mdXZk55dm0xZm5adUE/view) *Modeling the network dynamics underlying hippocampal sharp waves and sequence replay.*

------------------------------------------------------

> With the sripts in the repository one can create a [CA3 network model](https://github.com/andrisecker/KOKISharpWaves/blob/master/CA3_network_model.pdf) and examine the network dynamics during hippocampal SWRs (Sharp Wave-Ripples).

To run the scripts, [install Brian (version 1)](http://brian.readthedocs.org/en/latest/installation.html) and run:

    git clone https://github.com/andrisecker/KOKISharpWaves.git  # Clone this GitHub repository
    cd KOKISharpWaves
    mkdir figures  # creates directory for the saved figures
    cd scripts
    python generate_spike_train.py  # generate CA3 like spike trains (as exploration of a maze) -> files/spikeTrainR.npz
    python stdp_network_b.py  # learns the recurrent weight (via STDP, based on the spiketrain) -> files/wmx.txt
    python spw_network4a_1.py  # creates the network, runs the simulation, extracts dynamic features

![](https://raw.githubusercontent.com/andrisecker/KOKISharpWaves/master/E-I_network.png)

[network](https://github.com/andrisecker/KOKISharpWaves/blob/master/CA3_network_model.pdf) structure with the learned recurrent weightmatrix

![](https://raw.githubusercontent.com/andrisecker/KOKISharpWaves/master/spw.png)

extraction of dynamic features during SWRs

### Other features:

    python spw_network4_automatized.py  # investigates into network dynamics with varios scaling factor (of the weight matrix)

![](https://raw.githubusercontent.com/andrisecker/KOKISharpWaves/master/autamated_evaluation.png)

automated evaluation of the networks dynamics with differently scaled recurrent weightmatrix

    python spw_network4_BasInputs_f.py  # investigates into network dynamics with different outer inputs to the Basket cell population

    python bayesian_inference.py  # Bayesian decoding of place from spikes saved from spw...

[Bayesian decoding](https://github.com/andrisecker/KOKISharpWaves/blob/master/Bayesian_inference.pdf) of place (in a circle maze) from spike trains recorded during SWRs

![](https://raw.githubusercontent.com/andrisecker/KOKISharpWaves/master/bayesian_decoding.png)

Raster plot of the activity and infered place from spike trains + analysis of the angular velocity during SWRs

------------------------------------------------------

> note: (with some IDE) one has to change the PATH (SWBasePath at the top of the scripts)

