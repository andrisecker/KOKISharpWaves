## Sharp waves (Computational neuroscience, KOKI-HAS)

------------------------------------------------------

Reference: [BSc thesis](https://drive.google.com/drive/folders/0B089tpx89mdXdjdkbk9JSjBjMDQ) *Modeling the network dynamics underlying hippocampal sharp waves and sequence replay.*

    > With the sripts in the repository one can create a [CA3 network model](https://github.com/andrisecker/KOKISharpWaves/blob/master/CA3_network_model.pdf) and examine the network dynamics during hippocampal sharp waves.

To run the scripts, [install Brian (version 1)](http://brian.readthedocs.org/en/latest/installation.html) and run:

    git clone https://github.com/andrisecker/KOKISharpWaves.git  # Clone this GitHub repository
    cd KOKISharpWaves/scripts
    python generate_spike_train.py  # generate spike trains (as exploration of a maze) -> files/spikeTrainR.npz
    python stdp_network_b.py  # learns the recurrent weight (via STDP, based on the spiketrain) -> files/wmx.txt
    python spw_network4a_1.py  # creates the network, runs the simulation, extracts dynamic features

![](https://raw.githubusercontent.com/andrisecker/KOKISharpWaves/master/E-I_network.png)

------------------------------------------------------

Other features:

    python spw_network4_automatized.py  # investigates into network dynamics with varios scaling factor (of the weight matrix)

    python spw_network4_inputs.py  # investigates into network dynamics with different outer inputs



bayesian_inference.py + spikes.npz + PFstarts.npz -> decoding of the place from spikes saved from spw...
