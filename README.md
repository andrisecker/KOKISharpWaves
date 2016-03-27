## Sharp waves (Computational neuroscience, KOKI-HAS)

------------------------------------------------------

Reference: [BSc thesis](https://drive.google.com/drive/folders/0B089tpx89mdXdjdkbk9JSjBjMDQ) *Modeling the network dynamics underlying hippocampal sharp waves and sequence replay.*

To run the scripts, [install Brian (version 1)](http://brian.readthedocs.org/en/latest/installation.html) and run:

    git clone https://github.com/andrisecker/KOKISharpWaves.git  # Clone this GitHub repository
    cd KOKISharpWaves/scripts
    python generate_spike_train.py  # generate spike trains (as exploration of a maze) -> files/spikeTrainR.npz
    python stdp_network_b.py  # learns the recurrent weight (via STDP, based on the spiketrain) -> files/wmx.txt

spw_network4a_1.py + detect_oscillations.py - activity during sleep/rest (scaling factor = 1) (and detects replays, ripple-, gamma oscillation)

spw_network4_automatized.py + detect_oscillations.py - checks the activity with varios scaling factors (and detects replays, ripple-, gamma oscillation)

spw_network4_inputs.py + detect_oscillations.py - script for examination of the network dynamics to different inpust

bayesian_inference.py + spikes.npz + PFstarts.npz -> decoding of the place from spikes saved from spw...
