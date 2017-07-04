#!/usr/bin/python
# -*- coding: utf8 -*-
'''
weight matrix modification (see also helper file: wmx_modifications.py)
author: András Ecker last update: 06.2017
'''

import os
import sys
import matplotlib.pyplot as plt
from wmx_modifications import *
SWBasePath = os.path.sep.join(os.path.abspath('__file__').split(os.path.sep)[:-3])
# add the 'scripts' directory to the path (import the modules)
sys.path.insert(0, os.path.sep.join([SWBasePath, 'scripts']))
from plots import plot_wmx, plot_wmx_avg


STDP_mode = "asym"
fIn = "wmxR_%s.txt"%STDP_mode
fOut = "wmxR_%s_shuf_subpop_inp.txt"%STDP_mode

wmxO = np.genfromtxt(os.path.join(SWBasePath, "files", fIn))
print "weight matrix loaded"

#wmxM = shuffle(wmxO)
#wmxM = binary_weights(wmxO, 0.5)
#wmxM = shuffle_blocks(wmxO, 200)
wmxM = shuffle_subpop_input_weights(wmxO, 500)
# ===============================================
# wmxM = shuffle_block_rows_cols(wmxO)
# wmxM = avg_weak_weights(wmxO)
# wmxM = avg_x_weak_weights(wmxO, 3975)

assert np.shape(wmxM) == (4000, 4000), "output shape is not 4000*4000"
assert (wmxM >= 0).all(), "negative weights in the modified matrix!"
np.fill_diagonal(wmxM, 0)  # just to make sure
print "modification done"

fName = os.path.join(SWBasePath, "files", fOut)
np.savetxt(fName, wmxM)

# Plot modified matrix:
plot_wmx(wmxM, "wmx_mod")
plot_wmx_avg(wmxM, 100, "wmx_avg_mod")

plt.show()

