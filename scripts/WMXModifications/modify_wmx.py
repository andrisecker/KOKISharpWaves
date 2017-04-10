#!/usr/bin/python
# -*- coding: utf8 -*-
'''
weight matrix modification (see also helper file: wmx_modifications.py)
author: András Ecker last update: 10.2015
'''

import os
import matplotlib.pyplot as plt
from wmx_modifications import *


fIn = 'wmxR_asym.txt'
fOut = 'wmxR_asym_gauss.txt'

SWBasePath = '/'.join(os.path.abspath(__file__).split('/')[:-3]) 

wmxO = load_Wee(os.path.join(SWBasePath, "files", fIn))

# wmxM = gauss(wmxO)
wmxM = gauss_rectangle(wmxO)
# wmxM = mean_rectangle(wmxO)
# wmxM = shuffle_rows_cols(wmxO)
# wmxM = shuffle_block_rows_cols(wmxO)
# wmxM = avg_weak_weights(wmxO)
# wmxM = avg_x_weak_weights(wmxO, 3995)
# wmxM = disconnected(wmxO)
# wmxM = binary_weights(wmxO, 0.5)
# wmxM = shuffle_blocks(wmxO, 200)

print "modification done"

np.fill_diagonal(wmxM, 0)  # just to make sure
assert np.shape(wmxM) == (4000, 4000), "output shape is not 4000*4000"

fName = os.path.join(SWBasePath, "files", fOut)
np.savetxt(fName, wmxM)


# Plot matrices:
fig = plt.figure(figsize=(12, 6))

ax = fig.add_subplot(1, 2, 1)
i = ax.imshow(wmxO, interpolation='None')
ax.set_title('Original weight matrix')

ax2 = fig.add_subplot(1, 2, 2)
i = ax2.imshow(wmxM, interpolation='None')
ax2.set_title('Modified weight matrix')

plt.show()

