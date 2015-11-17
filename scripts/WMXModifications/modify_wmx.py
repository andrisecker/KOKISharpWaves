#!/usr/bin/python
# -*- coding: utf8 -*-

import matplotlib.pyplot as plt
import os
from wmx_modifications import *


fIn = 'wmxR.npz'
fOut = 'wmxR_shuffle_blocks.txt'

SWBasePath = os.path.split(os.path.split(os.path.split(__file__)[0])[0])[0]

fName = os.path.join(SWBasePath, 'files', 'wmxR.npz')
npzFile = np.load(fName)
wmxO = npzFile['wmx']

print 'original matrix loaded'

# wmxM = gauss(wmxO)
# wmxM = gauss_rectangle(wmxO)
# wmxM = mean_rectangle(wmxO)
# wmxM = shuffle_rows_cols(wmxO)
# wmxM = shuffle_block_rows_cols(wmxO)
# wmxM = avg_weak_weights(wmxO)
# wmxM = avg_x_weak_weights(wmxO, 3990)
# wmxM = disconnected(wmxO)
# wmxM = binary_weights(wmxO, 0.5)
wmxM = shuffle_blocks(wmxO, 200)

print 'modification done'

assert np.shape(wmxM) == (4000, 4000)

fName = os.path.join(SWBasePath, 'files', fOut)
np.savetxt(fName, wmxM)

# Plots
fig = plt.figure(figsize=(10, 8))

ax = fig.add_subplot(1, 2, 1)
i = ax.imshow(wmxO, interpolation='None')
ax.set_title('Original weight matrix')

ax2 = fig.add_subplot(1, 2, 2)
i = ax2.imshow(wmxM, interpolation='None')
ax2.set_title('Modified weight matrix')

plt.show()
