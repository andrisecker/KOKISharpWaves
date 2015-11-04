#!/usr/bin/python
# -*- coding: utf8 -*-

import numpy as np
import matplotlib.pyplot as plt
import os
from wmx_modifications import *


fIn = 'wmxR.npz'
fOut = 'wmxR_mix_block_rows_cols.txt'

SWBasePath = os.path.split(os.path.split(os.path.split(__file__)[0])[0])[0]

fName = os.path.join(SWBasePath, 'files', 'wmxR.npz')
npzFile = np.load(fName)
wmxO = npzFile['wmx']
print 'original matrix loaded'

# wmxM = gauss(wmxO)
# wmxM = gauss_rectangle(wmxO)
# wmxM = mean_rectangle(wmxO)
# wmxM = mix_rows_cols(wmxO)
wmxM = mix_block_rows_cols(wmxO)
print 'modification done'

fName = os.path.join(SWBasePath, 'files', fOut)
np.savetxt(fName, wmxM)

# Plots
fig = plt.figure(figsize=(10, 8))

ax = fig.add_subplot(1, 2, 1)
i = ax.imshow(wmxO, interpolation='None')
ax.set_title('Original weight matrix')

ax = fig.add_subplot(1, 2, 2)
i = ax.imshow(wmxM, interpolation='None')
ax.set_title('Modified weight matrix')

plt.show()
