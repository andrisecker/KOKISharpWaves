import brian_no_units
from brian import *
import numpy


N = 4000
Npop = 50
Nn = N/50

#importing spike times from file
f = file("spikeTrain.txt", "r")
data=[line.split() for line in f]
f.close()

spiketimes = []
for neuron in range(N):
    data[neuron][: ]=[float(x) for x in data[neuron]]
    nrn = neuron * numpy.ones(len(data[neuron]))
    z= zip(nrn, data[neuron])
    spiketimes.append(z)

spiketimes = [item for sublist in spiketimes for item in sublist]


PC = SpikeGeneratorGroup(N, spiketimes)
for i in range(Npop):
    PC.subgroup(Nn)

Conn = Connection(PC,PC,weight=0.1e-9, sparseness=0.16)
stdp=ExponentialSTDP(Conn,20e-3,20e-3,0.01,-0.010,wmax=40e-9,interactions='all',update='additive')



sp = SpikeMonitor(PC, record =True)
run(400,report='text')

weightmx = [[Conn[i,j] for j in range(N)] for i in range(N)]

numpy.savetxt('wmx.txt', weightmx)

import pylab as plt
figure()
raster_plot(sp)
#plt.hist(sp.spiketimes[0], linspace(0,0.5,100))
"""
figure()
m=plt.matshow(weightmx)
plt.colorbar(m)
"""
show()
