#!/usr/bin/python
# -*- coding: utf8 -*-

from brian import *
from brian.library.IF import *
import numpy as np
import matplotlib.pyplot as plt
import os

fIn = 'wmxC.txt'
fOut ='resultsC.txt'

SWBasePath = os.path.split(os.path.split(__file__)[0])[0]

fName = os.path.join(SWBasePath, 'files', fIn)

first = 0.5
last = 2.5
data_points = 21

multipliers = np.linspace(first, last, data_points)


NE=4000
NI=1000

eps_pyr=0.16
eps_bas=0.4

z=1*nS
gL_Pyr = 4.333e-3 * uS  # 3.3333e-3
tauMem_Pyr = 60.0 * ms
Cm_Pyr = tauMem_Pyr * gL_Pyr
Vrest_Pyr = -70.0 * mV
reset_Pyr = -53.0*mV
theta_Pyr = -50.0*mV
tref_Pyr = 5*ms

# Adaptation parameters for pyr cells
a_Pyr = -0.8*nS    # nS    Subthreshold adaptation conductance
# moves threshold up
b_Pyr = 0.04*nA     # nA    Spike-triggered adaptation
# -> decreases the slope of the f-I curve
delta_T_Pyr = 2.0*mV  # 0.8    # mV    Slope factor
tau_w_Pyr = 300*ms  # 88 # 144.0   # ms    Adaptation time constant

v_spike_Pyr = theta_Pyr + 10*delta_T_Pyr

gL_Bas = 5.0e-3*uS  # 7.14293e-3
tauMem_Bas = 14.0*ms
Cm_Bas = tauMem_Bas * gL_Bas
Vrest_Bas = -70.0*mV
reset_Bas = -64.0*mV  # -56.0
theta_Bas  = -50.0*mV
tref_Bas = 0.1*ms  # 0.1*ms

'''
gL_Bas = 10.0e-3*uS #5.0e-3 *uS#7.14293e-3
tauMem_Bas = 14.0*ms
Cm_Bas = tauMem_Bas * gL_Bas
Vrest_Bas = -70.0*mV
reset_Bas = -55*mV #-64.0*mV #-56.0
theta_Bas  = -50.0 *mV
tref_Bas = 2.0*ms
'''

# Initialize the synaptic parameters
# J_PyrExc  = 1.0e-5*0
# J_Pyr_Exc_factor = 20.e8    #20.0e8 #14.5e8 #12.0e8 #10.0e8 #7.0e8

'''
J_PyrInh  = 0.12500    # (nS)
J_BasExc  = 5.2083/2.
J_BasInh  = 0.15 #1.0  #0.15     #0.083333e-3
'''

J_PyrInh = 0.125
J_BasExc = 5.2083
J_BasInh = 0.15  # 0.08333 #0.15

print "J_PyrInh", J_PyrInh
print "J_BasExc", J_BasExc
print "J_BasInh", J_BasInh

J_PyrMF = 5.0  # 8.0 #2.0 #5.0

# Synaptic reversal potentials
E_Exc = 0.0*mV
E_Inh = -70.0*mV

# Synaptic time constants
tauSyn_PyrExc = 10.0*ms
tauSyn_PyrInh = 3.0*ms
tauSyn_BasExc = 3.0*ms
tauSyn_BasInh = 1.5*ms

# Synaptic delays
delay_PyrExc = 3.0*ms
delay_PyrInh = 1.5*ms
delay_BasExc = 3.0*ms
delay_BasInh = 1.5*ms

p_rate_mf = 5.0*Hz #10.0*Hz

# Creating populations
eqs_adexp = '''
dvm/dt=(gL_Pyr*(Vrest_Pyr-vm)+gL_Pyr*delta_T_Pyr*exp((vm- theta_Pyr)/delta_T_Pyr)-w -(g_ampa*z*(vm-E_Exc)+g_gaba*z*(vm-E_Inh)))/Cm_Pyr : volt
dw/dt=(a_Pyr*(vm- Vrest_Pyr )-w)/tau_w_Pyr : amp
dg_ampa/dt = -g_ampa/tauSyn_PyrExc : 1
dg_gaba/dt = -g_gaba/tauSyn_PyrInh : 1
'''

reset_adexp = '''
vm = reset_Pyr
w += b_Pyr
'''

eqs_bas = '''
dvm/dt = (-gL_Bas*(vm-Vrest_Bas)-(g_ampa*z*(vm-E_Exc)+g_gaba*z*(vm-E_Inh)))/Cm_Bas :volt
dg_ampa/dt = -g_ampa/tauSyn_BasExc : 1
dg_gaba/dt = -g_gaba/tauSyn_BasInh : 1
'''

def myresetfunc(P, spikes):
    P.vm_[spikes] = reset_Pyr   # reset voltage
    P.w_[spikes] += b_Pyr  # low pass filter of spikes (adaptation mechanism)


def replay():
    '''
    decides if there is a replay or not, and stores the results in a nunmpy ndarray (X)
    :param : 
    :return: adds plus one column to the result matrix (X)
    '''

    bin_means = np.linspace(175, 825, 14)
    tmp1 = isi.count[3:17]
    max_index = np.argmax(tmp1)

    if 1 <= max_index <= 12:
        tmp2 = tmp1[max_index-1:max_index+2]
        avg_replay_interval = (tmp1[max_index-1]*bin_means[max_index-1] + tmp1[max_index]*bin_means[max_index] + tmp1[max_index+1]*bin_means[max_index+1])/(tmp1[max_index-1]+tmp1[max_index]+tmp1[max_index+1])
    else:
        tmp2 = []

    if sum(int(i) for i in tmp1)*0.9 < sum(int(i) for i in tmp2):
        X[:,k] = [multiplier, meanre, reac[1:].max(), reac[1:].argmax()+1, reac[3:8].max(), reac[3:8].argmax()+3, meanri, riac[1:].max(), riac[1:].argmax()+1, riac[3:8].max(), riac[3:8].argmax()+3, 1, avg_replay_interval]
        print 'avg. replay interval: ', avg_replay_interval
    else:
        X[:,k] = [multiplier, meanre, reac[1:].max(), reac[1:].argmax()+1, reac[3:8].max(), reac[3:8].argmax()+3, meanri, riac[1:].max(), riac[1:].argmax()+1, riac[3:8].max(), riac[3:8].argmax()+3, 0, np.nan]
        print 'Not replay'


X = np.zeros((13, data_points))


for k in range(0, data_points):

    multiplier = multipliers[k]
    print "multiplier=" ,multiplier

    SCR = SimpleCustomRefractoriness(myresetfunc, tref_Pyr, state='vm')

    eqs=Brette_Gerstner(C=Cm_Pyr,gL=gL_Pyr,EL=Vrest_Pyr,VT=theta_Pyr,DeltaT=delta_T_Pyr,tauw=tau_w_Pyr,a=a_Pyr) + """ dg_ampa/dt = -g_ampa/tauSyn_PyrExc : 1
    dg_gaba/dt = -g_gaba/tauSyn_PyrInh : 1"""
    
    PE=NeuronGroup(NE,model=eqs_adexp,threshold=v_spike_Pyr,reset=SCR)
    PI = NeuronGroup(NI, model=eqs_bas, threshold=theta_Bas, reset=reset_Bas, refractory=tref_Bas)
    PE.vm = Vrest_Pyr
    PE.g_ampa = 0
    PE.g_gaba = 0
    PI.vm  = Vrest_Bas
    PI.g_ampa = 0
    PI.g_gaba = 0

    MF = PoissonGroup(NE, p_rate_mf)
        
    print 'Connecting the network'
    
    Cext = IdentityConnection(MF,PE, 'g_ampa', weight=J_PyrMF)
    Cee = Connection(PE,PE, 'g_ampa', delay=delay_PyrExc)
    
    f = file(fName, 'r')

    Wee = [line.split() for line in f]

    f.close()
    
    for i in range(NE):
        Wee[i][:] = [float(x) * 1.e9 * multiplier for x in Wee[i]]  # *0.6
        Wee[i][i] = 0.
        
    Cee.connect(PE, PE, Wee)
    Cei = Connection(PE, PI, 'g_ampa', weight=J_BasExc, sparseness=eps_pyr, delay=delay_BasExc)
    Cie = Connection(PI, PE, 'g_gaba', weight=J_PyrInh, sparseness=eps_bas, delay=delay_PyrInh)
    Cii = Connection(PI, PI, 'g_gaba', weight=J_BasInh, sparseness=eps_bas, delay=delay_PyrInh)
    
    print 'Connections done'

    sme = SpikeMonitor(PE)
    smi = SpikeMonitor(PI)
    popre = PopulationRateMonitor(PE, bin=0.001)
    popri = PopulationRateMonitor(PI, bin=0.001)
    poprext = PopulationRateMonitor(MF, bin=0.001)
    bins = [0*ms, 50*ms, 100*ms, 150*ms, 200*ms, 250*ms, 300*ms, 350*ms, 400*ms, 450*ms, 500*ms, 550*ms, 600*ms, 650*ms, 700*ms, 750*ms, 800*ms, 850*ms, 900*ms, 950*ms, 1000*ms]
    isi = ISIHistogramMonitor(PE,bins)


    run(10000*ms,report='text')

    # Saved figure
    figure()
    subplot(2, 1, 1)
    raster_plot(sme, spacebetweengroups=1, title='Raster plot', newfigure=False)
    subplot(2, 1, 2)
    hist_plot(isi, title='ISI histogram', newfigure=False)
    xlim([0, 1000])

    figName = os.path.join(SWBasePath, 'figures', str(multiplier)+'*.png')
    savefig(figName)
    close()

    # Saved results
    meanre = np.mean(popre.rate)  # mean exc. rate
    reub = popre.rate - meanre
    revar = np.sum(reub**2)
    reac = np.correlate(reub, reub, 'same') / revar  # max exc. autocorr: reac[1:].max() at reac[1:]argmax()+1 [ms]
    reac = reac[len(reac)/2:]  # max exc. autocorr. in ripple range: reac[3:8].max() at reac[3:8].argmax()+3 [ms]

    meanri=np.mean(popri.rate)  # mean inh. rate
    riub=popri.rate - meanri
    rivar=np.sum(riub**2)
    riac=np.correlate(riub, riub, 'same') / rivar
    riac=riac[len(riac)/2:]


    replay()  # function define before the for loop

    # Reinitialize variables
    reinit(states=True)


# Plots
fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(multipliers, X[12, :])
ax1.set_title('Average replay interval')
ax1.set_xlim(first, last)

ax2 = fig.add_subplot(2, 1, 2)
ax3 = ax2.twinx()
ax2.plot(multipliers, X[1, :], 'b-')
ax2.set_ylabel(ylabel='PC (exc.) rate', color='blue')
ax3.plot(multipliers, X[6, :], 'g-')
ax3.set_ylabel('BC (inh.) rate', color='green')
ax2.set_xlabel('scale factors')
ax2.set_xlim(first, last)
ax2.set_title('Mean firing rates')

savefig(os.path.join(SWBasePath, 'figures', 'replay_and_firing_rates.png'))

fig2 = plt.figure()
ax1 = fig2.add_subplot(2, 1, 1)
ax1.plot(multipliers, X[2, :], label='PC (exc.)')
ax1.plot(multipliers, X[7, :], label='BC (inh.)')
ax1.legend()
ax1.set_xlim(first, last)
ax1.set_title('Maximum autocerrelations')

ax2 = fig2.add_subplot(2, 1, 2)
ax2.plot(multipliers, X[4, :], label='PC (exc.)')
ax2.plot(multipliers, X[9, :], label='BC (inh.)')
ax2.legend()
ax2.set_xlabel('scale factors')
ax2.set_xlim(first, last)
ax2.set_title('Maximum autocerrelations in ripple range')

savefig(os.path.join(SWBasePath, 'figures', 'autocorrelations.png'))

# saving result array (X)    
fName= os.path.join(SWBasePath, 'files', fOut)
header = 'Multiplier, Mean_exc.rate, Max.exc.AC., at[ms], Max.exc.AC.in_ripple_range, at[ms], Mean_inh.rate, Max.inh.AC., at[ms], Max.inh.AC.in_ripple_range, at[ms], replay, avg. replay interval'
np.savetxt(fName, X, fmt='%.6f', delimiter='\t', header=header)
