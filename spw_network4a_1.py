from brian import *
from brian.library.IF import *
import numpy as np


NE=4000
NI=1000


eps_pyr=0.16
eps_bas=0.4

z=1*nS
gL_Pyr = 4.333e-3 * uS  #3.3333e-3
tauMem_Pyr = 60.0 * ms
Cm_Pyr = tauMem_Pyr * gL_Pyr
Vrest_Pyr = -70.0 * mV
reset_Pyr = -53.0*mV
theta_Pyr = -50.0*mV
tref_Pyr = 5*ms

#Adaptation parameters for pyr cells
a_Pyr = -0.8*nS    #nS    Subthreshold adaptation conductance
# moves threshold up
b_Pyr = 0.04*nA     #nA    Spike-triggered adaptation
# -> decreases the slope of the f-I curve
delta_T_Pyr = 2.0*mV #0.8    #mV    Slope factor
tau_w_Pyr = 300*ms #88 #144.0   #ms    Adaptation time constant

v_spike_Pyr = theta_Pyr + 10*delta_T_Pyr

gL_Bas = 5.0e-3*uS #7.14293e-3
tauMem_Bas = 14.0*ms
Cm_Bas = tauMem_Bas * gL_Bas
Vrest_Bas = -70.0*mV
reset_Bas = -64.0*mV #-56.0
theta_Bas  = -50.0*mV
tref_Bas = 0.1*ms #0.1*ms


"""
gL_Bas = 10.0e-3*uS #5.0e-3 *uS#7.14293e-3
tauMem_Bas = 14.0*ms
Cm_Bas = tauMem_Bas * gL_Bas
Vrest_Bas = -70.0*mV
reset_Bas = -55*mV #-64.0*mV #-56.0
theta_Bas  = -50.0 *mV
tref_Bas = 2.0*ms
"""
# Initialize the synaptic parameters
#J_PyrExc  = 1.0e-5*0
#J_Pyr_Exc_factor = 20.e8    #20.0e8 #14.5e8 #12.0e8 #10.0e8 #7.0e8
"""
J_PyrInh  = 0.12500    # (nS)
J_BasExc  = 5.2083/2.
J_BasInh  = 0.15 #1.0  #0.15     #0.083333e-3
"""

J_PyrInh = 0.125
J_BasExc = 5.2083
J_BasInh = 0.15 #0.08333 #0.15

print "J_PyrInh", J_PyrInh
print "J_BasExc", J_BasExc
print "J_BasInh", J_BasInh


J_PyrMF = 5.0 #8.0 #2.0 #5.0

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


#Creating populations
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
 P.vm[spikes] = reset_Pyr   #reset voltage
 P.w[spikes] += b_Pyr  #low pass filter of spikes (adaptation mechanism)

SCR = SimpleCustomRefractoriness(myresetfunc, tref_Pyr, state='vm')

eqs=Brette_Gerstner(C=Cm_Pyr,gL=gL_Pyr,EL=Vrest_Pyr,VT=theta_Pyr,DeltaT=delta_T_Pyr,tauw=tau_w_Pyr,a=a_Pyr) + """ dg_ampa/dt = -g_ampa/tauSyn_PyrExc : 1
dg_gaba/dt = -g_gaba/tauSyn_PyrInh : 1"""
PE=NeuronGroup(NE,model=eqs_adexp,threshold=v_spike_Pyr,reset=SCR)

#PE = NeuronGroup(NE, model=eqs_adexp, threshold=v_spike_Pyr, reset=reset_adexp, refractory=tref_Pyr)

PI = NeuronGroup(NI, model=eqs_bas, threshold=theta_Bas, reset=reset_Bas, refractory=tref_Bas)
PE.vm = Vrest_Pyr
PE.g_ampa = 0
PE.g_gaba = 0
PI.vm  = Vrest_Bas
PI.g_ampa = 0
PI.g_gaba = 0

MF = PoissonGroup(NE, p_rate_mf)


print "connecting the network"
Cext = IdentityConnection(MF,PE, 'g_ampa', weight=J_PyrMF)

Cee = Connection(PE,PE, 'g_ampa', delay=delay_PyrExc)
f = file("wmx.txt", "r")
Wee=[line.split() for line in f]
f.close()
for i in range(NE):
    Wee[i][:] = [float(x)*1.e9 for x in Wee[i]]
    Wee[i][i] = 0.

Cee.connect(PE,PE,Wee)

Cei = Connection(PE,PI, 'g_ampa', weight=J_BasExc, sparseness=eps_pyr, delay=delay_BasExc)
Cie = Connection(PI,PE, 'g_gaba', weight=J_PyrInh, sparseness=eps_bas, delay=delay_PyrInh)
Cii = Connection(PI,PI, 'g_gaba', weight=J_BasInh, sparseness=eps_bas, delay=delay_PyrInh)

print "connections done"

sme = SpikeMonitor(PE)
smi = SpikeMonitor(PI)
#smp = SpikeMonitor(MF)
vme = StateMonitor(PE, 'vm', record=[1,2,3])
vmi = StateMonitor(PI, 'vm', record=0)
popre = PopulationRateMonitor(PE,bin=0.001)
popri = PopulationRateMonitor(PI,bin=0.001)
poprext = PopulationRateMonitor(MF, bin = 0.001)

#@network_operation
#def reset_voltage():
#    if (defaultclock.t == 200*ms):
#        MF = PoissonGroup(NE, 10*Hz)
#        print "200 ms"
#        

# fpe = FileSpikeMonitor(PE, './PE.dat')
# fpi = FileSpikeMonitor(PI, './PI.dat')

run(5000*ms,report='text')

# fpe.close()
# fpi.close()

figure()
subplot(2,1,1)
raster_plot(sme,spacebetweengroups=1)
subplot(2,1,2)
raster_plot(smi)


figure()
subplot(3,1,1)
plot(popre.times, popre.rate)
subplot(3,1,2)
plot(popri.times, popri.rate)
subplot(3,1,3)
plot(poprext.times, poprext.rate)


figure()
subplot(1,2,1)
plot(vme.times,vme[1]/mV)
plot(vme.times,vme[2]/mV)
plot(vme.times,vme[3]/mV)
plot(sme[1], linspace(0,0,len(sme[1])) ,'*r')
title("Pyr cell Vm")
subplot(1,2,2)
plot(vmi.times, vmi[0]/mV)
title("Bas cell Vm")

meanre=np.mean(popre.rate)
print "Mean excitatory rate: ", meanre
reub=popre.rate-meanre
revar=np.sum(reub**2)
reac=np.correlate(reub, reub, "same")/revar
reac=reac[len(reac)/2:]
figure()
subplot(1,2,1)
plot(reac[:100])
print "Maximum exc. autocorrelation: ", reac[1:].max(), "at ", reac[1:].argmax()+1, "ms"
print "Maximum exc. AC in ripple range: ", reac[3:8].max(), "at ", reac[3:8].argmax()+3, "ms"

meanri=np.mean(popri.rate)
print "Mean inhibitory rate: ", meanri
riub=popri.rate-meanri
rivar=np.sum(riub**2)
riac=np.correlate(riub, riub, "same")/rivar
riac=riac[len(riac)/2:]
subplot(1,2,2)
plot(riac[:100])
show()
print "Maximum inh. autocorrelation: ", riac[1:].max(), "at ", riac[1:].argmax()+1, "ms"
print "Maximum inh. AC in ripple range: ", riac[3:8].max(), "at ", riac[3:8].argmax()+3, "ms"
