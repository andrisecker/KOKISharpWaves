#!/usr/bin/python
# -*- coding: utf8 -*-
'''
helper file to plot dynamics (and the weight matrix)
authors: Bence Bagi, András Ecker, last update: 05.2017
'''

import os
import numpy as np
import matplotlib.pyplot as plt

SWBasePath = os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-2])
figFolder = os.path.join(SWBasePath, "figures")


def plot_raster_ISI(spikeTimes, spikingNeurons, hist, color_, multiplier_):
    """
    saves figure with raster plot and ISI distribution
    (note: the main reason of this function is that Brian2 doesn't have ISIHistogramMonitor and the corresponding plot)
    :param spikeTimes, spikingNeurons: used for raster plot - precalculated by detect_oscillation.py/preprocess_spikes
    :param hist: used for plotting InterSpikeInterval histogram - result of a numpy.histogram call
    :param color_, multiplier_: outline and naming parameters
    """

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(2, 1, 1)
    ax.scatter(spikeTimes, spikingNeurons, c=color_, marker='.', lw=0)
    ax.set_title("Pyr_population raster")
    ax.set_xlim([0, 10000])
    ax.set_xlabel("Time (ms)")
    ax.set_ylim([0, 4000])
    ax.set_ylabel("Neuron number")

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.bar(hist[1][:-1], hist[0], width=50, align="edge", color=color_, edgecolor='black', lw=0.5, alpha=0.9)
    ax2.axvline(150, ls='--', c="gray", label="ROI for replay analysis")
    ax2.axvline(850, ls='--', c="gray")
    ax2.set_title("Pyr_population ISI distribution")
    ax2.set_xlabel("Delta_t (ms)")
    ax2.set_xlim([0, 1000])
    ax2.set_ylabel("Count")
    ax2.legend()

    fig.tight_layout()

    figName = os.path.join(figFolder, "%s*.png"%(multiplier_))
    fig.savefig(figName)


def plot_PSD(rate, rippleAC, f, Pxx, title_, linespec_, multiplier_):
    """
    saves figure with rate, auto-correlation plot, and PSD
    :param rate: firing rate - precalculated by detect_oscillation.py/preprocess_spikes
    :param rippleAC: auto-correlation function of the rate (returned by detect_oscillation.py/ripple)
    :param f, Pxx (returned by PSD analysis) see more: http://docs.scipy.org/doc/scipy-dev/reference/generated/scipy.signal.welch.html
    :param title_, linespec_, multiplier: outline and naming parameters
    """

    rEACPlot = rippleAC[2:201] # 500 - 5 Hz interval

    f = np.asarray(f)
    rippleS = np.where(145 < f)[0][0]
    rippleE = np.where(f < 250)[0][-1]
    gammaS = np.where(30 < f)[0][0]
    gammaE = np.where(f < 100)[0][-1]
    f.tolist()
    # ripple range
    PxxRipple = Pxx[rippleS:rippleE]
    PxxGamma = Pxx[gammaS:gammaE]
    # gamma range
    fRipple = f[rippleS:rippleE]
    fGamma = f[gammaS:gammaE]

    PxxPlot = 10 * np.log10(Pxx / max(Pxx))
    PxxRipplePlot = 10 * np.log10(PxxRipple / max(Pxx))
    PxxGammaPlot = 10 * np.log10(PxxGamma / max(Pxx))

    fig = plt.figure(figsize=(10, 8))

    ax = fig.add_subplot(3, 1, 1)
    ax.plot(np.linspace(0, 10000, len(rate)), rate, linespec_)
    ax.set_title("%s rate"%title_)
    ax.set_xlabel("Time (ms)")
    ax.set_xlim([0, 10000])

    ax2 = fig.add_subplot(3, 1, 2)
    ax2.plot(np.linspace(2, 200, len(rEACPlot)), rEACPlot, linespec_)
    ax2.set_title("Autocorrelogram 2-200 ms")
    ax2.set_xlabel("Time (ms)")
    ax2.set_xlim([2, 200])
    ax2.set_ylabel("AutoCorrelation")

    ax3 = fig.add_subplot(3, 1, 3)
    ax3.plot(f, PxxPlot, linespec_, marker='o')
    ax3.plot(fRipple, PxxRipplePlot, 'r-', marker='o', linewidth=1.5, label="ripple (145-250Hz)")
    ax3.plot(fGamma, PxxGammaPlot, 'k-', marker='o', linewidth=1.5, label="gamma (30-100Hz)")
    ax3.set_title("Power Spectrum Density")
    ax3.set_xlim([0, 500])
    ax3.set_xlabel("Frequency (Hz)")
    ax3.set_ylabel("PSD (dB)")
    ax3.legend()

    fig.tight_layout()

    figName = os.path.join(figFolder, "%s_%s.png"%(multiplier_, title_))
    fig.savefig(figName)


def plot_zoomed(spikeTimes, spikingNeurons, rate, title_, color_, multiplier_, Pyr_pop=True):
    """
    saves figure with zoomed in raster and rate (last 100ms)
    :param spikeTimes, spikingNeurons: used for raster plot - precalculated by detect_oscillation.py/preprocess_spikes
    :param rate: firing rate - precalculated by detect_oscillation.py/preprocess_spikes
    :param title_, color_, linespec_, multiplier_: outline and naming parameters
    :param Pyr_pop: flag for calculating and returning ymin and ymax (and zooming in the plot)
    :return ymin, ymax for further plotting (see plot_variables)
    """

    # get last 100ms
    ROI = np.where(spikeTimes > 9900)[0].tolist()  # hard coded for 10000ms...
    rasterX = spikeTimes[ROI]
    rasterY = spikingNeurons[ROI]

    if Pyr_pop:
        # boundaries
        if rasterY.min()-50 > 0:
            ymin = rasterY.min()-50
        else:
            ymin = 0
        if rasterY.max()+50 < 4000:
            ymax = rasterY.max()+50
        else:
            ymax = 4000
    else:
        ymin = 0
        ymax = 1000

    fig = plt.figure(figsize=(10, 8))

    ax = fig.add_subplot(2, 1, 1)
    ax.scatter(rasterX, rasterY, c=color_, marker='.', lw=0)
    ax.set_title("%s raster (last 100 ms)"%title_)
    ax.set_xlim([9900, 10000])
    ax.set_xlabel("Time (ms)")
    ax.set_ylim([ymin, ymax])
    ax.set_ylabel("Neuron number")

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(np.linspace(9900, 10000, len(rate[9900:10000])), rate[9900:10000], c=color_, linewidth=1.5)
    ax2.set_title("Rate (last 100 ms)")
    ax2.set_xlabel("Time (ms)")
    ax2.set_xlim([9900, 10000])

    fig.tight_layout()

    figName = os.path.join(figFolder, "%s_%s_zoomed.png"%(multiplier_, title_))
    fig.savefig(figName)

    if Pyr_pop:
        return ymin, ymax


def select_subset(selection, ymin, ymax):
    """
    select a subset of neurons for plotting more detailes (the subset is from the ones spiking in the last 100ms - see plot_zoomed())
    param selection: recorded neurons (ndarray)
    param ymin, ymax: lower and upper bound for the selection
    return subset: selected subset (list)
    """
    try:
        np.random.shuffle(selection)
        subset = []
        counter = 5
        for i in selection:
            if i >= ymin and i <= ymax:
                subset.append(i)
                counter -= 1
            if counter == 0:
                break
    except:  # if there isn't any cell firing
        subset = [400, 1000, 1500, 2300, 3600]
    return subset


def plot_detailed(msM, subset, multiplier_, plot_adaptation=True):
    """
    saves figure with more detailes about some selected neurons
    :param msM: Brian MultiStateMonitor object (could be more elegant...)
    :param subset: selected neurons to plot (max 5)
    :param multiplier_: naming parameter
    :param plot_adaptation: boolean flag for plotting adaptation var.
    """

    fig = plt.figure(figsize=(15, 12))
    #fig.suptitle("Detailed plots of selected vars. (Pyr. pop)")
    ax = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)

    import brian2.monitors.statemonitor
    if type(msM) is brian2.monitors.statemonitor.StateMonitor:
        t = msM.t_ * 1000.
    else:
        t = msM.times*1000  # *1000 ms convertion

    for i in subset:
        ax.plot(t, msM['vm', i]*1000, linewidth=1.5, label="%i"%i)  # *1000 mV conversion
        if plot_adaptation:
            ax2.plot(t, msM['w', i]*1e12, linewidth=1.5, label="%i"%i)  # *1e12 pA conversion
        ax3.plot(t, msM['g_ampa', i], linewidth=1.5, label="%i"%i)
        ax4.plot(t, msM['g_gaba', i], linewidth=1.5, label="%i"%i)

    ax.set_title("Membrane potential (last 100 ms)")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("V (mV)")
    ax.set_xlim([9900, 10000])
    ax.legend()

    ax2.set_title("Adaptation variable (last 100 ms)")
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("w (pA)")
    ax2.set_xlim([9900, 10000])
    if plot_adaptation:
        ax2.legend()

    ax3.set_title("Exc. inputs (last 100 ms)")
    ax3.set_xlabel("Time (ms)")
    ax3.set_ylabel("g_ampa (nS)")
    ax3.set_xlim([9900, 10000])
    ax3.legend()

    ax4.set_title("Inh. inputs (last 100 ms)")
    ax4.set_xlabel("Time (ms)")
    ax4.set_ylabel("g_gaba (nS)")
    ax4.set_xlim([9900, 10000])
    ax4.legend()

    fig.tight_layout()

    figName = os.path.join(figFolder, "%s_Pyr_population_zoomed_detailed.png"%(multiplier_))
    fig.savefig(figName)


def plot_adaptation(msM, subset, multiplier_):  # quick and dirty solution (4 subplots) to use the 40 recorded cell...
    """
    saves figure with the adaptation variables of some selected neurons
    :param msM: Brian MultiStateMonitor object (could be more elegant...)
    :param subset: selected neurons to plot (coded for 40...)
    :param multiplier_: naming parameter
    """

    fig = plt.figure(figsize=(15, 12))
    #fig.suptitle("adaptation variables")
    ax = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)

    import brian2.monitors.statemonitor
    t = msM.times*1000  # *1000 ms convertion
    if type(msM) is brian2.monitors.statemonitor.StateMonitor:
        t = msM.t_ * 1000.
    else:
        t = msM.times*1000  # *1000 ms convertion

    for i in subset:
        if i >= 0 and i < 1000:
            ax.plot(t, msM['w', i]*1e12, linewidth=1.5, label="%i"%i)  # *1e12 pA conversion
        elif i >= 1000 and i < 2000:
            ax2.plot(t, msM['w', i]*1e12, linewidth=1.5, label="%i"%i)  # *1e12 pA conversion
        elif i >= 2000 and i < 3000:
            ax3.plot(t, msM['w', i]*1e12, linewidth=1.5, label="%i"%i)  # *1e12 pA conversion
        else:
            ax4.plot(t, msM['w', i]*1e12, linewidth=1.5, label="%i"%i)  # *1e12 pA conversion

    ax.set_title("Adaptation variables (0-1000)")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("w (pA)")
    ax.set_xlim([0, 10000])
    ax.legend()
    ax2.set_title("Adaptation variables (1000-2000)")
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("w (pA)")
    ax2.set_xlim([0, 10000])
    ax2.legend()
    ax3.set_title("Adaptation variables (2000-3000)")
    ax3.set_xlabel("Time (ms)")
    ax3.set_ylabel("w (pA)")
    ax3.set_xlim([0, 10000])
    ax3.legend()
    ax4.set_title("Adaptation variables (3000-4000)")
    ax4.set_xlabel("Time (ms)")
    ax4.set_ylabel("w (pA)")
    ax4.set_xlim([0, 10000])
    ax4.legend()

    fig.tight_layout()

    figName = os.path.join(figFolder, "%s_adaptation.png"%(multiplier_))
    fig.savefig(figName)


def plot_STDP_rule(taup, taum, Ap, Am, saveName_):
    """
    Plots the STDP rule used for learning
    exponential STDP: f(s) = A_p * exp(-s/tau_p) (if s > 0), where s=tpost_{spike}-tpre_{spike}
    :param taup, taum: time constant of weight change
    :param Ap, Am: max amplitude of weight change
    :return mode: just for saving conventions (see other wmx figures)
    """

    # automate naming
    if Ap == Am:
        mode = "sym"
    elif Ap == Am*-1:
        mode = "asym"
    elif np.abs(Ap) != np.abs(Am):
        print "naming conventions won't work!"
        mode = "tmp"
    print "========== STDP rule: %s =========="%mode

    delta_t = np.linspace(-120, 120, 1000)
    delta_w = np.where(delta_t>0, Ap*np.exp(-delta_t/taup), Am*np.exp(delta_t/taum))

    fig = plt.figure(figsize=(10, 8))

    ax = fig.add_subplot(1, 1, 1)
    ax.plot(delta_t, delta_w, label="STDP rule taup:%s(ms), Ap:%s"%(taup, Ap))
    ax.set_title("STDP curve")
    ax.set_xlabel("delta_t /post-pre/ (ms)")
    ax.set_ylabel("delta_w (nS)")
    if mode == "asym":
        ax.set_ylim([-Ap*1.05, Ap*1.05])
    elif mode == "sym":
        ax.set_ylim([-Ap*0.05, Ap*1.05])
    ax.set_xlim([-100, 100])
    ax.axhline(0, ls='-', c='k')
    ax.legend()

    figName = os.path.join(figFolder, "%s_%s.png"%(saveName_, mode))
    fig.savefig(figName)
    plt.close()

    return mode


def plot_wmx(wmx, saveName_):
    """
    saves figure with the weight matrix
    :param wmx: ndarray representing the weight matrix
    :param saveName_: name of saved img
    """

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)
    i = ax.imshow(wmx, cmap=plt.get_cmap("jet"))
    i.set_interpolation("nearest")  # set to "None" to less pixels and smooth, nicer figure
    fig.colorbar(i)
    ax.set_title("Learned synaptic weights")
    ax.set_xlabel("target neuron")
    ax.set_ylabel("source neuron")

    figName = os.path.join(figFolder, "%s.png"%saveName_)
    fig.savefig(figName)


def plot_wmx_avg(wmx, nPop, saveName_):
    """
    saves figure with the averaged weight matrix (better view as a whole)
    :param wmx: ndarray representing the weight matrix
    :param nPop: number of populations
    :param saveName_: name of saved img
    """

    assert 4000 % nPop == 0

    popSize = int(4000.0 / nPop)
    wmxM = np.zeros((100, 100))
    for i in range(nPop):
        for j in range(nPop):
            tmp = wmx[int(i*popSize):int((i+1)*popSize), int(j*popSize):int((j+1)*popSize)]
            wmxM[i, j] = np.mean(tmp)

    fig = plt.figure(figsize=(10, 8))

    ax = fig.add_subplot(1, 1, 1)
    i = ax.imshow(wmxM, cmap=plt.get_cmap("jet"))
    i.set_interpolation("nearest")  # set to "None" to less pixels and smooth, nicer figure
    fig.colorbar(i)
    ax.set_title("Learned synaptic weights (avg.)")
    ax.set_xlabel("target neuron")
    ax.set_ylabel("source neuron")

    figName = os.path.join(figFolder, "%s.png"%saveName_)
    fig.savefig(figName)


def plot_w_distr(wmx, saveName_):
    """
    saves figure with the distribution of the weights
    :param wmx: ndarray representing the weight matrix
    :param saveName_: name of saved img
    """

    # deleting nulls from wmx to plot the distribution of the weights
    tmp = wmx.tolist()
    wmx = [val for sublist in tmp for val in sublist]
    wmx = filter(lambda i: i != 0, wmx)
    wmx = np.array(wmx)
    log10wmx = np.log10(wmx)
    print "mean(nonzero weights): %s (S)"%np.mean(wmx)

    fig = plt.figure(figsize=(10, 8))

    ax = fig.add_subplot(2, 1, 1)
    ax.hist(wmx, bins=150)
    ax.set_title('Distribution of synaptic weights')
    ax.set_xlabel('pyr-pyr synaptic weight strength (nS)')
    ax.set_ylabel('# of synapses (on logarithmic scale)')
    plt.yscale('log')

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.hist(log10wmx, bins=150, color='red')
    ax2.set_title('Distribution of synaptic weights')
    ax2.set_xlabel('log10(pyr-pyr synaptic weight strength)')
    ax2.set_ylabel('# of synapses (on logarithmic scale)')
    plt.yscale('log')

    fig.tight_layout()

    figName = os.path.join(figFolder, "%s.png"%saveName_)
    fig.savefig(figName)


def save_selected_w(Wee, selection):
    """saves the incomming weights of some selected neurons"""
    w = {}
    for i in selection:
        w[i] = Wee[:, i]
    return w


def plot_weights(dWee, saveName_):
    """
    saves figure with some selected weights
    :param dW: dictionary storing the input weights of some neurons (see save_selected_w())
    :param saveName_: name of saved img
    """

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)
    for i, val in dWee.items():
        ax.plot(val, alpha=0.5, label="%i"%i)

    ax.set_title("Incomming exc. weights")
    ax.set_xlabel("#Neuron")
    ax.set_ylabel("Weight (nS)")
    ax.set_xlim([0, 4000])
    ax.legend()

    figName = os.path.join(figFolder, "%s.png"%saveName_)
    fig.savefig(figName)
    
    
def plot_evolution(ngen, min_fit, mean_fit, std_fit, saveName_):
    """
    saves figure with the evolution of fittnes error (see: optimization/)
    :param ngen: number of generations
    :param min_fit: minimum of fitting errors (see bpop: _,_,log,_ = opt.run())
    :param mean_fit: mean of fitting errors (see bpop: _,_,log,_ = opt.run())
    :param std_fit: standard deviation of fitting errors (see bpop: _,_,log,_ = opt.run())
    :param saveName_: name of saved img
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(ngen, mean_fit, 'k-', linewidth=2, label="pop. average")
    ax.fill_between(ngen, mean_fit - std_fit, mean_fit + std_fit, color='lightgray', linewidth=1.5, label=r"pop. std")
    ax.plot(ngen, min_fit, "r-", linewidth=2, label="pop. minimum")
    ax.set_xlabel("#Generation")
    ax.set_xlim([1, max(ngen)])                                                         
    ax.set_ylabel("Fittnes score")                                                                                
    ax.legend()
    
    figName = os.path.join(figFolder, "%s.png"%saveName_)
    fig.savefig(figName)
    
