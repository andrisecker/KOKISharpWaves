fOut = 'spikeTrain_5_8_Matlab.txt'
f = fopen(fOut, 'w');

nPop = 5;
nNeuron = 8;

spikeTrains = cell(nPop*nNeuron, 1);

if nNeuron ~= 1
    seed = 0;
    for pop = 1:nPop
        for neuron = 1:nNeuron
            spikeTrains{(pop-1)*nNeuron + neuron}= inhom_poisson(0, nPop, pop, seed);
            fprintf(f, '%f\t', spikeTrains{(pop-1)*nNeuron + neuron});
            fprintf(f, '\n');
            seed = seed + 1;
            [pop neuron]
        end
    end
else
    for pop = 1:nPop
        spikeTrains{pop} = inhom_poisson(0, nPop, pop, seed);
        fprintf(f, '%f\t', spikeTrains{pop});
        fprintf(f, '\n');
        [pop]
    end
end

fclose(f);
