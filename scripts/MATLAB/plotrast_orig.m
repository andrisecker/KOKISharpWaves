clear all
close all

datin=importdata('PE_orig_1_0.dat');
cell=datin(:,1);
spt=datin(:,2);

plot(spt,cell,'.')
