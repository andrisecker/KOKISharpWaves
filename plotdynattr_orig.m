clear all
close all

weight = [0.7 0.8 0.9 1.0 1.1 1.2 1.5 2.0 2.5];
re = [0.05 0.85 1.36 1.91 2.74 3.65 6.5 9.6 13.4];
ri = [1.9 172 285 411 589 772 1315 2000 2900];
ace = [0.01 0.20 0.33 0.63 0.73 0.70 0.74 0.48 0.59];
aci = [0.25 0.47 0.66 0.82 0.84 0.79 0.84 0.86 0.81];

figure(1)
subplot(2,1,1)
[ax1,h11,h12]=plotyy(weight,re,weight,ri);
set(gca,'FontSize',16)
set(get(ax1(1),'Ylabel'),'String','PC rate (Hz)','FontSize',14)
set(get(ax1(2),'Ylabel'),'String','BC rate (Hz)','FontSize',14)
set(ax1(1),'YColor','k','FontSize',14)
set(ax1(2),'YColor','r','FontSize',14)
set(h11,'Color','k')
set(h12,'Color','r')
subplot(2,1,2)
[ax2,h21,h22]=plotyy(weight,ace,weight,aci);
set(ax2,'YLim',[0 1],'YTick',0:0.2:1)
set(get(ax2(1),'Ylabel'),'String','PC ripple coherence','FontSize',14)
set(get(ax2(2),'Ylabel'),'String','BC ripple coherence','FontSize',14)
set(ax2(1),'YColor','k','FontSize',14)
set(ax2(2),'YColor','r','FontSize',14)
set(h21,'Color','k')
set(h22,'Color','r')
xlabel('Pyr-Pyr weight scaling')