clear all
close all

datin=importdata('wmx.txt');

bdv=zeros(50,50*80*80);
bdv_nz=cell(1,50);
num_nz=zeros(1,50);
bsum=zeros(50);

for an=1:50,
    for bn=1:50,
        a=datin(80*(an-1)+1:80*an,80*(bn-1)+1:80*bn);
        bsum(an,bn)=sum(a(:));
        
        if bn<52-an
            b=datin(80*(bn-1)+1:80*bn,80*(bn+an-2)+1:80*(bn+an-1));
        else
            b=datin(80*(bn-1)+1:80*bn,80*(bn+an-2)+1-4000:80*(bn+an-1)-4000);
        end
        bdv(an,80*80*(bn-1)+1:80*80*bn)=b(:);
    end
    bdnzi=(bdv(an,:)~=0);
    bdv_nz{an}=bdv(an,bdnzi);
    num_nz(an)=length(bdv_nz{an});
end

for an=1:5,
    figure(1)
    subplot(3,3,an+4)
    hist(bdv(an,:),1000)
    
    figure(2)
    subplot(3,3,an+4)
    h0=histc(bdv_nz{an},0:1e-10:1.5e-8);
    bar(0:1e-10:1.5e-8,h0,'histc')
    % axis([0 1.5e-8 0 2000])
    title(strcat('dn = ',int2str(an-1)))
end

for an=47:50,
    figure(1)
    subplot(3,3,an-46)
    hist(bdv(an,:),1000)
    
    figure(2)
    subplot(3,3,an-46)
    h0=histc(bdv_nz{an},0:1e-10:1.5e-8);
    bar(0:1e-10:1.5e-8,h0,'histc')
    % axis([0 1.5e-8 0 2000])
    title(strcat('dn = ',int2str(an-51)))
end

bdsum=sum(bdv,2);
bdsum_c=[bdsum(27:50,1);bdsum(1:26,1)];
figure
plot(-24:25,bdsum_c)
xlabel('Block distance')
ylabel('Sum of all weights')

num_nz_c=[num_nz(27:50),num_nz(1:26)];
figure
plot(-24:25,num_nz_c)
xlabel('Block distance')
ylabel('Number of non-zero weights')

figure
imagesc(bsum)
title('Sum of weights in each block')

b1_1=datin(1:80,1:80);
figure
surf(b1_1)
title('Weights in block (1,1)')

figure
imagesc(b1_1)
title('Weights in block (1,1)')

dd=diag(datin);
figure
hist(dd,100)
title('Distribution of self-weights')
