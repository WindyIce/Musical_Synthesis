
clc
 
clear all;
 

filename='C4_fl.wav';
 
[y,fs]=audioread(filename,[1,44100]);

t=linspace(0,length(y)/fs,length(y));
 
figure
 
plot(t, y)
 
% ����С���任ʱƵͼ
 
wavename='cmor3-3';
 
totalscal=2048;
 
Fc=centfrq(wavename); % С��������Ƶ��
 
c=2*Fc*totalscal;
 
scals=c./(1:totalscal);
 
f=scal2frq(scals,wavename,1/fs); % ���߶�ת��ΪƵ��
 
coefs=cwt(y,scals,wavename); % ������С��ϵ��
 
figure
 
imagesc(t,f,abs(coefs));
 
set(gca,'YDir','normal')
 
colorbar;

set(gca,'ColorScale','log')
 
xlabel('ʱ�� t/s');
 
ylabel('Ƶ�� f/Hz');

ylim([0,2000])
 
title('С��ʱƵͼ');
 
% ��ʱ����Ҷ�任ʱƵͼ
 
figure
 
spectrogram(y,2048,1536,2048,fs);

set(gca,'YDir','normal')

figure

stft(y,fs,'Window',hamming(2048,'periodic'),'OverlapLength',1536,'FFTLength',2048);
ylim([0,2])



 
% ʱƵ������������Ķ�ʱ����Ҷ�任
 
f = 0:fs/2;
 
tfr = stft(y');
%tfr = tfr(1:floor(length(fs)/2), :);
figure
imagesc(t, f, abs(tfr));
set(gca,'YDir','normal')
colorbar;
xlabel('ʱ�� t/s');
ylabel('Ƶ�� f/Hz');
title('��ʱ����Ҷ�任ʱƵͼ');