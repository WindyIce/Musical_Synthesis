
clc
 
clear all;
 

filename='C4_fl.wav';
 
[y,fs]=audioread(filename,[1,44100]);

t=linspace(0,length(y)/fs,length(y));
 
figure
 
plot(t, y)
 
% 连续小波变换时频图
 
wavename='cmor3-3';
 
totalscal=2048;
 
Fc=centfrq(wavename); % 小波的中心频率
 
c=2*Fc*totalscal;
 
scals=c./(1:totalscal);
 
f=scal2frq(scals,wavename,1/fs); % 将尺度转换为频率
 
coefs=cwt(y,scals,wavename); % 求连续小波系数
 
figure
 
imagesc(t,f,abs(coefs));
 
set(gca,'YDir','normal')
 
colorbar;

set(gca,'ColorScale','log')
 
xlabel('时间 t/s');
 
ylabel('频率 f/Hz');

ylim([0,2000])
 
title('小波时频图');
 
% 短时傅里叶变换时频图
 
figure
 
spectrogram(y,2048,1536,2048,fs);

set(gca,'YDir','normal')

figure

stft(y,fs,'Window',hamming(2048,'periodic'),'OverlapLength',1536,'FFTLength',2048);
ylim([0,2])



 
% 时频分析工具箱里的短时傅里叶变换
 
f = 0:fs/2;
 
tfr = stft(y');
%tfr = tfr(1:floor(length(fs)/2), :);
figure
imagesc(t, f, abs(tfr));
set(gca,'YDir','normal')
colorbar;
xlabel('时间 t/s');
ylabel('频率 f/Hz');
title('短时傅里叶变换时频图');