import librosa
import scipy
import wave
import numpy as np
from sklearn.preprocessing import minmax_scale

from scipy.signal import spectrogram
import librosa.display
import matplotlib.pyplot as plt
import pywt

#filepath='output/07_05/generated_5.wav'

filepath=[
'C1_fl.wav',
'C4_fl.wav',
'C6_fl.wav',
'generated_0.wav',
'generated_2.wav',
'generated_3.wav',
'generated_4.wav',
'acou/generated_0.wav',
'acou/generated_1.wav',
'acou/generated_2.wav',
'acou/generated_3.wav'
]
#
# for file in filepath:
#     y,fs=librosa.load(file)
#
#     zero_crossing = librosa.zero_crossings(y, pad=False)
#     print(sum(zero_crossing))
#

file=filepath[6]
y,fs=librosa.load(file)


chromagram = librosa.feature.chroma_stft(y,sr=fs)
librosa.display.specshow(chromagram,x_axis='time',y_axis='chroma',
                         cmap='coolwarm')
plt.show()


# spectral_centroids=librosa.feature.spectral_centroid(y,sr=fs)[0]
# print(spectral_centroids.shape)
# frames=range(len(spectral_centroids))
# t=librosa.frames_to_time((frames))
# def normalize(x,axis=0):
#     return minmax_scale(x,axis=axis)
#
# librosa.display.waveplot(y,sr=fs)
# plt.plot(t,normalize(spectral_centroids),color='r')
#
# plt.ylabel('Normalized Spectral Centroid')
# plt.show()



# fft=2048
#
# spec = np.abs(librosa.stft(y, fft))
# spec = librosa.amplitude_to_db(spec, ref=np.max)
# librosa.display.specshow(spec, sr=fs, x_axis='time', y_axis='log')
#
# plt.colorbar(format='%+2.0f dB')
# plt.title('Spectrogram (fft length = ' + str(fft) + ')')
# plt.ylim(256, 4096)
# plt.show()