import librosa
import scipy
import wave
import numpy as np

from scipy.signal import spectrogram
import librosa.display
import matplotlib.pyplot as plt
import pywt

#filepath='output/07_05/generated_5.wav'

filepath='C6_fl.wav'

y,fs=librosa.load(filepath)

ffts=[2048]

for fft in ffts:
    spec = np.abs(librosa.stft(y, fft))
    spec = librosa.amplitude_to_db(spec, ref=np.max)
    librosa.display.specshow(spec, sr=fs, x_axis='time', y_axis='log')

    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram (fft length = '+str(fft)+')')
    plt.ylim(256,4096)
    plt.show()

# spec = np.abs(librosa.stft(y,n_fft=256))
# spec = librosa.amplitude_to_db(spec, ref=np.max)
# librosa.display.specshow(spec, sr=fs, x_axis='time', y_axis='log')
# plt.colorbar(format='%+2.0f dB')
# plt.title('Spectrogram (fft length = 256)')
# plt.show()
#
# spec2 = np.abs(librosa.stft(y,n_fft=512))
# spec2 = librosa.amplitude_to_db(spec2, ref=np.max)
# librosa.display.specshow(spec2, sr=fs, x_axis='time', y_axis='log')
# plt.colorbar(format='%+2.0f dB')
# plt.title('Spectrogram (fft length = 512)')
# plt.show()
#
# spec3 = np.abs(librosa.stft(y,n_fft=1024))
# spec3 = librosa.amplitude_to_db(spec3, ref=np.max)
# librosa.display.specshow(spec3, sr=fs, x_axis='time', y_axis='log')
# plt.colorbar(format='%+2.0f dB')
# plt.title('Spectrogram (fft length = 1024)')
# plt.show()
#
# spec4 = np.abs(librosa.stft(y,n_fft=2048))
# spec4 = librosa.amplitude_to_db(spec4, ref=np.max)
# librosa.display.specshow(spec4, sr=fs, x_axis='time', y_axis='log')
# plt.colorbar(format='%+2.0f dB')
# plt.title('Spectrogram (fft length = 2048)')
# plt.show()
#
# spec5 = np.abs(librosa.stft(y,n_fft=4096))
# spec5 = librosa.amplitude_to_db(spec5, ref=np.max)
# librosa.display.specshow(spec5, sr=fs, x_axis='time', y_axis='log')
# plt.colorbar(format='%+2.0f dB')
# plt.title('Spectrogram (fft length = 4096)')
# plt.show()
