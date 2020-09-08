import librosa
import scipy
import wave
import numpy as np

from scipy.signal import spectrogram
import librosa.display
import matplotlib.pyplot as plt
import pywt

#filepath='output/07_05/generated_5.wav'

filepath='C_major.wav'

# x=wave.open(filepath,'rb')
# params = x.getparams()
# nchannels, sampwidth, framerate, nframes = params[:4]
#
# f,t,sxx=spectrogram(x)

y,fs=librosa.load(filepath)

# plt.plot(y)
# plt.title('Signal')
# plt.xlabel('Time (samples)')
# plt.ylabel('Amplitude')
# plt.show()


wavename = 'cgau8'
totalscal = 256
fc = pywt.central_frequency(wavename)
cparam = 2 * fc * totalscal
scales = cparam / np.arange(totalscal, 1, -1)
[cwtmatr, frequencies] = pywt.cwt(y, scales, wavename, 1.0 / fs)

spec = np.abs(librosa.stft(y,n_fft=2048, hop_length=512))
spec = librosa.amplitude_to_db(spec, ref=np.max)
librosa.display.specshow(spec, sr=fs, x_axis='time', y_axis='log')

plt.colorbar(format='%+2.0f dB');
plt.title('Spectrogram');
plt.show()




# mel_spect = librosa.feature.melspectrogram(y=y, sr=fs, n_fft=2048, hop_length=512)
# mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
# librosa.display.specshow(mel_spect, y_axis='mel', fmax=8000, x_axis='time');
# plt.title('Mel Spectrogram');
# plt.colorbar(format='%+2.0f dB');
# plt.show()
