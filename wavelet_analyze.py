import matplotlib.pyplot as plt
import numpy as np
import pywt
import librosa


#filepath='output/07_05/generated_5.wav'
filepath='C_major.wav'
y,fs=librosa.load(filepath)

t = np.arange(0, len(y)/float(fs), 1.0 / fs)

wavename = 'cgau8'
totalscal = 256
fc = pywt.central_frequency(wavename)
cparam = 2 * fc * totalscal
scales = cparam / np.arange(totalscal, 1, -1)
[cwtmatr, frequencies] = pywt.cwt(y, scales, wavename, 1.0 / fs)
plt.figure(figsize=(8, 4))
plt.subplot(211)
plt.plot(t, y)
plt.xlabel("time (sec)")
plt.title("Spectrogram")
plt.subplot(212)
plt.contourf(t, frequencies, abs(cwtmatr))
plt.ylabel("frequency (Hz)")
plt.ylim(0,1000)
plt.xlabel("time (sec)")
plt.subplots_adjust(hspace=0.4)

plt.show()