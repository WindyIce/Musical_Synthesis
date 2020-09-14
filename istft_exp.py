import librosa
import scipy
import wave
import numpy as np

from scipy.signal import spectrogram
import librosa.display
import matplotlib.pyplot as plt
import pywt
import scipy.signal as signal

#filepath='output/07_05/generated_5.wav'

def inv_magphase(mag, phase_angle):
  phase = np.cos(phase_angle) + 1.j * np.sin(phase_angle)
  return mag * phase

def griffin_lim(mag, phase_angle, n_fft, hop, num_iters):

  global audio
  fft_config = dict(n_fft=n_fft, win_length=n_fft, hop_length=hop, center=True)
  ifft_config = dict(win_length=n_fft, hop_length=hop, center=True)
  complex_specgram = inv_magphase(mag, phase_angle)
  for i in range(num_iters):
    audio = librosa.istft(complex_specgram, **ifft_config)
    if i != num_iters - 1:
      complex_specgram = librosa.stft(audio, **fft_config)
      _, phase = librosa.magphase(complex_specgram)

      phase_angle = np.angle(phase)
      complex_specgram = inv_magphase(mag, phase_angle)
  return audio

def GLA(S, n_iter = 100, n_fft = 2048, hop_length = None, window = 'hann'):
    hop_length = n_fft // 4 if hop_length is None else hop_length
    phase = np.exp(2j * np.pi * np.random.rand(*S.shape))
    for i in range(n_iter):
        xi = np.abs(S).astype(np.complex) * phase
        signal = librosa.istft(xi, hop_length=hop_length, window=window)
        next_xi = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length, window=window)
        phase = np.exp(1j * np.angle(next_xi))
    xi = np.abs(S).astype(np.complex) * phase
    signal = librosa.istft(xi, hop_length=hop_length, window=window)
    return signal

filepath='C6_fl.wav'

y,fs=librosa.load(filepath)

fft=2048

spec = abs(librosa.stft(y, fft))
angles = np.exp(2j * np.pi * np.random.rand(*spec.shape))

#audio=griffin_lim(spec,angles,2048,512,1000)
audio=GLA(spec,n_iter = 1000, n_fft = 2048)

plt.plot(y)
plt.title('Signal')
plt.xlabel('Time (samples)')
plt.ylabel('Amplitude')
plt.show()

plt.plot(audio)
plt.title('Signal After Reconstruction')
plt.xlabel('Time (samples)')
plt.ylabel('Amplitude')
plt.show()

f, Cxy = signal.coherence(y, audio, fs,nfft=2048, nperseg=2048,noverlap=int(2048*0.75))
plt.semilogy(f, Cxy)
plt.xlabel('frequency [Hz]')
plt.ylabel('Coherence')
plt.show()


