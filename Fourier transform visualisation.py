import wave
import numpy as np
from scipy.fft import *
import matplotlib.pyplot as plt

wav_obj = wave.open('#24.wav', 'rb')
sample_freq = wav_obj.getframerate()
n_samples = wav_obj.getnframes()
t_audio = n_samples/sample_freq
n_channels = wav_obj.getnchannels()

signal_wave = wav_obj.readframes(n_samples)
signal_array = np.frombuffer(signal_wave, dtype=np.int16)

l_channel = signal_array[0::2]
r_channel = signal_array[1::2]

times = np.linspace(0, n_samples/sample_freq, num=n_samples)

ff_transform = fft(l_channel)
inverse_ff_transform = ifft(ff_transform)

y_noise = np.copy(ff_transform)
y_noise[y_noise > 30000000] = np.nan
y_noise[y_noise < -30000000] = np.nan

x_noise = np.copy(times)
x_noise[x_noise < 1] = np.nan
x_noise[x_noise > 89] = np.nan

plt.plot(times, ff_transform, 'r')
plt.plot(x_noise, ff_transform, 'c')
plt.plot(times, y_noise, 'r')

plt.legend(['noise freq', 'occuring freq'])

plt.xlabel('frequency')
plt.xlabel('time')
plt.show()
