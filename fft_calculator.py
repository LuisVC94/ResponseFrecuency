import numpy as np
import matplotlib.pyplot as plt

FILE_SAVE_NAME_FFT = "measurements/FFT_speaker_2.npy"

N_SPEAKERS = 2
N_MICS     = 2

FS = 96000
IMPULSE_RESPONSE_SAVE_LEN = int(0.5*FS)

original_signal = np.load("measurements/Base_signal_speaker_1.npy")
original_signal = original_signal[:,0]

recorded_signals = list()
fft_recorded_signals = list()
fft_recorded_signals_valid = list()
fft_gain = list()
fft_gain_valid = list()

fft_original = np.fft.rfft(original_signal)

for speaker in range(N_SPEAKERS):
    recorded_signals.append(np.load(f"measurements/Recording_speaker_{speaker+1}.npy"))
    for mic in range(N_MICS):
        fft_recorded_signals.append(np.fft.rfft(recorded_signals[speaker][:,mic]))
        fft_gain.append(fft_recorded_signals[speaker*2+mic]/fft_original)
        fft_gain_valid.append(np.zeros_like(fft_original))
        fft_recorded_signals_valid.append(np.zeros_like(fft_original))


signal_len = original_signal.shape[0]

frequencies = np.fft.rfftfreq(signal_len, d=1 / FS)

# Get valid index of fft division and create valid arrays ---

valid = ((frequencies >= 15) & (frequencies <= 25000))
for i in range(len(fft_gain)):
    fft_gain_valid[i][valid] = fft_gain[i][valid]
    fft_recorded_signals_valid[i][valid] = fft_recorded_signals[i][valid]

np.save("measurements/fft_original.npy", fft_original)
np.save("measurements/fft_recorded_signals.npy", fft_recorded_signals_valid)
np.save("measurements/fft_gain.npy", fft_gain_valid)

fig, axes = plt.subplots(
    N_SPEAKERS, N_MICS*2 + 1,
    figsize=(6 * 4, 10),
    sharex=True,
    sharey=False,
    squeeze=False,
    layout="constrained"
)
fig.suptitle("FFTs")

for speaker in range(N_SPEAKERS):
    axes[speaker, 0].plot(frequencies, np.abs(fft_original), color='red')
    axes[speaker, 0].set_title('Original')
    for mic in range(1, N_MICS+1):
        axes[speaker, mic].plot(frequencies, np.abs(fft_recorded_signals_valid[speaker*2+mic-1]), color='red')
        axes[speaker, mic].set_title(f'Recorded SP {speaker}, Mic {mic-1}')
    for mic in range(3, N_MICS+3):
        axes[speaker, mic].plot(frequencies, np.abs(fft_gain_valid[speaker*2+mic-3]), color='red')
        axes[speaker, mic].set_title(f'Gain SP {speaker}, Mic {mic-3}')

plt.show()