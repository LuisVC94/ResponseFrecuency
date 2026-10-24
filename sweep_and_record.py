
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt

from scipy.signal import chirp, fftconvolve
from pathlib import Path

# -----------------------------
# CONFIGURACION
# -----------------------------
N_SPEAKERS         = 2
N_MICS             = 2

INPUT_N_CHANNELS   = 2
INPUT_DEVICE_NAME  = "IN 01-02"
OUTPUT_N_CHANNELS  = 2
OUTPUT_DEVICE_NAME = "OUT 01-02"

FS = 96000
DURATION = 15.0
F_START = 15
F_END = 30000

GAIN = 0.5            # Start volume
PRE_SILENCE = 1.0     # Silence before sweep
POST_SILENCE = 2.0    # Silence after sweep

# Open path to save measurements ----------------------------
OUTPUT_DIR = Path("measurements")
OUTPUT_DIR.mkdir(exist_ok=True)
# -----------------------------------------------------------

# Buscar dispositivos por nombre y API ----------------------
def find_device(prefix, api="Windows WASAPI"):
    for i, d in enumerate(sd.query_devices()):
        host = sd.query_hostapis(d["hostapi"])["name"]
        if d["name"].upper().startswith(prefix.upper()) and host == api:
            return i
    raise RuntimeError(f"No se encontro {prefix} en {api}")


INPUT_DEVICE = find_device(INPUT_DEVICE_NAME)
OUTPUT_DEVICE = find_device(OUTPUT_DEVICE_NAME)
print("Entrada:", INPUT_DEVICE, sd.query_devices(INPUT_DEVICE)["name"])
print("Salida:", OUTPUT_DEVICE, sd.query_devices(OUTPUT_DEVICE)["name"])

sd.check_input_settings(device=INPUT_DEVICE, channels=INPUT_N_CHANNELS, samplerate=FS)
sd.check_output_settings(device=OUTPUT_DEVICE, channels=OUTPUT_N_CHANNELS, samplerate=FS)
# -----------------------------------------------------------



# Logaritmic sweep ----------------------------------------
N = int(FS * DURATION)
t = np.arange(N) / FS

sweep = chirp(
    t,
    f0=F_START,
    f1=F_END,
    t1=DURATION,
    method="logarithmic"
).astype(np.float64)

# Fades at start and end to avoid clicks ----------------
fade_n = int(0.05 * FS)
fade = np.sin(np.linspace(0, np.pi / 2, fade_n)) ** 2

sweep[:fade_n] *= fade
sweep[-fade_n:] *= fade[::-1]
# -----------------------------------------------------------

# Sweep out signal ------------------------------------------
pre_n = int(FS * PRE_SILENCE)
post_n = int(FS * POST_SILENCE)

test_signal = np.concatenate([
    np.zeros(pre_n),
    sweep * GAIN,
    np.zeros(post_n)
])

# -----------------------------------------------------------

# Repeat in each speaker and record from both microphones ---
output = np.zeros((len(test_signal), N_SPEAKERS), dtype=np.float32)

for speaker in range(N_SPEAKERS):
    output.fill(0)
    output[:, speaker] = test_signal
    
    np.save(
        OUTPUT_DIR / f"Base_signal_speaker_{speaker+1}.npy",
        output
    )

    print(f"\nTesting Speaker:  {speaker + 1}...")

    recording = sd.playrec(
        output,
        samplerate=FS,
        channels=N_MICS,
        device=(INPUT_DEVICE, OUTPUT_DEVICE),
        dtype="float32",
        blocking=True,
        latency="low"
    )

    # Save microphone recordings ---------
    np.save(
        OUTPUT_DIR / f"Recording_speaker_{speaker+1}.npy",
        recording
    )

    print("Test Speaker:", speaker+1, "\nMAX Mic 1:", np.max(np.abs(recording[:, 0])), "\nMAX Mic 2:", np.max(np.abs(recording[:, 1])))



# Plot microphone recordings loaded from the saved files ------
recordings = [
    np.load(OUTPUT_DIR / f"Recording_speaker_{speaker+1}.npy")
    for speaker in range(2)
]
num_mics = recordings[0].shape[1]

fig, axes = plt.subplots(
    2, num_mics+1,
    figsize=(6 * num_mics, 7),
    sharex=True,
    sharey=True,
    squeeze=False,
    layout="constrained"
)
fig.suptitle("Grabaciones de los microfonos por bocina")

for speaker, saved_recording in enumerate(recordings):
    recording_time = np.arange(saved_recording.shape[0]) / FS

    for mic in range(num_mics+1):
        ax = axes[speaker, mic]
        if mic == num_mics:
            ax.plot(recording_time, output[:, N_SPEAKERS-1], linewidth=0.5)
            ax.set_title(f"Señal Base {mic+1}")
        else:
            ax.plot(recording_time, saved_recording[:, mic], linewidth=0.5)
            ax.set_title(f"Bocina {speaker+1} -> Microfono {mic+1}")
        ax.grid(True, alpha=0.3)

        if speaker == 1:
            ax.set_xlabel("Tiempo (s)")
        if mic == 0:
            ax.set_ylabel("Amplitud")

plt.show()
