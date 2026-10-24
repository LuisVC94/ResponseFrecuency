import numpy as np
import matplotlib.pyplot as plt

FS = 96000

gains = np.load("measurements/fft_gain.npy")

# Dimensiones: frecuencia, micrófono, bocina
H = np.empty((gains.shape[1], 2, 2), dtype=complex)

H[:, 0, 0] = gains[0]
H[:, 1, 0] = gains[1]
H[:, 0, 1] = gains[2]
H[:, 1, 1] = gains[3]

def regularized_inverse(H, lam):
    # Admite una matriz o un conjunto con forma (frecuencias, 2, 2).
    Hh = H.conj().swapaxes(-1, -2)
    I = np.eye(H.shape[-1])

    return np.linalg.solve(Hh @ H + lam * I, Hh)

n_samples = np.load("measurements/Base_signal_speaker_1.npy").shape[0]
frequencies = np.fft.rfftfreq(n_samples, d=1 / FS)
if frequencies.size != H.shape[0]:
    raise ValueError("Las FFT y la senal base tienen longitudes incompatibles.")
valid = ((frequencies >= 15) & (frequencies <= 25000))

alpha = 1e-3

singular_values = np.linalg.svd(H[valid], compute_uv=False)
sigma_ref = singular_values[:, 0].max()

lam = alpha * sigma_ref**2
if not np.isfinite(lam) or lam <= 0:
    raise ValueError("La banda seleccionada debe contener respuestas finitas y no nulas.")

H_i = np.zeros_like(H)
H_i[valid] = regularized_inverse(H[valid], lam)

R = H @ H_i

fig, axes = plt.subplots(
    2, 2,
    figsize=(6 * 4, 10),
    sharex=True,
    sharey=False,
    squeeze=False,
    layout="constrained"
)
fig.suptitle("FFTs")

axes[0, 0].plot(frequencies, np.abs(R[:, 0, 0]), color='red')
axes[0, 0].set_title('Hi_0_0')
axes[0, 1].plot(frequencies, np.abs(R[:, 0, 1]), color='red')
axes[0, 1].set_title('Hi_0_1')
axes[1, 0].plot(frequencies, np.abs(R[:, 1, 0]), color='red')
axes[1, 0].set_title('Hi_1_0')
axes[1, 1].plot(frequencies, np.abs(R[:, 1, 1]), color='red')
axes[1, 1].set_title('Hi_1_1')

plt.show()
