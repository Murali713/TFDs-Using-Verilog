import numpy as np
import matplotlib.pyplot as plt

def wavelet_function(A, w, l, t, sigma, f0):
    """
    Compute the wavelet function defined by the equation:

    psi(A, w, l; t) = A * (1 / np.sqrt(sigma)) * np.exp(-w * (t - l)**2 / 2) * np.exp(1j * 2 * np.pi * f0 * (t - l))

    Parameters:
        A (float): Amplitude parameter.
        w (float): Width-scaling factor.
        l (float): Translation parameter.
        t (numpy array): Array of time values where the function is evaluated.
        sigma (float): Standard deviation of the Gaussian window.
        f0 (float): Central frequency of the wavelet.

    Returns:
        numpy array: The computed values of the wavelet function at each time point t.
    """
    return A * (1 / np.sqrt(sigma)) * np.exp(-w * (t - l)**2 / 2) * np.exp(1j * 2 * np.pi * f0 * (t - l))

def compute_tfd(signal, time, frequencies, A, w, l, sigma, f0):
    """
    Compute the Time-Frequency Distribution (TFD) using the Short-Time Fourier Transform (STFT).

    Parameters:
        signal (numpy array): The input time-domain signal.
        time (numpy array): Array of time values corresponding to the signal.
        frequencies (numpy array): Array of frequency values where the TFD is computed.
        A (float): Amplitude parameter for the wavelet function.
        w (float): Width-scaling factor for the wavelet function.
        l (float): Translation parameter for the wavelet function.
        sigma (float): Standard deviation of the Gaussian window for the wavelet function.
        f0 (float): Central frequency of the wavelet function.

    Returns:
        numpy array: The computed Time-Frequency Distribution (TFD).
    """
    tfd = np.zeros((len(frequencies), len(time)), dtype=np.complex128)
    for i, freq in enumerate(frequencies):
        psi_freq = wavelet_function(A, w, l, time, sigma, freq)
        for j in range(len(time)):
            segment = signal[j:j+len(psi_freq)]
            if len(segment) < len(psi_freq):
                break
            tfd[i, j] = np.sum(segment * np.conj(psi_freq))

    return tfd

# Example usage:
# Replace these with your actual signal and time values
signal = np.sin(2 * np.pi * np.linspace(0, 1, 100))  
time = np.linspace(0, 1, 100)

# Define the frequency range for the TFD
frequencies = np.linspace(0, 10, 100)

# Define wavelet parameters
A = 1.0
w = 1.0
l = 0.5
sigma = 0.1
f0 = 1.0

# Compute the TFD
tfd = compute_tfd(signal, time, frequencies, A, w, l, sigma, f0)

# Plot the TFD
plt.imshow(np.abs(tfd), aspect='auto', origin='lower', extent=[0, 1, 0, 10], cmap='viridis')
plt.colorbar(label='Magnitude')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.title('Time-Frequency Distribution (TFD)')

# Incorporating newton method

import numpy as np
import matplotlib.pyplot as plt

def wavelet_function(A, w, l, t, sigma, f0):
    return A * (1 / np.sqrt(sigma)) * np.exp(-w * (t - l)**2 / 2) * np.exp(1j * 2 * np.pi * f0 * (t - l))

def compute_tfd(signal, time, frequencies, A, w, l, sigma, f0):
    tfd = np.zeros((len(frequencies), len(time)), dtype=np.complex128)
    for i, freq in enumerate(frequencies):
        psi_freq = wavelet_function(A, w, l, time, sigma, freq)
        for j in range(len(time)):
            segment = signal[j:j+len(psi_freq)]
            if len(segment) < len(psi_freq):
                break
            tfd[i, j] = np.sum(segment * np.conj(psi_freq))

    return tfd

def calculate_error(signal, time, frequencies, A, w, l, sigma, f0):
    tfd = compute_tfd(signal, time, frequencies, A, w, l, sigma, f0)
    signal_tfd = np.sum(np.abs(tfd), axis=0)
    return np.sum(np.abs(signal - signal_tfd)**2)

def newton_method(signal, time, frequencies, A, w, l, sigma, f0, max_iterations=100, tolerance=1e-6):
    for i in range(max_iterations):
        tfd = compute_tfd(signal, time, frequencies, A, w, l, sigma, f0)
        gradient_A = np.sum(np.real(tfd) * np.real(signal - np.sum(np.abs(tfd)**2, axis=0)))
        gradient_w = np.sum(np.real(tfd) * np.real(signal - np.sum(np.abs(tfd)**2, axis=0)) * (time - l)**2)
        gradient_l = np.sum(np.real(tfd) * np.real(signal - np.sum(np.abs(tfd)**2, axis=0)) * w * (time - l))
        gradient_sigma = np.sum(np.real(tfd) * np.real(signal - np.sum(np.abs(tfd)**2, axis=0)) * (time - l)**2 / (2 * sigma))
        gradient_f0 = np.sum(np.imag(tfd) * np.real(signal - np.sum(np.abs(tfd)**2, axis=0)) * 2 * np.pi * (time - l))
        
        gradient = np.array([gradient_A, gradient_w, gradient_l, gradient_sigma, gradient_f0])
        hessian_A = np.sum(np.real(tfd) * np.real(np.sum(np.abs(tfd)**2, axis=0)))
        hessian_w = np.sum(np.real(tfd) * np.real(np.sum(np.abs(tfd)**2, axis=0)) * (time - l)**2)
        hessian_l = np.sum(np.real(tfd) * np.real(np.sum(np.abs(tfd)**2, axis=0)) * w * (time - l))
        hessian_sigma = np.sum(np.real(tfd) * np.real(np.sum(np.abs(tfd)**2, axis=0)) * (time - l)**2 / (2 * sigma**2))
        hessian_f0 = np.sum(np.imag(tfd) * np.real(np.sum(np.abs(tfd)**2, axis=0)) * 2 * np.pi * (time - l))

        hessian = np.array([[hessian_A, hessian_w, hessian_l, hessian_sigma, hessian_f0]])
        new_params = np.array([A, w, l, sigma, f0]) - np.linalg.solve(hessian, gradient)
        if np.allclose(new_params, [A, w, l, sigma, f0], rtol=tolerance):
            break
        A, w, l, sigma, f0 = new_params

    return A, w, l, sigma, f0

# Example usage:
# Replace these with your actual signal and time values
signal = np.sin(2 * np.pi * np.linspace(0, 1, 100))  
time = np.linspace(0, 1, 100)

# Define the frequency range for the TFD
frequencies = np.linspace(0, 10, 100)

# Initial wavelet parameters
A = 1.0
w = 1.0
l = 0.5
sigma = 0.1
f0 = 1.0

# Optimize wavelet parameters using Newton method
A, w, l, sigma, f0 = newton_method(signal, time, frequencies, A, w, l, sigma, f0)

# Compute the TFD with optimized parameters
tfd = compute_tfd(signal, time, frequencies, A, w, l, sigma, f0)

# Plot the TFD
plt.imshow(np.abs(tfd), aspect='auto', origin='lower', extent=[0, 1, 0, 10], cmap='viridis')
plt.colorbar(label='Magnitude')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.title('Time-Frequency Distribution (TFD)')
plt.show()

print("Optimized Wavelet Parameters:")
print("A:", A)
print("w:", w)
print("l:", l)
print("sigma:", sigma)
print("f0:", f0)



plt.show()
