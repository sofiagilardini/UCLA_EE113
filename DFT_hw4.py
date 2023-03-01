import matplotlib.pyplot as plt
import numpy as np 

from dsp_toolbox import dft, apply_dft_matrix, create_dft_matrix, plot_dft_magnitude_angle, idft, convolve_signals, zero_pad_signal
from utils import visualize_dft_matrix, generate_gaussian_kernel

N = 32 # number of samples in the signal 
t_indices = np.arange(0,N)

print(t_indices) # time series indices go include 0 and N-1

# frequency of the signal 
f = 2/N

y_signal = np.sin(2*np.pi*f*t_indices)

size_of_T_array = t_indices.shape
size_of_Y_array = y_signal.shape
print(f"Size of T array is: {size_of_T_array}")
print(f"Size of Y array is: {size_of_Y_array}")
print(f"Size of 4th item in y_signal is: {y_signal[3]}")
print(f"Values of 2nd through 5th items in y_signal are: {y_signal[1:6]}")

plt.plot(t_indices, y_signal, 'o')
plt.title("Signal")
plt.xlabel("Sample number (n)")
plt.ylabel("Signal amplitude")
plt.show()