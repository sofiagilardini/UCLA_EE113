import numpy as np
import matplotlib.pyplot as plt 

def dft(signal : np.array) -> np.array:
    """Returns the Discrete Fourier Transform of a signal

    Args:
        signal (np.array): Input signal

    Returns:
        np.array: Input Signal in the Frequency Domain
    """

    #1. Generate the DFT matrix by completing the <create_dft_matrix> function.
    length_signal = len(signal)
    dft_matrix = create_dft_matrix(length=length_signal)

    #2. Apply the DFT matrix to the signal by completing the <apply_dft_matrix> function.
    #   Store the frequency domain representation of the signal in variable <F_Signal>
    F_signal = apply_dft_matrix(dft_matrix=dft_matrix, signal=signal)

    return F_signal

def create_dft_matrix(length : int) -> np.array:
    """Returns a DFT matrix to enable calculation of the DFT.

    Args:
        length (int): length of the input signal or N of the NxN DFT Matrix.

    Returns:
        np.array: NxN DFT Matrix
    """
    dft_mat = np.zeros(shape=(length,length), dtype=np.cdouble)

    #***************************** Please add your code implementation under this line *****************************
    # Hint: The complex number e^(-4j) can be represented in numpy by <np.exp(-4j)>

    w = np.exp(-1j*(2/length)*(np.pi))

    for k in range(length):
        for n in range(length):
            dft_mat[k][n]=w**(k*n)

    #***************************** Please add your code implementation above this line *****************************

    return dft_mat

def apply_dft_matrix(dft_matrix : np.array, signal : np.array):
    """_summary_

    Args:
        dft_matrix (np.array): The DFT Matrix
        signal (np.array): The time-domain signal

    Returns:
        F_signal (np.array): The frequency-domain signal
    """
    signal_length = signal.shape[0]
    F_signal = np.zeros(shape=signal_length)

    #***************************** Please add your code implementation under this line *****************************
    # Hint: search the numpy library documentation for the <np.matmul> operation.

    F_signal =np.matmul(dft_matrix,signal)

    #***************************** Please add your code implementation above this line *****************************

    return F_signal

def plot_dft_magnitude_angle(frequency_axis : np.array, f_signal : np.array, fs : int=1, format : str=None):
    """_summary_

    Args:
        frequency_axis (np.array).
        f_signal (np.array): input signal in the frequency domain
        fs (int, optional): Sampling Frequency. Defaults to 1.
        format (string, optional): Plotting Format. Defaults to None.
    """
    
    N = len(f_signal)
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212, sharex=ax1)

    #***************************** Please add your code implementation under this line *****************************
    if(format == None):
        ax1.set_ylabel("Magnitude")
        ax1.stem(frequency_axis, np.abs(f_signal))
        ax2.set_ylabel("Phase (radians)")
        ax2.stem(frequency_axis, np.angle(f_signal))
        ax2.set_xlabel("Frequency Bins")
        ax2.set_ylim((-np.pi, np.pi))

    if(format == "ZeroPhase"):

        #Hint: the numpy library documentation for the <np.where> operation might be useful.

        ax1.set_ylabel("Magnitude")
        ax1.stem(frequency_axis, np.abs(f_signal))
        ax2.set_ylabel("Phase (radians)")
        ax2.stem(frequency_axis, np.where(np.abs(f_signal) < 1e-1, 0, np.angle(f_signal)))
        ax2.set_xlabel("Frequency Bins")
        ax2.set_ylim((-np.pi, np.pi))
        
    if(format == "Normalized"):

        normalisedfreq = frequency_axis / len(frequency_axis)

        ax1.set_ylabel("Magnitude")
        ax1.stem(normalisedfreq, np.abs(f_signal))
        ax2.set_ylabel("Phase (radians)")
        ax2.stem(normalisedfreq, np.where(np.abs(f_signal) < 1e-1, 0, np.angle(f_signal)))
        ax2.set_xlabel("Normalised Freq Bins")
        ax2.set_ylim((-np.pi, np.pi))

    if(format == "Centered_Normalized"):

        normalisedfreq = (frequency_axis / len(frequency_axis)) - 0.5

        # rollednormalisedfreq = np.arange(-0.5, 0.5, N) #np.roll(normalisedfreq, 16) 
        # rolledmag = np.roll(np.abs(f_signal), int(len(normalisedfreq/2)))
        # rolledphase = np.roll(np.angle(f_signal), int(len(normalisedfreq/2)))

        part1 = f_signal[0:N//2]
        part2 = f_signal[N//2:]
        f_signalnew = np.concatenate((part2, part1))

        ax1.set_ylabel("Magnitude")
        ax1.stem(normalisedfreq, np.abs(f_signalnew))
        ax2.set_ylabel("Phase (radians)")
        ax2.stem(normalisedfreq, np.where(np.abs(f_signalnew) < 1e-1, 0, np.angle(f_signalnew)))
        ax2.set_xlabel("Normalised Freq Bins")
        ax2.set_ylim((-np.pi, np.pi))



        # print(normalisedfreq)
        # print("_______", rollednormalisedfreq)
        
        
        # ax1.set_ylabel("Magnitude")
        # ax1.stem(rollednormalisedfreq, rolledmag)
        # ax2.set_ylabel("Phase (radians)")
        # ax2.stem(rollednormalisedfreq, np.where(rolledmag < 1e-1, 0, rolledphase))
        # ax2.set_xlabel("Normalised Freq Bins")
        # ax2.set_ylim((-np.pi, np.pi))

        # normalisedfreq = (frequency_axis / len(frequency_axis)) - 0.5

        # ax1.set_ylabel("Magnitude")
        # ax1.stem(normalisedfreq, np.abs(f_signal))
        # ax2.set_ylabel("Phase (radians)")
        # ax2.stem(normalisedfreq, np.where(np.abs(f_signal) < 1e-1, 0, np.angle(f_signal)))
        # ax2.set_xlabel("Normalised Freq Bins")
        # ax2.set_ylim((-np.pi, np.pi))



    if(format == "Centered_Original_Scale"):
        pass

    #***************************** Please add your code implementation above this line *****************************

def idft(signal : np.array) -> np.array:
    """Returns the Inverse Discrete Fourier Transform of a frequency domain signal. 

    Args:
        signal (np.array): Input frequency signal

    Returns:
        np.array: Transforms input signal to time-domain.
    """
    length_signal = len(signal)
    time_domain_signal = np.zeros(length_signal)

    #***************************** Please add your code implementation under this line *****************************
    # Try to only use the <create_dft_matrix> and <apply_dft_matrix> operations that you have already implemented and some numpy operations.
    # Hint: look up the <np.conjugate>

    dft_mat = create_dft_matrix(length_signal)

    # DFT matrix is unitary, which means that the inverse is equal to the transpose of the conjugate

    conj_dft_mat = (np.conjugate(dft_mat))
    #print(conj_dft_mat)
    inv_dft_mat = conj_dft_mat.transpose()
    #print("______________________", inv_dft_mat)

    time_domain_signal = (apply_dft_matrix(inv_dft_mat, signal)) / (length_signal) 

    #***************************** Please add your code implementation above this line *****************************

    return time_domain_signal

def convolve_signals(x_signal : np.array, y_signal : np.array) -> np.array:
    """Convolve the X and Y signal using the DFT.

    Args:
        x_signal (np.array)
        y_signal (np.array)

    Returns:
        np.array: z_signal = x_signal * y_signal
    """
    z_signal = np.zeros(len(x_signal))
    
    #***************************** Please add your code implementation under this line *****************************
    #Make sure you do the time-domain convolution in the frequency domain. 

    dft_mat_x = create_dft_matrix(len(x_signal))
    dft_mat_y = create_dft_matrix(len(y_signal))
    x_dft = apply_dft_matrix(dft_mat_x, x_signal)
    y_dft = apply_dft_matrix(dft_mat_y, y_signal)

    z_signal_dft = x_dft * y_dft

    z_signal = idft(z_signal_dft)



    #***************************** Please add your code implementation above this line *****************************

    return z_signal

def zero_pad_signal(x_signal : np.array, new_length : int) -> np.array:
    """Append zeros to the end of the input signal until it reaches the desired [new_length].

    Args:
        x_signal (np.array): input signal.
        new_length (int): new length of signal such that we add new_length minus original_length number of zeros to the end of the signal

    Returns:
        np.array: zero-padded signal
    """
    zero_padded_signal = np.zeros(new_length)

    #***************************** Please add your code implementation under this line *****************************

    #***************************** Please add your code implementation above this line *****************************

    return zero_padded_signal
