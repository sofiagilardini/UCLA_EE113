from matplotlib import pyplot as plt
import numpy as np 

t = np.linspace(1, 5, 1000)

def plot(f0, fs):
    x_tc = np.cos(2*np.pi*f0*t)
    tsample = np.linspace(1, 5, fs)
    x_td = np.cos(2*np.pi*f0*tsample)

    fig, axs = plt.subplots(2, sharey = True)
    fig.subplots_adjust(hspace=0.5)
    axs[0].plot(t, x_tc)
    axs[0].title.set_text('Continous')
    axs[1].plot(tsample, x_td,'o', markersize = 5)
    axs[1].title.set_text('Discrete')
    plt.show()

#i
plot(3,10)
#ii
plot(7, 10)
#iii
plot(13, 10)

# Nyquist frequency is 2 * the maximum frequency in the spectrum of a signal. In this case with a cos function, there is only one frequency.
# Aliasing happens when sampling is below Nyquist - which means the signal cannot be recovered perfectly
# Therefore i) will be recovered perfectly as 10 > 6
# ii) will not be recovered perfectly as 10 < 14
# ii) will not be recovered perfectly as 10 < 26
    


