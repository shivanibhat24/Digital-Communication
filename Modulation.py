import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Set dark theme and custom colors
plt.style.use('dark_background')
mod_colors = {'BPSK': '#39FF14',    # Neon Green
              'QPSK': '#7DF9FF',    # Electric Blue
              '16-PSK': '#FFFF00',  # Yellow
              '16-QAM': '#BF00FF'}  # Neon Purple

# Function to add AWGN noise
def add_awgn_noise(signal, snr_db):
    avg_signal_power = np.mean(np.abs(signal)**2)
    snr_linear = 10**(snr_db / 10)
    noise_power = avg_signal_power / snr_linear
    noise = np.sqrt(noise_power / 2) * (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape))
    return signal + noise

# Function for BPSK modulation
def bpsk_modulation(data):
    return 2*data - 1

# Function for QPSK modulation
def qpsk_modulation(data):
    data_reshaped = data.reshape((-1, 2))
    symbols = (2*data_reshaped[:, 0] - 1) + 1j * (2*data_reshaped[:, 1] - 1)
    return symbols / np.sqrt(2)

# Function for 16-PSK modulation
def psk16_modulation(data):
    M = 16
    data_symbols = np.packbits(data).reshape(-1) % M
    phase = (2 * np.pi * data_symbols) / M
    return np.exp(1j * phase)

# Function for 16-QAM modulation
def qam16_modulation(data):
    data_reshaped = data.reshape((-1, 4))
    symbols = (2*data_reshaped[:, 0] - 1) + 1j * (2*data_reshaped[:, 2] - 1)  # Real and imaginary components
    symbols += (2 * (2*data_reshaped[:, 1] - 1) + 1j * 2 * (2*data_reshaped[:, 3] - 1)) * 1j
    return symbols

# Plotting function with custom color
def plot_constellation(mod_data, title, ax, color):
    ax.scatter(mod_data.real, mod_data.imag, color=color, edgecolor=color, facecolor=color)
    ax.set_title(title, color='white')
    ax.grid(True, color='white')
    ax.axhline(0, color='white', linewidth=0.5)
    ax.axvline(0, color='white', linewidth=0.5)
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])

# Get user input for SNR values
snr_input = input("Enter SNR values (comma separated): ")
SNR_values = list(map(int, snr_input.split(',')))

# Parameters
numBits = 5000  # Number of bits

# Make sure the number of bits is a multiple of 4 for 16-QAM and 16-PSK
if numBits % 4 != 0:
    numBits = numBits + (4 - (numBits % 4))

# Generate binary data
data = np.random.randint(0, 2, numBits)

# Modulation schemes
modSchemes = {'BPSK': bpsk_modulation, 'QPSK': qpsk_modulation, '16-PSK': psk16_modulation, '16-QAM': qam16_modulation}

# Loop through different modulation schemes
fig, axes = plt.subplots(len(modSchemes), len(SNR_values) + 1, figsize=(15, 10))
fig.suptitle('Digital Communication Plots', fontsize=16, color='white')

for modIdx, (modName, modFunction) in enumerate(modSchemes.items()):
    color = mod_colors[modName]
    
    # Perform modulation
    if modName in ['BPSK', 'QPSK']:
        modData = modFunction(data[:numBits if modName == 'BPSK' else numBits//2])
    elif modName == '16-PSK':
        modData = modFunction(data[:numBits])  # 4 bits per symbol for 16-PSK
    elif modName == '16-QAM':
        modData = modFunction(data[:numBits])  # 4 bits per symbol for 16-QAM
    
    # Plot constellation before noise
    plot_constellation(modData, f'{modName} Constellation (No Noise)', axes[modIdx, 0], color)
    
    # Add AWGN noise for different SNRs and plot
    for snrIdx, snr in enumerate(SNR_values):
        noisyData = add_awgn_noise(modData, snr)
        plot_constellation(noisyData, f'{modName} with AWGN {snr} dB', axes[modIdx, snrIdx + 1], color)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to fit the title
plt.show()
