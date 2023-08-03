import matplotlib.pyplot as plt
import numpy as np
from pylsl import StreamInlet, resolve_stream
from muselsl import stream, list_muses
import threading


## Searches for a muse and then connects to it
# @param None
# @return None
def connect_to_muse():
    # Search for available Muse devices
    muses = list_muses()
    # Start another thread and start streaming from the first available Muse
    thread = threading.Thread(target=stream, args=(muses[0]['address'],))
    thread.daemon = True
    thread.start()
    thread.join()
    streams = resolve_stream('type', 'EEG')

    if len(streams) == 0:
        raise RuntimeError('Can\'t find EEG stream.')

    # Set active EEG stream to inlet and apply time correction
    print("Start acquiring data")
    inlet = StreamInlet(streams[0], max_chunklen=12)
    eeg_time_correction = inlet.time_correction()

    # Get the stream info and description
    info = inlet.info()
    description = info.desc()

    # Get the sampling frequency
    # This is an important value that represents how many EEG data points are
    # collected in a second. This influences our frequency band calculation.
    # for the Muse 2016, this should always be 256
    fs = int(info.nominal_srate())

    return inlet, fs

## Connects to eeg stream and then plots the data from it live in a matplotlib window
# @param inlet: the inlet to the eeg stream
# @param fs: the sampling frequency of the eeg stream
# @return None
def live_plot_muse_eeg(inlet, fs):
    # Resolve the Muse EEG stream
    streams = resolve_stream('type', 'EEG')

    if not streams:
        print("Muse EEG stream not found.")
        return

    # Number of channels in the EEG stream
    num_channels = 4

    # Set up the matplotlib figure
    plt.ion()
    fig, ax = plt.subplots(num_channels, 1, figsize=(10, 8), sharex=True)
    fig.suptitle('Muse EEG Live Plot')

    # Initialize empty lists to store data for each channel
    data = [[] for _ in range(num_channels)]

    try:
        while True:
            # Get a chunk of data from the stream
            chunk, timestamps = inlet.pull_chunk(timeout=1, max_samples=12)

            # Unpack and append the data to the corresponding channel's list
            if chunk:
                for sample in chunk:
                    for i in range(num_channels):
                        data[i].append(sample[i])

                # Plot the data for each channel
                for i in range(num_channels):
                    ax[i].clear()
                    ax[i].plot(data[i], color='b')
                    ax[i].set_ylabel(f'Channel {i + 1} (uV)')
                    ax[i].grid(True)

                plt.tight_layout()
                plt.draw()
                plt.pause(0.01)

    except KeyboardInterrupt:
        print("Live plot stopped.")

if __name__ == "__main__":
    inlet, fs = connect_to_muse()
    live_plot_muse_eeg(inlet=inlet, fs=fs)
