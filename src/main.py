from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from eeg_data import load_eeg_file,attach_annotations,select_channel,print_eeg_summary

def plot_eeg_segment(raw, start_sec=0, duration_sec=30): # visu rapide

    sfreq = raw.info["sfreq"]

    start_sample = int(start_sec * sfreq)
    stop_sample = int((start_sec + duration_sec) * sfreq)

    data = raw.get_data(start=start_sample, stop=stop_sample)

    eeg_signal = data[0]
    time = np.arange(len(eeg_signal)) / sfreq + start_sec

    plt.figure(figsize=(12, 4))
    plt.plot(time, eeg_signal)
    plt.xlabel("Time (seconds)")
    plt.ylabel("EEG amplitude")
    plt.title(f"EEG segment from {start_sec}s to {start_sec + duration_sec}s")
    plt.grid(True)
    plt.show()



def main() :
    psg_path = Path("C:\\Users\\delha\\Downloads\\SC4001E0-PSG.edf")
    hypnogram_path = Path("C:\\Users\\delha\\Downloads\\SC4001EC-Hypnogram.edf")
    channel_name = "EEG Fpz-Cz"

    raw = load_eeg_file(psg_path, preload=False)
    raw = attach_annotations(raw, hypnogram_path)
    raw = select_channel(raw, channel_name)

    print_eeg_summary(raw)
    plot_eeg_segment(raw, start_sec=0, duration_sec=30)


if __name__ == "__main__":
    main()