from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from eeg_data import load_eeg_file,attach_annotations,select_channel,print_eeg_summary, create_epochs_labelled, print_label_distribution, ID_TO_STAGE, balancing_dataset_on_wake, extract_epoch_features

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
    plot_eeg_segment(raw, start_sec=31500, duration_sec=30)

    X, y = create_epochs_labelled(raw, epoch_length_sec=30)

    X, y = balancing_dataset_on_wake(X, y)

    X_features = extract_epoch_features(X, raw.info["sfreq"])

    print("\nFeatures extracted")
    print("-" * 40)
    print("X features shape:", X_features.shape)
    print("y shape:", y.shape)
    
    """
    print("\nDataset created")
    print("-" * 40)
    print("X shape:", X.shape)
    print("y shape:", y.shape)"""
    

    print_label_distribution(y)

    print("\nFirst 10 labels:")
    for label in y[:100]:
        print(label, ID_TO_STAGE[label])

if __name__ == "__main__":
    main()