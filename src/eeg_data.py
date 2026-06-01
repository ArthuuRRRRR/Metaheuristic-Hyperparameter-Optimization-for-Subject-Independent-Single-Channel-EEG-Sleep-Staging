from pathlib import Path
import numpy as np
import mne

STAGE_TO_ID = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,   # On regroupe stage 3 et 4 en N3
    "Sleep stage R": 4,
}

ID_TO_STAGE = {
    0: "W",
    1: "N1",
    2: "N2",
    3: "N3",
    4: "REM",
}





def check_file_exists(file_path: str | Path) -> Path:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    return path


def load_eeg_file(psg_path: str | Path, preload: bool = False) -> mne.io.BaseRaw:

    psg_path = check_file_exists(psg_path)

    raw = mne.io.read_raw_edf(input_fname=str(psg_path),preload=preload,verbose=False)

    return raw


def load_hypnogram(hypnogram_path: str | Path) -> mne.Annotations:

    hypnogram_path = check_file_exists(hypnogram_path)

    annotations = mne.read_annotations(str(hypnogram_path))

    return annotations


def attach_annotations(raw: mne.io.BaseRaw,hypnogram_path: str | Path) -> mne.io.BaseRaw: # attacher les annotations de sommeil au signal EEG
    annotations = load_hypnogram(hypnogram_path)

    raw.set_annotations(annotations, emit_warning=False)

    return raw


def select_channel(raw: mne.io.BaseRaw, channel_name: str) -> mne.io.BaseRaw: # Mon choix de canal

    if channel_name not in raw.ch_names:
        available_channels = ", ".join(raw.ch_names)
        raise ValueError(f"Channel '{channel_name}' not found.\n"f"Available channels: {available_channels}")

    raw = raw.copy()
    raw.pick([channel_name])

    return raw


def print_eeg_summary(raw: mne.io.BaseRaw) -> None:

    duration_seconds = raw.n_times / raw.info["sfreq"]
    duration_minutes = duration_seconds / 60

    print("EEG file loaded successfully")
    print("-" * 40)
    print(f"Sampling frequency: {raw.info['sfreq']} Hz")
    print(f"Number of channels: {len(raw.ch_names)}")
    print(f"Channels: {raw.ch_names}")
    print(f"Duration: {duration_minutes:.2f} minutes")
    print(f"Number of annotations: {len(raw.annotations)}")

    if len(raw.annotations) > 0:
        print("\nFirst annotations:")
        for annotation in raw.annotations[:10]:
            print(annotation)

def get_stage_at_time(raw, time_sec: float):

    for onset, duration, description in zip(raw.annotations.onset,raw.annotations.duration,raw.annotations.description):
        if onset <= time_sec < onset + duration:
            return description

    return None




def create_epochs_labelled(raw, epoch_length_sec: int = 30) :

    sfreq = raw.info["sfreq"]
    samples_per_epoch = int(epoch_length_sec * sfreq)

    signal = raw.get_data()[0]

    n_epochs = len(signal) // samples_per_epoch

    X = []
    y = []

    for epoch_index in range(n_epochs):
        start_sample = epoch_index * samples_per_epoch
        stop_sample = start_sample + samples_per_epoch

        start_time = epoch_index * epoch_length_sec
        center_time = start_time + epoch_length_sec / 2

        stage_description = get_stage_at_time(raw, center_time)

        if stage_description not in STAGE_TO_ID:
            continue

        epoch_signal = signal[start_sample:stop_sample]
        label = STAGE_TO_ID[stage_description]

        X.append(epoch_signal)
        y.append(label)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    return X, y


def print_label_distribution(y):

    unique_labels, counts = np.unique(y, return_counts=True)

    print("\nLabel distribution:")
    for label_id, count in zip(unique_labels, counts):
        label_name = ID_TO_STAGE[label_id]
        print(f"{label_name}: {count} epochs")