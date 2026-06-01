from pathlib import Path

import mne


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