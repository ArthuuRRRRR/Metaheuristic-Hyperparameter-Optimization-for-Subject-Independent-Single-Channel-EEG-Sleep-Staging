from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from mne.datasets.sleep_physionet.age import fetch_data

from eeg_data import (
    load_eeg_file,
    attach_annotations,
    select_channel,
    create_epochs_labelled,
    balancing_dataset_on_wake,
    extract_epoch_features,
)


AVAILABLE_SUBJECTS = [
    subject_id
    for subject_id in range(83)
    if subject_id not in {39, 68, 69, 78, 79}
]

EXPECTED_SUBJECTS = 78
EXPECTED_RECORDINGS = 153


def process_patient(psg_path, hypnogram_path, channel_name):

    raw = load_eeg_file(psg_path, preload=False)

    try:
        raw = attach_annotations(raw, hypnogram_path)
        raw = select_channel(raw, channel_name)

        X, y = create_epochs_labelled(
            raw,
            epoch_length_sec=30
        )

        X, y = balancing_dataset_on_wake(
            X,
            y,
            wake_label=0,
            margin_epochs=60
        )

        X_features = extract_epoch_features(
            X,
            raw.info["sfreq"]
        )

    finally:
        raw.close()

    return X_features, y


def load_subject_cache(cache_path):

    cached_data = np.load(
        cache_path,
        allow_pickle=False
    )

    return (
        cached_data["X_features"],
        cached_data["y"],
        cached_data["groups_subject"],
        cached_data["groups_recording"],
    )


def process_subject(
    subject_id,
    channel_name,
    subject_cache_path
):

    print(f"\nProcessing subject {subject_id}")
    print("-" * 40)

    all_X_features = []
    all_y = []
    all_groups_recording = []

    # Les EDF sont stockés seulement pendant le traitement du sujet
    with TemporaryDirectory(
        prefix=f"sleep_edf_subject_{subject_id:02d}_"
    ) as temporary_directory:

        recordings = fetch_data(
            subjects=[subject_id],
            recording=[1, 2],
            path=temporary_directory,
            on_missing="ignore",
            force_update=False
        )

        if len(recordings) == 0:
            raise RuntimeError(
                f"No recording found for subject {subject_id}"
            )

        for psg_path, hypnogram_path in recordings:

            psg_path = Path(psg_path)
            hypnogram_path = Path(hypnogram_path)

            recording_id = psg_path.name.replace(
                "-PSG.edf",
                ""
            )

            print(f"Processing recording: {recording_id}")

            X_patient, y_patient = process_patient(
                psg_path,
                hypnogram_path,
                channel_name
            )

            all_X_features.append(X_patient)
            all_y.append(y_patient)

            recording_groups = np.full(
                len(y_patient),
                recording_id,
                dtype="<U32"
            )

            all_groups_recording.append(
                recording_groups
            )

    X_features = np.concatenate(
        all_X_features,
        axis=0
    )

    y = np.concatenate(
        all_y,
        axis=0
    )

    groups_recording = np.concatenate(
        all_groups_recording,
        axis=0
    )

    # Les deux nuits du même sujet reçoivent le même groupe
    groups_subject = np.full(
        len(y),
        subject_id,
        dtype=np.int16
    )

    np.savez_compressed(
        subject_cache_path,
        X_features=X_features,
        y=y,
        groups_subject=groups_subject,
        groups_recording=groups_recording
    )

    print(
        f"Subject {subject_id} saved: "
        f"{len(y)} epochs, "
        f"{len(np.unique(groups_recording))} recording(s)"
    )

    return (
        X_features,
        y,
        groups_subject,
        groups_recording
    )


def load_sleep_cassette_dataset(
    channel_name="EEG Fpz-Cz",
    cache_directory="data/sleep_cassette_features",
    force_rebuild=False
):

    cache_directory = Path(cache_directory)
    cache_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    complete_cache_path = (
        cache_directory
        / "sleep_cassette_complete.npz"
    )

    # Chargement direct lors des exécutions suivantes
    if complete_cache_path.exists() and not force_rebuild:

        print("\nLoading complete feature cache")
        print(complete_cache_path)

        return load_subject_cache(
            complete_cache_path
        )

    all_X_features = []
    all_y = []
    all_groups_subject = []
    all_groups_recording = []

    for subject_id in AVAILABLE_SUBJECTS:

        subject_cache_path = (
            cache_directory
            / f"subject_{subject_id:02d}.npz"
        )

        if subject_cache_path.exists() and not force_rebuild:

            print(
                f"Loading cached subject {subject_id}"
            )

            (
                X_subject,
                y_subject,
                groups_subject,
                groups_recording
            ) = load_subject_cache(
                subject_cache_path
            )

        else:

            (
                X_subject,
                y_subject,
                groups_subject,
                groups_recording
            ) = process_subject(
                subject_id=subject_id,
                channel_name=channel_name,
                subject_cache_path=subject_cache_path
            )

        all_X_features.append(X_subject)
        all_y.append(y_subject)
        all_groups_subject.append(groups_subject)
        all_groups_recording.append(groups_recording)

    X_features = np.concatenate(
        all_X_features,
        axis=0
    )

    y = np.concatenate(
        all_y,
        axis=0
    )

    groups_subject = np.concatenate(
        all_groups_subject,
        axis=0
    )

    groups_recording = np.concatenate(
        all_groups_recording,
        axis=0
    )

    number_of_subjects = len(
        np.unique(groups_subject)
    )

    number_of_recordings = len(
        np.unique(groups_recording)
    )

    print("\nComplete Sleep Cassette dataset")
    print("-" * 40)
    print("X features shape:", X_features.shape)
    print("y shape:", y.shape)
    print("Number of subjects:", number_of_subjects)
    print("Number of recordings:", number_of_recordings)

    if number_of_subjects != EXPECTED_SUBJECTS:
        raise RuntimeError(
            f"Expected {EXPECTED_SUBJECTS} subjects, "
            f"but found {number_of_subjects}"
        )

    if number_of_recordings != EXPECTED_RECORDINGS:
        raise RuntimeError(
            f"Expected {EXPECTED_RECORDINGS} recordings, "
            f"but found {number_of_recordings}"
        )

    np.savez_compressed(
        complete_cache_path,
        X_features=X_features,
        y=y,
        groups_subject=groups_subject,
        groups_recording=groups_recording
    )

    print(
        f"Complete dataset saved to: "
        f"{complete_cache_path}"
    )

    return (
        X_features,
        y,
        groups_subject,
        groups_recording
    )