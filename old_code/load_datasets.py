# This provides functions for loading audio data. It also provides a caching mechanism, which makes loading audio data between runs
# much faster (30 to 40 times faster).
# To check if your computer is physically capable of loading all the data,
# run this file as a script:
# python load_datasets.py
import os
import constants
import csv
import gc
from helpers import adjust_length
from labels import label_to_number
import librosa
import numpy as np
import torch


train_dir = "datasets/FSD50K.dev_audio"
test_dir = "datasets/FSD50K.eval_audio"


def load_data_to_device(backend_device, extract_features, force_reload=False):
    """
    Load training, validation and test data, using a cache if possible
    - backend_device - torch device to use, see helpers.get_backend_device()
    - extract_features - function to extract features from data
    - force_reload - if true, reload the data from the audio files

    returns
    train_features, train_labels, val_features, val_labels, test_features, test_labels : torch tensors
    """

    features, labels = load_data(extract_features, force_reload=force_reload)

    # Number of data points in each subset
    Ntrain = round(0.8 * len(labels))
    Nval = round(0.1 * len(labels))
    Ntest = len(labels) - Ntrain - Nval

    print(
        f"Loading {Ntrain} training, {Nval} validation and {Ntest} test data examples"
    )

    # Get train/val/test subsets
    train_features = features[:Ntrain]
    train_labels = labels[:Ntrain]
    val_features = features[Ntrain : Ntrain + Nval]
    val_labels = labels[Ntrain : Ntrain + Nval]
    test_features = features[Ntrain + Nval :]
    test_labels = labels[Ntrain + Nval :]

    # Convert to tensors
    train_features = torch.tensor(
        train_features, dtype=torch.float32, device=backend_device
    )
    train_labels = torch.tensor(
        train_labels, dtype=torch.float32, device=backend_device
    )
    val_features = torch.tensor(
        val_features, dtype=torch.float32, device=backend_device
    )
    val_labels = torch.tensor(val_labels, dtype=torch.float32, device=backend_device)
    test_features = torch.tensor(
        test_features, dtype=torch.float32, device=backend_device
    )
    test_labels = torch.tensor(test_labels, dtype=torch.float32, device=backend_device)

    # Check that labels are correct shape
    assert train_labels.shape == torch.Size([Ntrain])
    assert val_labels.shape == torch.Size([Nval])
    assert test_labels.shape == torch.Size([Ntest])

    return (
        train_features,
        train_labels,
        val_features,
        val_labels,
        test_features,
        test_labels,
    )


def load_data(extract_features, force_reload=False):
    """
    Load training, validation and test data, using a cache if possible
    - backend_device - torch device to use, see helpers.get_backend_device()
    - extract_features - function to extract features from data
    - force_reload - if true, reload the data from the audio files

    returns
    train_features, train_labels, val_features, val_labels, test_features, test_labels : numpy arrays
    """
    # If force_reload is true, reload, otherwise try to use cache (but fall back on reloading)
    if force_reload:
        features, labels = _load_files_in_csv("shuffled_dataset.csv", extract_features)
    else:
        try:
            features = np.load("datasets/features.npy")
            labels = np.load("datasets/labels.npy")
            return features, labels
        except FileNotFoundError:
            features, labels = _load_files_in_csv(
                "shuffled_dataset.csv", extract_features
            )

    # If the cache wasn't used, save the features/labels to the cache
    np.save("datasets/features.npy", features)
    np.save("datasets/labels.npy", labels)

    return features, labels


def _load_files_in_csv(audio_list_filename, extract_features):
    """
    Load all audio files listed in the CSV file.
    Returns: features, labels : numpy arrays
    """

    # Lists to hold features/labels
    features = []
    labels = []

    # Open the CSV file containing the list of audio filenames
    with open(audio_list_filename) as csvfile:
        reader = csv.reader(csvfile)
        # Skip the heading
        next(iter(reader))

        # Get each audio file and append to features/labels
        for i, (audiofile, label, _) in enumerate(reader):
            # Load file from either dev or eval folder
            dev_path = f"datasets/FSD50K.dev_audio/{audiofile}.wav"
            eval_path = f"datasets/FSD50K.eval_audio/{audiofile}.wav"

            if os.path.exists(dev_path):
                audio, sr = librosa.load(dev_path, sr=constants.SAMPLE_RATE)
            elif os.path.exists(eval_path):
                audio, sr = librosa.load(eval_path, sr=constants.SAMPLE_RATE)
            else:
                raise FileNotFoundError(f"datasets/*/{audiofile}.wav")

            assert sr == constants.SAMPLE_RATE

            # Get features
            audio_features = extract_features(audio)

            # If there are no features, skip this audio file
            if audio_features is None:
                print(f"No features found for file {audiofile}.wav, skipping")
                continue

            # Append features/label to lists
            features.append(audio_features)
            labels.append(label_to_number(label))

    # Convert to numpy arrays
    features = np.array(features)
    labels = np.array(labels)

    assert len(features) == len(labels)
    print(f"Loaded {len(features)} audio files")

    # Force garbage collection, so that we remove unnecessary audio file data from memory before we load the next data subset
    gc.collect()

    return features, labels


def load_files(audio_filenames, backend_dev, extract_features):
    """
    Load features from a list of filenames. This is a shortcut, used in the GUI application.
    Returns: features : torch tensor
    """

    # List to hold features
    features = []

    # Get each audio file and append to features
    for audiofile in audio_filenames:
        # Load file
        audio, sr = librosa.load(audiofile, sr=constants.SAMPLE_RATE)
        assert sr == constants.SAMPLE_RATE

        # Get features
        audio_features = extract_features(audio)

        # If there are no features, skip this audio file
        if audio_features is None:
            print(f"No features found for file {audiofile}.wav, skipping")
            continue

        # Append features/label to lists
        features.append(audio_features)

    # Convert to numpy arrays
    features = np.array(features)

    return torch.tensor(features, device=backend_dev, dtype=torch.float32)


if __name__ == "__main__":
    # Feature is just making all the files the same length (by truncating/zero-padding)
    extract_features = adjust_length

    # Load the data - to check if it works! This also saves to the cache
    load_data(extract_features, force_reload=True)
