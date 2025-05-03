# This provides functions for loading audio data. It also provides a caching mechanism, which makes loading audio data between runs
# much faster (30 to 40 times faster).
# To check if your computer is physically capable of loading all the data,
# run this file as a script:
# python load_datasets.py

import librosa
import numpy as np
import torch

from lib import constants
from lib.cache import load_data_from_cache
from lib.helpers import get_torch_backend
from model import extract_features


def load_data():
    """
    Load training, validation and test data from the cache. Run the extract_features file to populate this cache.

    Returns:
    train_features, train_labels, val_features, val_labels, test_features, test_labels : torch tensors
    """

    features, labels = load_data_from_cache()

    backend_device = get_torch_backend()

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


def load_files(audio_filenames):
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

    return torch.tensor(features, device=get_torch_backend(), dtype=torch.float32)
