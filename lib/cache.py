"""
The cache stores the extracted features from the audio data, which makes it
much faster to iterate (since feature extraction can be very slow).

It stores the features and labels on disk.
"""

import numpy as np

features_cache = "datasets/features.npy"
labels_cache = "datasets/labels.npy"


def save_data_to_cache(features, labels):
    """
    Save extracted features and labels into the cache.

    Parameters:
        features (numpy.ndarray)
        labels (numpy.ndarray)
    """
    np.save(features_cache, features)
    np.save(labels_cache, labels)


def load_data_from_cache():
    """
    Load training, validation and test data from the cache

    Returns:
        features (numpy.ndarray)
        labels (numpy.ndarray)
    """
    return np.load(features_cache), np.load(labels_cache)
