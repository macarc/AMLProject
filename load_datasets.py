# This provides functions for loading audio data. It also provides a caching mechanism, which makes loading audio data between runs
# much faster (30 to 40 times faster).
# MAKE SURE THE CACHE IS RELOADED IF extract_features HAS CHANGED!
# To reload the cache, run this as a script (making sure to set the correct feature extraction at the bottom of the file):
# python load_datasets.py


import constants
import csv
import gc
from helpers import adjust_length
from labels import label_to_number
import librosa
import numpy as np


train_csv = "datasets/ARCA23K-FSD.ground_truth/train.csv"
train_dir = "datasets/FSD50K.dev_audio"
val_csv = "datasets/ARCA23K-FSD.ground_truth/val.csv"
val_dir = "datasets/FSD50K.dev_audio"
test_csv = "datasets/ARCA23K-FSD.ground_truth/test.csv"
test_dir = "datasets/FSD50K.eval_audio"


def load_train_data(extract_features, force_reload=False):
    """
    Load the training dataset
    - extract features: function that takes an audio file (a numpy array with length (N,)) and produces its features
    - force_reload: if True, reloads all audio files from disk; otherwise, load them from cached files (see header)
    """
    if force_reload:
        return _reload_train_data(extract_features)

    try:
        return np.load("datasets/train_features.npy"), np.load("datasets/train_labels.npy")
    except:
        return _reload_train_data(extract_features)


def load_val_data(extract_features, force_reload=False):
    """
    Load the validation dataset
    - extract features: function that takes an audio file (a numpy array with length (N,)) and produces its features
    - force_reload: if True, reloads all audio files from disk; otherwise, load them from cached files (see header)
    """
    if force_reload:
        return _reload_val_data(extract_features)

    try:
        return np.load("datasets/val_features.npy"), np.load("datasets/val_labels.npy")
    except:
        return _reload_val_data(extract_features)


def load_test_data(extract_features, force_reload=False):
    """
    Load the test dataset
    - extract features: function that takes an audio file (a numpy array with length (N,)) and produces its features
    - force_reload: if True, reloads all audio files from disk; otherwise, load them from cached files (see header)
    """
    if force_reload:
        return _reload_test_data(extract_features)

    try:
        return np.load("datasets/test_features.npy"), np.load("datasets/test_labels.npy")
    except:
        return _reload_test_data(extract_features)


def _reload_train_data(extract_features):
    train_features, train_labels = _load(train_csv, train_dir, extract_features)
    np.save("datasets/train_features.npy", train_features)
    np.save("datasets/train_labels.npy", train_labels)
    return train_features, train_labels


def _reload_val_data(extract_features):
    val_features, val_labels = _load(val_csv, val_dir, extract_features)
    np.save("datasets/val_features.npy", val_features)
    np.save("datasets/val_labels.npy", val_labels)
    return val_features, val_labels


def _reload_test_data(extract_features):
    test_features, test_labels = _load(test_csv, test_dir, extract_features)
    np.save("datasets/test_features.npy", test_features)
    np.save("datasets/test_labels.npy", test_labels)
    return test_features, test_labels


def _load(audio_list_filename, audio_directory, extract_features):
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
            audiofilename = f"{audio_directory}/{audiofile}.wav"

            audio, sr = librosa.load(audiofilename, sr=constants.SAMPLE_RATE)
            assert sr == constants.SAMPLE_RATE

            features.append(extract_features(audio))
            labels.append(label_to_number(label))

    print(f"Loaded {len(features)} audio files")

    # Convert to numpy arrays
    features = np.array(features)
    labels = np.array(labels)

    # Force garbage collection, so that we remove unnecessary audio file data from memory before we load the next data subset
    gc.collect()

    return features, labels


def reload_cache(extract_features):
    """
    Reload the cache. This is necessary if you wish to extract different features.
    - extract features: function that takes an audio file (a numpy array with length (N,)) and produces its features
    """
    # Reload the data
    load_train_data(extract_features, force_reload=True)
    load_val_data(extract_features, force_reload=True)
    load_test_data(extract_features, force_reload=True)


if __name__ == "__main__":
    # Feature is just making all the files the same length (by truncating/zero-padding)
    extract_features = adjust_length

    # Reload cache
    reload_cache(extract_features)
