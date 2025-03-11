from helpers import adjust_length
from labels import label_count, number_to_label
from load_datasets import load_train_data, load_val_data, load_test_data, reload_cache
import numpy as np
import librosa
import constants
import torch


def extract_features(audio_data):
    """Extract a mel spectrogram of the audio data"""
    assert len(audio_data.shape) == 1

    # Adjust length
    audio_data = adjust_length(audio_data)

    # Check that we did adjust correctly
    assert len(audio_data) == constants.AUDIO_LENGTH

    # If there are no non-zero values, return none
    if not np.any(audio_data):
        return None

    # Divide by the maximum value to normalise
    normalised = audio_data / np.max(np.abs(audio_data))

    # Check that we did indeed normalise correctly
    assert np.max(np.abs(normalised)) == 1

    # Extract a mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio_data, sr=constants.SAMPLE_RATE
    )

    return mel_spectrogram


if __name__ == "__main__":
    # Uncomment this for the first run!
    # reload_cache(extract_features)

    # Load dataset
    train_features, train_labels = load_train_data(extract_features)
    val_features, val_labels = load_val_data(extract_features)
    test_features, test_labels = load_test_data(extract_features)

    # Convert to tensors
    train_features = torch.tensor(train_features)
    train_labels = torch.tensor(train_labels)
    val_features = torch.tensor(val_features)
    val_labels = torch.tensor(val_labels)
    test_features = torch.tensor(test_features)
    test_labels = torch.tensor(test_labels)

    # Get number of data points in each subset
    Ntrain = train_features.shape[0]
    Nval = val_features.shape[0]
    Ntest = test_features.shape[0]

    # Check that features are correct shape
    assert train_features.shape == torch.Size([Ntrain, 128, 87])
    assert val_features.shape == torch.Size([Nval, 128, 87])
    assert test_features.shape == torch.Size([Ntest, 128, 87])

    # Check that labels are correct shape
    assert train_labels.shape == torch.Size([Ntrain])
    assert val_labels.shape == torch.Size([Nval])
    assert test_labels.shape == torch.Size([Ntest])

    # Get label with minimum number of training examples (just to check it isn't a tiny number of examples)
    min_label, min_count = min(
        [(i, torch.sum(train_labels == i).item()) for i in range(label_count())],
        key=lambda a: a[1],
    )
    print(
        f"Label with the least training examples is '{number_to_label(min_label)}' with {min_count} instances"
    )

    ## DO THINGS HERE :)
