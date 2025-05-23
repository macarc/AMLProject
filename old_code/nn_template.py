from audiodataset import AudioDataSet
from helpers import adjust_length, get_torch_backend
from labels import label_count, number_to_label
from load_datasets import load_data_to_device
import numpy as np
import librosa
import constants
import torch
from torch.utils.data import DataLoader


# %% FEATURE EXTRACTION


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


# %% MODEL TRAINING
if __name__ == "__main__":
    # %% TORCH BACKEND

    backend_dev = get_torch_backend()

    # %% LOAD FEATURES

    # NB: pass force_reload = True here when extract_features has changed!
    (
        train_features,
        train_labels,
        val_features,
        val_labels,
        test_features,
        test_labels,
    ) = load_data_to_device(backend_dev, extract_features, force_reload=False)

    train_dataset = AudioDataSet(train_features, train_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=50, shuffle=True)

    # %% DO THINGS HERE!
