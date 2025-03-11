import constants
import csv
import gc
from helpers import adjust_length
from labels import label_to_number
import librosa
import numpy as np


def load_train_data(extract_features):
    """
    Load the training dataset
    - extract features: function that takes an audio file (a numpy array with length (N,)) and produces its features
    """
    return _load(
        "datasets/ARCA23K-FSD.ground_truth/train.csv",
        "datasets/FSD50K.dev_audio",
        extract_features,
    )


def load_val_data(extract_features):
    """
    Load the validation dataset
    - extract features: function that takes an audio file (a numpy array with length (N,)) and produces its features
    """
    return _load(
        "datasets/ARCA23K-FSD.ground_truth/val.csv",
        "datasets/FSD50K.dev_audio",
        extract_features,
    )


def load_test_data(extract_features):
    """
    Load the test dataset
    - extract features: function that takes an audio file (a numpy array with length (N,)) and produces its features
    """
    return _load(
        "datasets/ARCA23K-FSD.ground_truth/test.csv",
        "datasets/FSD50K.eval_audio",
        extract_features,
    )


def _load(audio_list_filename, audio_directory, extract_features):
    # Lists to hold data
    data = []
    labels = []

    # Open the CSV file containing the list of audio filenames
    with open(audio_list_filename) as csvfile:
        reader = csv.reader(csvfile)
        # Skip the heading
        next(iter(reader))

        # Get each audio file and append to audio_data (and label to labels)
        for i, (audiofile, label, _) in enumerate(reader):
            audiofilename = f"{audio_directory}/{audiofile}.wav"

            audio, sr = librosa.load(audiofilename, sr=constants.SAMPLE_RATE)
            assert sr == constants.SAMPLE_RATE

            data.append(extract_features(audio))
            labels.append(label_to_number(label))

    print(f"Loaded {len(data)} audio files")

    # Convert to numpy arrays
    data = np.array(data)
    labels = np.array(labels)

    # Force garbage collection, so that we remove unnecessary audio file data from memory before we load the next data subset
    gc.collect()

    return data, labels


if __name__ == "__main__":
    # Load all the data, just taking the first bit
    load_train_data(adjust_length)
    load_val_data(adjust_length)
    load_test_data(adjust_length)
