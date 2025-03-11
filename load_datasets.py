import constants
import csv
import gc
from helpers import adjust_length
from labels import label_to_number
import librosa
import numpy as np


def load_train_data():
    return _load(
        "datasets/ARCA23K-FSD.ground_truth/train.csv", "datasets/FSD50K.dev_audio"
    )


def load_val_data():
    return _load(
        "datasets/ARCA23K-FSD.ground_truth/val.csv", "datasets/FSD50K.dev_audio"
    )


def load_test_data():
    return _load(
        "datasets/ARCA23K-FSD.ground_truth/test.csv", "datasets/FSD50K.eval_audio"
    )


def _load(audio_list_filename, audio_directory):
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

            data.append(adjust_length(audio))
            labels.append(label_to_number(label))

    print(f"Loaded {len(data)} audio files")

    # Convert to numpy arrays
    data = np.array(data)
    labels = np.array(labels)

    # Force garbage collection, so that we remove unnecessary audio file data from memory before we load the next data subset
    gc.collect()

    return data, labels


if __name__ == "__main__":
    # Load all the data
    load_train_data()
    load_val_data()
    load_test_data()
