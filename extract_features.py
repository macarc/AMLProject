"""
Extract the features defined in model.py, saving the features to a cache.
Caching the feature improves load time of the audio during training by 30-40x.

This must be run after changing the features defined in model.py.
"""

import csv
import os

import librosa
import numpy as np
from tqdm import tqdm

from lib import constants
from lib.cache import save_data_to_cache
from lib.labels import label_to_number
from model import extract_features

# ==================================================
# Parameters
# ==================================================

csv_filename = "shuffled_dataset.csv"

augment = False

# ==================================================
# Extract features and save to cache
# ==================================================

# Lists to hold features/labels
features = []
labels = []

noise_to_add = [0.0, 0.01, 0.02] if augment else [0.0]

print("Extracting features...")

# Open the CSV file containing the list of audio filenames
with open(csv_filename) as csvfile:
    reader = csv.reader(csvfile)
    # Skip the heading
    next(iter(reader))

    # Get each audio file and append to features/labels
    for i, (audiofile, label, _) in tqdm(enumerate(reader)):
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
        for n in noise_to_add:
            audio_features = extract_features(audio + n * np.random.randn(*audio.shape))

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

# Save the features/labels to the cache
print("Saving to cache...")
save_data_to_cache(features, labels)
