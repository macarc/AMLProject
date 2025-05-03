import librosa
import numpy as np

from lib import constants
from lib.helpers import adjust_length


def extract_features(audio_data):
    """
    Extract MFCCs from audio data.

    Parameters:
        audio_data (numpy.ndarray)
    Returns:
        features (numpy.ndarray or None):
    """
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

    # Extract MFCCs
    features = librosa.feature.mfcc(y=audio_data, sr=constants.SAMPLE_RATE)

    return features
