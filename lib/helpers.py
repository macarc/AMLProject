import librosa
import numpy as np
import progressbar
import torch
from sklearn.utils import compute_class_weight

from lib import constants
from lib.labels import label_count


# Taken from https://stackoverflow.com/posts/53643011/revisions
class DownloadProgressBar:
    """Progress bar for urlretrieve, using the progressbar Python library"""

    def __init__(self):
        self.pbar = None

    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar = progressbar.ProgressBar(maxval=total_size)
            self.pbar.start()

        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(downloaded)
        else:
            self.pbar.finish()


def adjust_length(audio_data):
    """
    Resize audio_data to have length constants.AUDIO_LENGTH by either
    - truncating if audio_data is too long
    - appending zero-padding if audio-data is too long
    """

    # Ensure audio data has shape (N,) - i.e. it is mono
    assert len(audio_data.shape) == 1

    audio_data, _ = librosa.effects.trim(y=audio_data)

    if len(audio_data) <= constants.AUDIO_LENGTH:
        # If the audio data is too short, append zeros
        adjusted = np.concatenate(
            (audio_data, np.zeros((constants.AUDIO_LENGTH - len(audio_data),)))
        )
    else:
        # If the audio data is too long, truncate it
        adjusted = audio_data[: constants.AUDIO_LENGTH]

    # Check that we did indeed adjust the length correctly
    assert adjusted.shape == (constants.AUDIO_LENGTH,)

    return adjusted


def get_torch_backend(notify_user=False):
    """
    Get the most optimal torch backend - i.e. the device that the training runs on.
    Ideally, this is MPS/CUDA (which runs on GPU, and is ~20x faster!).
    Usage:
    backend_dev = get_torch_backend()
    data1 = torch.tensor(..., device=backend_dev) # load data on to GPU
    model = ...
    model.to(backend_dev) # convert model to run on GPU

    - notify_user : if True, print the backend that is being used.
    """
    if torch.backends.mps.is_available():
        if notify_user:
            print("Using MPS backend (fast!)")
        return torch.device("mps")
    elif torch.cuda.is_available():
        if notify_user:
            print("Using CUDA backend (fast!)")
        return torch.device("cuda")
    else:
        if notify_user:
            print("GPU not available, using CPU instead (slow...)")
        return torch.device("cpu")


def get_label_weights(labels):
    label_classes = np.arange(0, label_count())
    return compute_class_weight(
        class_weight="balanced", classes=label_classes, y=labels.cpu().numpy()
    )
