import constants
import numpy as np
import progressbar


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
