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

