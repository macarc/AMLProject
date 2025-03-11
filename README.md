# AMLProject

## Project Proposal

Stage 1 would be to train a classifier that takes a sound effect and returns the ‘type’ of sound effect.

Stage 2 would be to take a text label (convert to a vector using word2vec) and a sound effect, and return a probability that the sound effect matches this label. This would allow selecting a sound effect from a (unlabelled) library, by running the model on each sound effect and choosing the one with highest probability.

Stage 3 (our stretch goal) would be to take a text label and generate a sound effect. Whether or not this is feasible will depend on what direction the course takes.

For all 3 of these, data is easy to obtain, as we can just use any labelled sound effect library. For stage 1, ARCA23K (https://zenodo.org/records/5117901) will provide suitable labelled sounds.

## Usage

### Installing dependencies

Install dependencies with `pip install -r requirements.txt`.

These can be installed globally with pip, or [using a virtual environment](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/).

### Downloading datasets

`download_fsd50k.py` downloads the FSD50K dataset. This requires manually extracting a ZIP file at the end. There is ~30GB uncompressed data, so it may take a while to download!

### Loading datasets

`load_datasets.py` can be used as a library, exposing `load_dev_data` and `load_eval_data`. By default, these functions cache the loaded numpy arrays. To reload the cache (e.g. if `constants.AUDIO_LENGTH` has changed), either pass `force_reload = True`, or run the file as a script: `python load_datasets.py`.

### Code formatting

If [black](https://black.readthedocs.io/en/stable/) is installed, run `black .` to format.
