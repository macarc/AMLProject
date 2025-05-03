# Audio Machine Learning Project - Usage

## Installing dependencies

Install dependencies with `pip install -r requirements.txt`.

These can be installed globally with pip, or [using a virtual environment](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/).

## Loading the data

- `python download_fsd50k.py` to download datasets
- `python extract_features.py` to load the features and cache them

## Training the model

First, make sure that you've created directories called `trained_models`, `graphs` and `runs`. Then run:

- `python train.py` to train the model
- `python plot_model_results.py` to see the results of a run
- `python gui.py` to use the model in an example GUI application

## Modifying the model
Edit the model defined in `model.py`, and then run `extract_features.py` (if the features have changed) and `train.py` again.

## Code formatting

If [black](https://black.readthedocs.io/en/stable/) is installed, run `black .` to format.

If [isort](https://pycqa.github.io/isort/) is installed, run `isort .` to sort imports.
