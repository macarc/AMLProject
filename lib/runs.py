"""
Functions for loading/saving individual run data.

Running `python train.py` stores the run information using the save_run_data() function defined here.

Other scripts (such as plot_data) then load the run information using load_run_data().
"""

import pickle


def save_run_data(run_name, train_losses, train_accuracies, val_losses, val_accuracies):
    """
    Save run data to disk.

    Parameters:
        run_name (str): unique identifier for the run
        train_losses, train_accuracies, val_losses, val_accuracies (list): losses/accuracies during training
    """
    with open(f"runs/{run_name}.pickle", "wb") as f:
        pickle.dump([train_losses, train_accuracies, val_losses, val_accuracies], f)


def load_run_data(run_name):
    """
    Load run data from disk.

    Parameters:
            run_name (str): unique identifier for the run
    Returns:
        train_losses, train_accuracies, val_losses, val_accuracies (list): losses/accuracies during training
    """
    with open(f"runs/{run_name}.pickle", "rb") as f:
        return pickle.load(f)
