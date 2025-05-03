import torch

from features import melspect, mfccs
from lib.helpers import get_torch_backend
from lib.labels import label_count
from models import convnet_with_network, resnet

# ==================================================
# Model definition - adjust parameters here!
#
# Note that you must run the extract_features script after
# changing the _extract_features variable
# ==================================================

_model = convnet_with_network.ConvNet(
    [20, 26, 28], [3, 3, 3], [28, 32, label_count()]
)
_extract_features = mfccs.extract_features
_optimiser = torch.optim.Adam(_model.parameters())

# ==================================================
# Set model device to match hardware
# ==================================================

_model.to(get_torch_backend(True))

# ==================================================
# Public functions
# ==================================================


def load_model():
    """
    Gets the model - either loads from file (if it has been trained), or uses a new model.

    Returns:
        model (torch.nn.Module): the model
        optimiser (torch.optim.Optimiser): the optimiser
    """
    try:
        state = torch.load(_filename())
        _model.load_state_dict(state["model"])
        _optimiser.load_state_dict(state["optimiser"])
        _model.eval()
        print("Loaded model from file")
        return _model, _optimiser

    except FileNotFoundError:
        print("Using a new model")
        return _model, _optimiser


def save_model():
    """Save model/optimiser state to file, so that it can be loaded again later."""
    state = {
        "model": _model.state_dict(),
        "optimiser": _optimiser.state_dict(),
    }
    torch.save(state, _filename())


def get_model_name():
    return str(_model)


def _filename():
    return f"trained_models/{get_model_name()}.pt"


def extract_features(audiodata):
    return _extract_features(audiodata)
