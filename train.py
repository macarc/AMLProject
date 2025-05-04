"""
Train the model defined in model.py, saving the training and validation loss/accuracy.
"""

import datetime

import torch
from torch.utils.data import DataLoader
from tqdm import trange

from lib.accuracy import accuracy
from lib.audiodataset import AudioDataSet
from lib.helpers import get_label_weights, get_torch_backend
from lib.load_datasets import load_data
from lib.runs import save_run_data
from model import get_model_name, load_model, save_model
from plot_model_results import plot_accuracy, plot_run_data

# ==================================================
# Parameters
# ==================================================

# Number of iterations
n_epochs = 200

# Unique name per run
run_name = get_model_name() + datetime.datetime.now().strftime("__%H_%M_%S__%d_%m_%Y")

print(f"Starting training run {run_name}")

# ==================================================
# Setup
# ==================================================

# Load model
model, optimiser = load_model()

# Load dataset
(
    train_features,
    train_labels,
    val_features,
    val_labels,
    test_features,
    test_labels,
) = load_data()

train_dataset = AudioDataSet(train_features, train_labels)
train_dataloader = DataLoader(train_dataset, batch_size=50, shuffle=True)

# Initialise loss function
label_weights = torch.tensor(
    get_label_weights(train_labels), dtype=torch.float32, device=get_torch_backend()
)

loss_fcn = torch.nn.CrossEntropyLoss(weight=label_weights)

# Initialise lists to hold data during run
train_losses = []
val_losses = []
val_accuracies = []
train_accuracies = []

# ==================================================
# Training loop
# ==================================================

for n in trange(n_epochs):
    # Loop over all the batches
    for features, labels in train_dataloader:
        # Reset the optimiser
        optimiser.zero_grad()

        # Get the loss from the model
        model_output = model(features)
        loss = loss_fcn(model_output, labels.long())

        # Update the parameters based on the loss
        loss.backward()
        optimiser.step()

    # Append train loss over whole training set to train_losses
    train_output = model(train_features)
    loss = loss_fcn(train_output, train_labels.long())
    train_losses.append(loss.item())
    train_accuracies.append(100 * accuracy(train_output, train_labels))

    # Every 5th epoch, append loss/accuracy over whole validation set to val_losses/accuracies
    if n % 5 == 0:
        val_output = model(val_features)
        loss = loss_fcn(val_output, val_labels.long())
        val_losses.append(loss.item())
        val_accuracies.append(100 * accuracy(val_output, val_labels))

# ==================================================
# Result processing
# ==================================================

# Save model and run data
save_model()
save_run_data(run_name, train_losses, train_accuracies, val_losses, val_accuracies)

# Display results
plot_run_data(run_name)
plot_accuracy()
