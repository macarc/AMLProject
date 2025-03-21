# Implement a convolutional network
# For me, this achieves over 95% accuracy on the training set,
# but only ~55% on the validation/test sets (where "success" is the correct label
# being in the top 5 predicted labels). This suggests the model doesn't generalise very well.
#
# Things to try:
# - different model (fewer parameters perhaps, if it's overfitting?)
# - better feature extraction (currently only taking the first two seconds... is that representative
#   of the whole sample?)

# %% IMPORTS

from audiodataset import AudioDataSet
import constants
from helpers import (
    adjust_length,
    load_model,
    save_model,
    get_torch_backend,
    get_label_weights,
)
from labels import label_count, number_to_label
import librosa
from load_datasets import load_data_to_device
import numpy as np
from tqdm import trange
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import sklearn


# %% FEATURE EXTRACTION


def extract_features(audio_data):
    """Extract a mel spectrogram of the audio data"""
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

    # Extract a mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio_data, sr=constants.SAMPLE_RATE
    )

    return mel_spectrogram


# %% CONVOLUTIONAL NETWORK


class ConvBlock(torch.nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size):
        super(ConvBlock, self).__init__()

        # Define network block layers
        self.lin = nn.Conv1d(input_channels, output_channels, kernel_size)
        self.act = nn.ReLU()

        # Save number of input/output channels for validation later
        self.n_input_channels = input_channels
        self.n_output_channels = output_channels

    def forward(self, x):
        """
        Forward pass of the Convolutional block.

        x : numpy array with shape (N, C, T)
            where N is the number of data points,
                  C is the number of channels (must match the ConvBlock's input_channels)
                  T is the number of timesteps

        Returns:
        block_output : numpy array with shape (N, Cout, Tout)
            where N is the number of data points,
                  Cout is the number of output channels
                  Tout is the number of output timesteps
        """

        # Check input has the right number of channels
        assert x.shape[1] == self.n_input_channels

        # Run model
        block_output = self.act(self.lin(x))

        # Check output has the right number of channels
        assert block_output.shape[1] == self.n_output_channels

        return block_output


class ConvNet(torch.nn.Module):
    def __init__(self, layers, out_channels, kernel_size):
        super(ConvNet, self).__init__()

        # The net consists of 8 blocks
        # nn.Sequential simply takes the output of the last block and feeds it into the next
        # which makes ConvNet's forward() definition less tedious
        self.blocks = nn.Sequential()

        for last_channels, this_channels in zip(layers, layers[1:]):
            self.blocks.add_module(
                f"layer{last_channels}",
                ConvBlock(last_channels, this_channels, kernel_size),
            )

        # Linear layer gives the output
        self.lin = nn.Linear(layers[-1], out_channels)

        # Save for later validation
        self.n_input_channels = layers[0]
        self.n_hidden_channels = len(self.blocks)
        self.kernel_size = kernel_size

    def forward(self, x):
        """
        Forward pass of the Convolutional Network, to predict digits

        x : numpy array with shape (N, C, T)
            where N is the number of data points,
                  C is the number of channels (must match the ConvNet's input_channels)
                  T is the number of timesteps

        Returns:
        block_output : numpy array with shape (N, 10)
            where N is the number of data points,
            with block_output[, ki] corresponding to how 'certain' the net is that
             the kth audio sample is digit i (though not normalised between 0 and 1!)
        """
        # Get the number of data points and the length of the sequence
        N = x.shape[0]
        T = x.shape[2]

        # Check that the number of channels matches the number of input channels for the net
        assert x.shape == torch.Size([N, self.n_input_channels, T])

        # Get output of 8 ConvBlocks
        block_output = self.blocks(x)

        # Check the number of data points and channels are as expected
        assert block_output.shape[0] == N
        # assert block_output.shape[1] == self.n_hidden_channels

        # Perform global average pooling by averaging over the time dimension
        pooled = block_output.mean(dim=2).squeeze()

        # Check that the shape of the pooled averages matches the output shape
        # assert pooled.shape == torch.Size([N, self.n_hidden_channels])

        # Finally, apply a linear layer to the pooled output
        network_output = self.lin(pooled)

        return network_output


# %% ACCURACY CALCULATIONS


def accuracy(output, targets, top_n=5):
    """
    Get model accuracy - the percentage of outputs where
    the correct label is in the top top_n predictions

    e.g. if top_n = 5, then this is the percentage of model outputs
         that put the correct label in the top 5

    Doing the "top_n" predictions helps deal with similar labels.
    """
    assert len(targets.shape) == 1
    assert output.shape == torch.Size([len(targets), label_count()])

    # Convert targets to numpy
    targets = targets.cpu().numpy()

    # Get the top top_n predictions
    top_n_predictions = np.argpartition(
        output.detach().cpu().numpy(), label_count() - top_n, axis=1
    )[:, -top_n:]
    assert top_n_predictions.shape == torch.Size([len(targets), top_n])

    # After the loop, predictions[k] is 1 only if top_n_predictions[k, :] includes targets[k]
    predictions = np.zeros(len(targets))
    for i in range(top_n):
        predictions += top_n_predictions[:, i] == targets

    # Count the number of correct predictions
    num_correct = np.sum(predictions)

    # Get accuracy
    accuracy = num_correct / len(predictions)

    return accuracy


def label_accuracy(output, targets, top_n=5):
    """
    Get model accuracy per label - the percentage of outputs where
    the correct label is in the top top_n predictions, grouped by label

    e.g. if top_n = 5, then this is the percentage of model outputs
         that put the correct label in the top 5

    Doing the "top_n" predictions helps deal with similar labels.

    Returns: numpy array where the i'th element is
            (#correct predictions of label i / (#total times label i appears in target))
    """
    assert len(targets.shape) == 1
    assert output.shape == torch.Size([len(targets), label_count()])

    # Convert targets to numpy
    targets = targets.cpu().numpy()

    # Get the top top_n predictions
    top_n_predictions = np.argpartition(
        output.detach().cpu().numpy(), label_count() - top_n, axis=1
    )[:, -top_n:]
    assert top_n_predictions.shape == torch.Size([len(targets), top_n])

    # Number of correctly/incorrectly predicted datapoints per label
    correct_counts = np.zeros(label_count())
    incorrect_counts = np.zeros(label_count())

    for p in range(len(output)):
        for i in range(top_n):
            if top_n_predictions[p, i].item() == targets[p]:
                correct_counts[int(targets[p])] += 1
                break
        else:
            incorrect_counts[int(targets[p])] += 1

    return correct_counts / (incorrect_counts + correct_counts)


# %% MODEL TRAINING
if __name__ == "__main__":
    # %% TORCH BACKEND

    backend_dev = get_torch_backend()

    # %% LOAD FEATURES

    # NB: pass force_reload = True here when extract_features has changed!
    (
        train_features,
        train_labels,
        val_features,
        val_labels,
        test_features,
        test_labels,
    ) = load_data_to_device(backend_dev, extract_features, force_reload=True)

    train_dataset = AudioDataSet(train_features, train_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=50, shuffle=True)

    # %% INITIALISATION

    train_losses = []
    val_losses = []
    val_accuracies = []
    train_accuracies = []

    # %% MODEL STORAGE FILENAME

    # needs to be changed per model
    model_filename = "models/conv.pt"

    # %% LOAD MODEL

    nnet = ConvNet([128, 64, 64, 64, 70], label_count(), 5)
    optimiser = torch.optim.Adam(nnet.parameters())
    nnet.to(backend_dev)

    if load_model(model_filename, nnet, optimiser):
        print("Loaded model from file!")

    label_weights = torch.tensor(
        get_label_weights(train_labels), dtype=torch.float32, device=backend_dev
    )
    loss_fcn = torch.nn.CrossEntropyLoss(weight=label_weights)

    # %% TRAINING LOOP

    n_epochs = 100

    for n in trange(n_epochs):
        # Loop over all the batches
        for features, labels in train_dataloader:
            # Reset the optimiser
            optimiser.zero_grad()

            # Get the loss from the model
            model_output = nnet(features)
            loss = loss_fcn(model_output, labels.long())

            # Update the parameters based on the loss
            loss.backward()
            optimiser.step()

        # Append train loss over whole training set to train_losses
        train_output = nnet(train_features)
        loss = loss_fcn(train_output, train_labels.long())
        train_losses.append(loss.item())
        train_accuracies.append(100 * accuracy(train_output, train_labels))

        # Every 5th epoch, append loss/accuracy over whole validation set to val_losses/accuracies
        if n % 5 == 0:
            val_output = nnet(val_features)
            loss = loss_fcn(val_output, val_labels.long())
            val_losses.append(loss.item())
            val_accuracies.append(100 * accuracy(val_output, val_labels))

    # %% SAVE MODEL

    save_model(model_filename, nnet, optimiser)

    # %% DISPLAY RESULTS

    train_output = nnet(train_features)
    train_accuracy = accuracy(train_output, train_labels)
    print(f"Final training accuracy: {train_accuracy}")

    val_output = nnet(val_features)
    val_accuracy = accuracy(val_output, val_labels)
    print(f"Validation accuracy: {val_accuracy}")

    test_output = nnet(test_features)
    test_accuracy = accuracy(test_output, test_labels)
    print(f"Test accuracy: {test_accuracy}")

    plt.plot(train_accuracies)
    plt.title("Training Accuracy")
    plt.show()

    plt.plot(val_accuracies)
    plt.title("Validation Accuracy")
    plt.show()

    plt.plot(train_losses)
    plt.title("Training Loss")
    plt.show()

    label_names = [number_to_label(i) for i in range(label_count())]

    plt.xticks(rotation="vertical", fontsize=5)
    plt.bar(label_names, label_accuracy(train_output, train_labels))
    plt.title("Training Accuracy per label")
    plt.show()

    plt.xticks(rotation="vertical", fontsize=5)
    plt.bar(label_names, label_accuracy(val_output, val_labels))
    plt.title("Validation Accuracy per label")
    plt.show()

    plt.xticks(rotation="vertical", fontsize=5)
    plt.bar(label_names, label_accuracy(test_output, test_labels))
    plt.title("Test Accuracy per label")
    plt.show()
