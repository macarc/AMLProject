"""
Functions for calculating the accuracy of the model.
"""

import numpy as np
import torch

from lib.labels import label_count


def accuracy(output, targets, top_n=5):
    """
    Get model accuracy - the percentage of outputs where
    the correct class is in the top top_n predictions

    e.g. if top_n = 5, then this is the percentage of model outputs
         that put the correct label in the top 5

    Parameters:
        output (torch.Tensor): model predictions
        targets (torch.Tensor): target labels
        top_n (int): number of top predictions considered when calculating the accuracy
    Returns:
        accuracy (float): accuracy of the model
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


def class_accuracy(output, targets, top_n=5):
    """
    Get model accuracy per class - the percentage of outputs where
    the correct class is in the top top_n predictions, grouped by target class

    e.g. if top_n = 5, then this is the percentage of model outputs
         that put the correct label in the top 5

    Parameters:
        output (torch.Tensor): model predictions
        targets (torch.Tensor): target labels
        top_n (int): number of top predictions considered when calculating the accuracy
    Returns:
        class_accuracies (numpy float array):
            class accuracies where the i'th element is
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

    # For each output label
    for p in range(len(output)):
        label_class = int(targets[p])
        for i in range(top_n):
            # If the label is in the top_n predicted, add 1 to the correct prediction count for the class
            if top_n_predictions[p, i].item() == targets[p]:
                correct_counts[label_class] += 1
                break
        else:
            # Otherwise, add 1 to the incorrect prediction count for the class
            incorrect_counts[label_class] += 1

    return correct_counts / (incorrect_counts + correct_counts)
