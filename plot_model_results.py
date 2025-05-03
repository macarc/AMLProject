"""
Plot the results of a training run.
"""

import matplotlib.pyplot as plt
import numpy as np

from lib.accuracy import accuracy, class_accuracy
from lib.labels import label_count, number_to_label
from lib.load_datasets import load_data
from lib.runs import load_run_data
from model import load_model

# ==================================================
# Parameters
# ==================================================

run_name = "ConvNet__20_26_28__28_32_70__3_3_3__13_02_45__30_04_2025"

# ==================================================
# Plotting code
# ==================================================


def plot_run_data(filename):
    train_losses, train_accuracies, val_losses, val_accuracies = load_run_data(filename)

    assert len(train_losses) == len(train_accuracies)
    assert len(val_losses) == len(val_accuracies)

    Niters = len(train_losses)

    train_iters = np.arange(0, len(train_losses))
    val_iters = 5 * np.arange(0, len(val_losses))

    plt.plot(train_iters, train_accuracies)
    plt.title("Training Accuracy")
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy (%)")
    plt.xlim(0, Niters)
    plt.ylim(0, 100)
    plt.savefig(f"graphs/{filename}_accuracy_train.png")
    plt.show()

    plt.plot(val_iters, val_accuracies)
    plt.title("Validation Accuracy")
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy (%)")
    plt.xlim(0, Niters)
    plt.ylim(0, 100)
    plt.savefig(f"graphs/{filename}_accuracy_val.png")
    plt.show()

    plt.plot(train_iters, train_accuracies)
    plt.plot(val_iters, val_accuracies)
    plt.legend(["Training Accuracy", "Validation Accuracy"], loc="lower right")
    plt.title("Training and Validation Accuracy")
    plt.xlim(0, Niters)
    plt.ylim(0, 100)
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy (%)")
    plt.savefig(f"graphs/{filename}_accuracy_both.png")
    plt.show()

    plt.plot(train_iters, train_losses)
    plt.title("Training Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.xlim(0, Niters)
    plt.savefig(f"graphs/{filename}_loss_train.png")
    plt.show()

    plt.plot(val_iters, val_losses)
    plt.title("Validation Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.xlim(0, Niters)
    plt.savefig(f"graphs/{filename}_loss_val.png")
    plt.show()

    plt.plot(train_iters, train_losses)
    plt.plot(val_iters, val_losses)
    plt.legend(["Training Loss", "Validation Loss"], loc="upper right")
    plt.title("Training and Validation Loss")
    plt.xlim(0, Niters)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.savefig(f"graphs/{filename}_loss_both.png")
    plt.show()


def plot_accuracy():
    (
        train_features,
        train_labels,
        val_features,
        val_labels,
        test_features,
        test_labels,
    ) = load_data()
    model, _ = load_model()

    train_output = model(train_features)
    train_accuracy = accuracy(train_output, train_labels)
    print(f"Final training accuracy: {train_accuracy}")

    val_output = model(val_features)
    val_accuracy = accuracy(val_output, val_labels)
    print(f"Validation accuracy: {val_accuracy}")

    test_output = model(test_features)
    test_accuracy = accuracy(test_output, test_labels)
    print(f"Test accuracy: {test_accuracy}")

    label_names = [number_to_label(i) for i in range(label_count())]

    plt.xticks(rotation="vertical", fontsize=5)
    plt.bar(label_names, class_accuracy(train_output, train_labels))
    plt.title("Training Accuracy per label")
    plt.xlabel("Label")
    plt.ylabel("Accuracy (%)")
    plt.show()

    plt.xticks(rotation="vertical", fontsize=5)
    plt.bar(label_names, class_accuracy(val_output, val_labels))
    plt.title("Validation Accuracy per label")
    plt.xlabel("Label")
    plt.ylabel("Accuracy (%)")
    plt.show()

    plt.xticks(rotation="vertical", fontsize=5)
    plt.bar(label_names, class_accuracy(test_output, test_labels))
    plt.title("Test Accuracy per label")
    plt.xlabel("Label")
    plt.ylabel("Accuracy (%)")
    plt.show()


# ==================================================
# Script
# ==================================================

if __name__ == "__main__":
    if run_name:
        plot_run_data(run_name)

    plot_accuracy()
