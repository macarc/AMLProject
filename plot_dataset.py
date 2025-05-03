"""
Plot dataset statistics.
"""

import matplotlib
import matplotlib.pyplot as plt
import torch

from lib.labels import label_count, number_to_label
from lib.load_datasets import load_data


def plot_label_frequencies(labels, title):
    """Plot frequencies of each label in the dataset."""
    labels_and_counts = [
        (i, torch.sum(labels == i).item()) for i in range(label_count())
    ]
    min_label, min_count = min(
        labels_and_counts,
        key=lambda a: a[1],
    )
    print(
        f"Label with the least training examples is '{number_to_label(min_label)}' with {min_count} instances"
    )

    label_names = [number_to_label(i)[-20:] for i, _ in labels_and_counts]
    label_counts = [count for _, count in labels_and_counts]
    print(label_names)

    matplotlib.rc("font", size=18)
    plt.bar(label_names, label_counts, align="edge")
    plt.xticks(rotation=90, fontsize=5)
    plt.xlim(-0.3, label_count())
    plt.ylabel("Number of occurrences")
    plt.xlabel("Class")
    plt.title(title)
    plt.tight_layout()
    plt.show()


(
    train_features,
    train_labels,
    val_features,
    val_labels,
    test_features,
    test_labels,
) = load_data()

plot_label_frequencies(train_labels, "Train labels")
plot_label_frequencies(val_labels, "Val labels")
plot_label_frequencies(test_labels, "Test labels")

plot_label_frequencies(
    torch.concat((train_labels, val_labels, test_labels)), "Class frequencies"
)
