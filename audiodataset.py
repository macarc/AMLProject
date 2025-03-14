import torch
from torch.utils.data import Dataset

class AudioDataSet(Dataset):
    def __init__(self, features, labels):
        # Save features and labels
        self.features = features
        self.labels = labels

        # Save the number of datapoints, number of channels in each datapoint, and length of each datapoint
        self.N, self.C, self.T = self.features.shape

        # Sanity check!
        assert self.features.shape == torch.Size([self.N, self.C, self.T])
        assert self.labels.shape == torch.Size([self.N])

    def __getitem__(self, i):
        """
        Get the i'th data point in the dataset

        Returns:
        features - numpy array
        label - number
        """
        # Ensure that i is within bounds
        assert 0 <= i < self.N

        # Return feature and label
        return self.features[i], self.labels[i]

    def __len__(self):
        """The number of points in the dataset"""
        return self.N
