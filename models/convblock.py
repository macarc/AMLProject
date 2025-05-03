import torch.nn as nn


class ConvBlock(nn.Module):
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
