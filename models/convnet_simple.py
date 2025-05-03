import torch
import torch.nn as nn
from convblock import ConvBlock

# ==================================================
# Network definition
# ==================================================


class ConvNetSimple(torch.nn.Module):
    def __init__(self, convolutional_layers, kernel_sizes):
        """
        Parameters:
            convolutional_layers (list): list of ConvBlock layer sizes
            kernel_sizes (list): list of kernel sizes
            net_layers (list): list of network (after pooling) layer sizes
        """
        super(ConvNetSimple, self).__init__()

        # nn.Sequential simply takes the output of the last block and feeds it into the next
        self.convblocks = nn.Sequential()

        # Add convolutional blocks
        for in_channels, out_channels, kernel_size in zip(
            convolutional_layers, convolutional_layers[1:], kernel_sizes
        ):
            self.convblocks.add_module(
                f"convblock_{in_channels}_{out_channels}",
                ConvBlock(in_channels, out_channels, kernel_size),
            )

        n_output_channels = convolutional_layers[-1]

        # Linear layer applied to pooled output
        self.lin = nn.Linear(n_output_channels, n_output_channels)

        # Save for later validation
        self.n_conv_input_channels = convolutional_layers[0]
        self.n_output_channels = n_output_channels
        self.name = f"ConvNetSimple__{"_".join(map(str, convolutional_layers))}__{"_".join(map(str, net_layers))}__{"_".join(map(str, kernel_sizes))}"

    def __str__(self):
        return self.name

    def forward(self, x):
        """
        Forward pass of the Convolutional Network, to predict digits

        Parameters:
            x (numpy.ndarray): input data with shape (N, C, T)
                where N is the number of data points,
                      C is the number of channels (must match the ConvNet's input_channels)
                      T is the number of timesteps

        Returns:
            block_output (numpy.ndarray): output data with shape (N, Cout)
                where N is the number of data points and Cout is the last layer output in the neural net,
                with block_output[, ki] corresponding to how 'certain' the net is that
                 the kth audio sample is digit i (though not normalised between 0 and 1!)
        """
        # Get the number of data points and the length of the sequence
        N = x.shape[0]
        T = x.shape[2]

        # Check that the number of channels matches the number of input channels for the net
        assert x.shape == torch.Size([N, self.n_conv_input_channels, T])

        # Get output of convolutional network
        block_output = self.convblocks(x)

        # Check the number of data points is as expected
        assert block_output.shape[0] == N
        assert block_output.shape[1] == self.n_output_channels

        # Perform global average pooling by averaging over the time dimension
        pooled = block_output.mean(dim=2).squeeze()

        # Finally, apply linear layer
        network_output = self.lin(pooled)

        return network_output
