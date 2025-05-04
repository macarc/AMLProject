import torch
import torch.nn as nn

from models.convblock import ConvBlock
from models.resblock import ResBlock

# ==================================================
# Network definition
# ==================================================


class ResNet(nn.Module):
    def __init__(
        self,
        res_layer_count,
        res_kernel_sizes,
        convolutional_layers,
        kernel_sizes,
        net_layers,
    ):
        """
        Parameters:
            convolutional_layers (list): list of ConvBlock layer sizes
            kernel_sizes (list): list of kernel sizes
            net_layers (list): list of network (after pooling) layer sizes
        """
        super(ResNet, self).__init__()

        # nn.Sequential simply takes the output of the last block and feeds it into the next
        self.resblocks = nn.Sequential()
        for res_layer in range(res_layer_count):
            self.resblocks.add_module(
                f"resblock_{res_layer}",
                ResBlock(convolutional_layers[0], res_kernel_sizes[res_layer]),
            )

        self.convblocks = nn.Sequential()

        # Add convolutional blocks
        for last_channels, this_channels, kernel_size in zip(
            convolutional_layers, convolutional_layers[1:], kernel_sizes
        ):
            self.convblocks.add_module(
                f"convblock_{last_channels}_{this_channels}",
                ConvBlock(last_channels, this_channels, kernel_size),
            )

        self.net = nn.Sequential()
        for i, (last_layer_size, this_layer_size) in enumerate(
            zip(net_layers, net_layers[1:])
        ):
            self.net.add_module(
                f"layer{last_layer_size}", nn.Linear(last_layer_size, this_layer_size)
            )
            # Add activation to all layers but the last
            if i < len(net_layers) - 2:
                self.net.add_module(f"act{i}", nn.ReLU())

        # Save for later validation
        self.n_conv_input_channels = convolutional_layers[0]
        self.n_net_input_channels = net_layers[0]
        self.name = f"ResNet__{res_layer_count}__{"_".join(map(str, res_kernel_sizes))}__{"_".join(map(str, convolutional_layers))}__{"_".join(map(str, net_layers))}__{"_".join(map(str, kernel_sizes))}"

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
        block_output = self.convblocks(self.resblocks(x) + x)

        # Check the number of data points is as expected
        assert block_output.shape[0] == N
        assert block_output.shape[1] == self.n_net_input_channels

        # Perform global average pooling by averaging over the time dimension
        pooled = block_output.mean(dim=2).squeeze()

        # Finally, apply normal network to the pooled output
        network_output = self.net(pooled)

        return network_output
