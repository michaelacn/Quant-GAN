import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class TemporalBlock(nn.Module):
    """A Temporal Block module for Temporal Convolutional Networks (TCNs).

    This block consists of two 1D convolutional layers with optional downsampling
    and residual connections. It is designed to capture temporal dependencies in
    sequential data.

    Args:
        n_inputs (int): Number of input channels (features) to the block.
        n_hidden (int): Number of hidden units (channels) in the intermediate layer.
        n_outputs (int): Number of output channels (features) from the block.
        kernel_size (int): Size of the convolutional kernel along the temporal axis.
        dilation (int): Dilation factor for the convolutional layers. Controls the
                       spacing between kernel elements to capture long-range dependencies.

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, n_outputs, sequence_length).
    """
    def __init__(self, n_inputs, n_hidden, n_outputs, kernel_size, dilation):
        super(TemporalBlock, self).__init__()

        # First convolutional layer
        self.conv1 = nn.Conv1d(
            in_channels=n_inputs,
            out_channels=n_hidden,
            kernel_size=kernel_size,
            stride=1,
            dilation=dilation,
            padding='same'  # Ensures the output has the same length as the input
        )

        # Activation function after the first convolution
        self.relu1 = nn.PReLU() 

        # Second convolutional layer
        self.conv2 = nn.Conv1d(
            in_channels=n_hidden,
            out_channels=n_outputs,
            kernel_size=kernel_size,
            stride=1,
            dilation=dilation,
            padding='same' 
        )

        # Activation function after the second convolution
        self.relu2 = nn.PReLU()

        # Main network: Sequence of layers
        self.net = nn.Sequential(self.conv1, self.relu1, self.conv2, self.relu2)

        # Downsample layer (used if input and output channels differ)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        """Initialize weights for convolutional layers.

        Weights are initialized using a normal distribution with mean 0 and standard
        deviation 0.01. This helps stabilize training and avoid vanishing/exploding gradients.
        """
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        # Apply downsampling if necessary (to match input and output dimensions)
        res = x if self.downsample is None else self.downsample(x)

        return out + res


class TCN(nn.Module):
    """
    A Temporal Convolutional Network (TCN) for sequence modeling.

    This implementation stacks multiple causal convolutional blocks 
    (TemporalBlock) in a sequential manner. Each TemporalBlock can include 
    convolutions with increasing dilation factors to capture a broader 
    context in the time dimension.

    Args
    ----------
    input_size : int
        Number of channels (features) in the input sequence.
    output_size : int
        Number of output channels (features) for the final layer.
    n_hidden : int, optional
        Number of hidden channels used in each TemporalBlock. Default is 80.
    """
    def __init__(self, input_size, output_size, n_hidden=80):
        super(TCN, self).__init__()
        layers = []
        for i in range(7):
            num_inputs = input_size if i == 0 else n_hidden
            kernel_size = 2 if i > 0 else 1
            dilation = 2 * dilation if i > 1 else 1
            layers += [TemporalBlock(num_inputs, n_hidden, n_hidden, kernel_size, dilation)]
        self.conv = nn.Conv1d(n_hidden, output_size, 1)
        self.net = nn.Sequential(*layers)
        self.init_weights()

    def init_weights(self):
        self.conv.weight.data.normal_(0, 0.01)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_channels)
        y1 = self.net(x.transpose(1, 2))  # Now shape: (batch_size, n_hidden, seq_len)
        return self.conv(y1).transpose(1, 2)  # Final shape: (batch_size, seq_len, output_channels)


class Generator(nn.Module):
    """Generator: 3 to 1 Causal temporal convolutional network with skip connections.
       This network uses 1D convolutions in order to model multiple timeseries co-dependency.
    """ 
    def __init__(self, input_size, output_size):
        super(Generator, self).__init__()
        self.net = TCN(input_size, output_size)

    def forward(self, x):
        return torch.tanh(self.net(x))


class Discriminator(nn.Module):
    """Discrimnator: 1 to 1 Causal temporal convolutional network with skip connections.
       This network uses 1D convolutions in order to model multiple timeseries co-dependency.
    """ 
    def __init__(self, input_size, output_size):
        super(Discriminator, self).__init__()
        self.net = TCN(input_size, output_size)

    def forward(self, x):
        return torch.sigmoid(self.net(x))