import torch

class MLP(torch.nn.Module):

    def __init__(self, input_size, hidden_size, output_size, depth, activation=torch.nn.Tanh, postprocessing_layer=None):
        """
        Classical MLP implementation with a postprocessing layer.

        Args:
            input_size: size of the input layer.
            hidden_size: size of the hidden layer.
            output_size: size of the output layer.
            depth: depth of the network.
            activation: The activation layer to use . Defaults to torch.nn.Tanh.
            postprocessing_layer: The postprocessing layer to use (for example, softmax). Defaults to None.
        """
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.depth = depth
        self.activation = activation

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            self.activation(),
            *[torch.nn.Sequential(
                torch.nn.Linear(hidden_size, hidden_size),
                self.activation()
            ) for _ in range(depth-1)],
            torch.nn.Linear(hidden_size, output_size),
        )

        self.postprocessing_layer = postprocessing_layer

    def forward(self, x):

        x = self.layers(x)

        if self.postprocessing_layer is not None:
            x = self.postprocessing_layer(x)

        return x

class RNNEncoder(torch.nn.Module):

    def __init__(self, input_size, embedding_size, activation=None):

        super(RNNEncoder, self).__init__()

        self.input_size = input_size
        self.embedding_size = embedding_size

        self.RNN = torch.nn.LSTM(input_size, embedding_size, batch_first=True)
        if activation is not None:
            self.activation = activation()
        else:
            self.activation = None

    def forward(self, x):
        out = self.RNN(x)[0][:, -1, :]
        if self.activation is not None:
            out = self.activation(out)
        return out


