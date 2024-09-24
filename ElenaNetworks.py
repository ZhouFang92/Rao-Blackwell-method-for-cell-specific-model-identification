import torch 


class MLP(torch.nn.Module):

    def __init__(self, input_size, hidden_size, output_size, depth, activation=torch.nn.Tanh, postprocessing_layer=None):
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
        

class NeuralMartingale(torch.nn.Module):

    def __init__(self, X_encoder, Y_encoder, backbone):

        super(NeuralMartingale, self).__init__()

        self.X_encoder = X_encoder
        self.Y_encoder = Y_encoder

        self.backbone = backbone

        # TODO: add
        # noise_covariance, h_transform, g_function
        # add additional NNs

    def forward(self, t, XatT, Yslice):

        eX = self.X_encoder(XatT)
        eY = self.Y_encoder(Yslice)

        return self.backbone( torch.cat([t, eX, eY], dim=1) )


class RNNEncoder(torch.nn.Module):

    def __init__(self, input_size, embedding_size):

        super(RNNEncoder, self).__init__()

        self.input_size = input_size
        self.embedding_size = embedding_size

        self.RNN = torch.nn.LSTM(input_size, embedding_size, batch_first=True)

    def forward(self, x):

        return self.RNN(x)

