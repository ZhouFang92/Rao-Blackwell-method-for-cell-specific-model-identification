import torch 
from copy import deepcopy

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

    def __init__(self, X_encoder, Y_encoder, backbone, likelihood_function, likelihood_function_parameters, h_transform, g_function, stochiometry_matrix, discretization_size, output_sampling_size, times_tau, times_t, centered_pp):

        super(NeuralMartingale, self).__init__()

        self.X_encoder = X_encoder
        self.Y_encoder = Y_encoder

        self.backbone = backbone

        # TODO: add
        # noise_covariance, h_transform, g_function
        # add additional NNs
        # add sotch matrix
        self.likelihood_function = likelihood_function
        self.likelihood_function_parameters = likelihood_function_parameters
        self.h_transform = h_transform
        self.g_function = g_function
        self.stochiometry_matrix = stochiometry_matrix

        # maybe
        self.discretization_size = discretization_size # this is the number points between t0 and t1 excluded. 
        self.M_bar = discretization_size + 1
        self.output_sampling_size = output_sampling_size

        self.times_tau = times_tau
        self.times_t = times_t

        self.centered_pp = centered_pp
        self.centered_pp_delta = self.centered_pp[1:] - self.centered_pp[:-1]


    def G_n_minus_1(self, t, Xslice, Yslice):
        return self.likelihood_function(Yslice[:, -1, :], Xslice, self.h_transform, self.parameters)*self.g_function(Xslice)
    
    def G(self, t, Xslice, Yslice):
        return self.likelihood_function(Yslice[:, 0, :], Xslice, self.h_transform, self.parameters)*self.forward(t, Xslice, Yslice[:, 1, :])
    
    def stochastic_integral(self, t_k, k, X, Y):
        # precompute Yslice
        # --- the code was checked --- 
        Yslice = Y[:, k+1:, :]
        integral = 0.0
        # naive implementation
        for j in range(self.stochiometry_matrix.shape[1]):
            for i in range(self.M_bar):
                Xslice = X[:, k*(self.M_bar)+i, :]
                current_time = self.times_tau[i] + t_k 
                # before and after displacement
                nn_before = self.forward(current_time, Xslice, Yslice)
                nn_after = self.forward(current_time, Xslice + self.stochiometry_matrix[:, j], Yslice)
                nn_delta = nn_after - nn_before
                integral += nn_delta * self.centered_pp_delta[i]

        return integral



    def loss(self, X, Y, k):
        # TODO completare, inoltre ricorda che hai diviso l in time e trajectory indexes. 
        # TODO calcolare la computational complexity del codice per il training (operazioni per epoch)

        loss = 0.0

        for measurement_index in range(self.times_t.shape[0]): # n * p operations p is the number of intervals and n the size of the batch

            # indexes on X
            tau_index_abs_begin = measurement_index * self.M_bar 
            tau_index_abs_end = tau_index_abs_begin + self.M_bar



            if  k == self.times_t.shape[0] - 2:
                # don't loose the dimensionality of Y
                loss += self.G_n_minus_1(self.times_t[k+1], X[:, tau_index_abs_end, :], Y[:, k+1:, :]) - self.forward(self.times_t[k+1], X[:, tau_index_abs_end, :], Y[:, k+1:, :]) 
            else:
                loss += self.G(self.times_t[k+1], X[:, k+1, :], Y[:, k:k+1, :]) - self.forward(self.times_t[k], X[:, k, :], Y[:, k:k+1, :]) + self.stochastic_integral(self.times_t[k], k, X, Y)

        return loss
            
        

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


class NeuralMaringaleChain(torch.nn.Module):

    def __init__(self, N, neural_martingale):
        super(NeuralMaringaleChain, self).__init__()
        self.N = N
        # create N NeuralMartingale objects
        self.chain = torch.nn.ModuleList([deepcopy(neural_martingale) for _ in range(N)])

    def forward(self, t, XatT, Yslice, i):    
        return self.chain[i](t, XatT, Yslice)
    
