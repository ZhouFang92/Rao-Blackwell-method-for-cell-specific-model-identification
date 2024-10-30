from OtherNetworks import MLP 
import torch


class TemporalFeatureExtractor(torch.nn.Module):

    def __init__(self, n_features, T, device="cpu"):
        """
        Create a series of n temportal features from a single time point.
        This helps in conditioning the model on the time point.



        Args:
            n_features (int): number of temporal features to generate
            T (float): time point to generate the features from
            device (str, optional): The device for the layer's parameters. Defaults to "cpu".
        """
        # n_features is r in the paper 
        super(TemporalFeatureExtractor, self).__init__()
        self.n_features = n_features
        self.lambda_1 = torch.nn.parameter.Parameter(torch.ones(n_features).to(device), requires_grad=True)
        self.lambda_2 = torch.nn.parameter.Parameter(torch.ones(n_features).to(device), requires_grad=True)
        self.phi_1 = torch.nn.parameter.Parameter(torch.ones(n_features).to(device), requires_grad=True)
        self.device = device
        self.T = T

    def forward(self, t):
        """ 
        Generate the temporal features at time t.

        Args:
            t (float): time point to generate the features from

        Returns:
            torch.Tensor: The temporal features at time t. [shape [n_features]]

        The features have formulas:
        - lambda_1 * exp(T-t)
        - sin(lambda_2 * (T-t) + phi_1)
        """
        exp_vec = torch.exp(self.lambda_1 * (self.T-t))
        sin_vec = torch.sin(self.lambda_2 * (self.T-t) + self.phi_1)
        return torch.cat([exp_vec, sin_vec], dim=0)
    

class DeepCME(torch.nn.Module):
    """ 
    DeepCME implementation with vectorization and batching.

    Usage:
        Train the model on a specific initial condition X0 and a list of time points T.

        The output is the given by the value of self.Y which corresponds to:

        Y = E(g_1(X_T), ..., g_R(X_T)) 

        where X_T is the value of the state at time T. 
    """

    def __init__(self, backbone, time_list, g_functions, temporal_feature_extractor, R, K, device="cpu"):
        """ 
        Default constructor

        Args:
            backbone (torch.nn.Module): The backbone network V (an MLP in the original formulation)
            time_list (torch.Tensor): A list of time points to condition the model on
            g_functions (list): A list of functions to compute the g functions
            temporal_feature_extractor (TemporalFeatureExtractor): The temporal feature extractor
            R (int): The number of g_functions (output)
            K (int): The number of reactions
            device (str, optional): The device for the layer's parameters. Defaults to "cpu".
        """
        super(DeepCME, self).__init__()
        self.V = backbone
        self.time_list = time_list.to(device)
        # self.time_indexes = torch.arange(0, self.time_list.shape[0], 1).to(device)
        # add trainable parameter Y (baseline)
        self.Y = torch.nn.parameter.Parameter(torch.rand(R).to(device), requires_grad=True)
        
        self.g_functions = g_functions

        self.temporal_feature_extractor = temporal_feature_extractor

        self.R = R
        self.K = K
        self.device = device

        # use xavier initialization for the backbone
        # for m in self.V.modules():
        #     if isinstance(m, torch.nn.Linear):
        #         torch.nn.init.xavier_uniform_(m.weight)
        #         m.bias.data.fill_(0.01)

    def forward(self, t, X):
        """ 
        Forward pass of the model

        Args:
            t (float): The time point to condition the model on
            X (torch.Tensor): The input tensor [shape [batch_size, K]]

        Returns:
            torch.Tensor: The output of the model [shape [batch_size, R, K]]
        """
        current_time_features = self.temporal_feature_extractor(t).to(X.device).repeat(X.shape[0], 1)
        X_with_temporal_features = torch.cat([current_time_features, X], dim=1)
        out = self.V(X_with_temporal_features)
        # reshape the output to be of shape (batch_size, R, K)
        return out.view(-1, self.R, self.K)

    def phi(self, x, delta_threshold):
        """ 
        The phi function in the paper, so the quadratic to absolute value loss function.

        Args:
            x (torch.Tensor): The input tensor [shape [batch_size, R]]
            delta_threshold (torch.Tensor): The delta threshold tensor (normalization) [shape [R]]

        Returns:
            torch.Tensor: The output of the phi function [shape [batch_size, R]]
        """
        x = x/delta_threshold # [bs, R] / [R] = [bs, R]
        # we distingush f values below and above 1
        mask = torch.abs(x) <= 1
        X_upper = (x**2)*(mask)
        X_lower = (torch.abs(x)*2 - 1)*(~mask)
        # assert torch.all(X_upper >= 0)
        # assert torch.all(X_lower >= 0)
        out = (X_upper + X_lower)
        return out # [bs, R]

    def L(self, x, delta_threshold):
        """ 
        Computation of the inner part of the loss function

        Args:
            x (torch.Tensor): The input tensor [shape [batch_size, R]]
            delta_threshold (torch.Tensor): The delta threshold tensor (normalization) [shape [R]]

        Returns:
            torch.Tensor: The output of the L function [shape [batch_size]]
        """
        return self.phi(x, delta_threshold).sum(dim=-1)

    def g(self, X):
        """ 
        Compute the output (g) functions

        Args:
            X (torch.Tensor): The input tensor [shape [batch_size, K]]

        Returns:
            torch.Tensor: The output of the g functions [shape [batch_size, R]]

        Note: I don't think this is further vecorizable, anyway the number of g functions is small (at most 4, usually)
        """
        out = []
        for i in range(len(self.g_functions)):
            out.append(self.g_functions[i](X))
        return torch.stack(out, dim=1)
    
    def SI(self, X, R):
        """ 
        Compute the stochastic integral

        Args:
            X (torch.Tensor): The input tensor [shape [batch_size, T, K]]
            R (torch.Tensor): The output of the g functions [shape [batch_size, T, R]]

        Returns:
            torch.Tensor: The output of the stochastic integral [shape [batch_size, R]]
        """
        # -- iterative version
        # out_old = None
        # for j in range(1, self.time_list.shape[0]):
        #     if out_old is None:
        #         out_old = torch.bmm(self.forward(self.time_list[j-1], X[:, j-1]), (R[:, j] - R[:, j-1]).unsqueeze(-1)).squeeze(-1)
        #     else:
        #         out_old += torch.bmm(self.forward(self.time_list[j-1], X[:, j-1]), (R[:, j] - R[:, j-1]).unsqueeze(-1)).squeeze(-1) 
        
        # precompute the delta in the centered poisson process
        dR = R[:, 1:] - R[:, :-1]
        # compute the forward pass for all time points
        stacked_fwd = torch.vmap(lambda t, x : self.forward(t, x), in_dims=(0, 1))(self.time_list[:-1], X[:,:-1])
        # compute the product in the stochastic integral, then sum over time points
        out = torch.vmap(lambda v, dr: torch.bmm(v, dr.unsqueeze(-1)), in_dims=(0, 1))(stacked_fwd, dR).squeeze(-1).sum(dim=0)
        # return the sum
        return out


    def compute_delta_threshold(self, vals):
        """ 
        Compute the delta threshold for the loss function
        (this is a normalization factor)

        Args:
            vals (torch.Tensor): The input tensor [shape [batch_size, R]]

        Returns:
            torch.Tensor: The output of the delta threshold [shape [R]]

        Note that delta = 1 + mu + 2*sigma where mu is the mean and sigma is the standard deviation (for the considered batch!)
        """
        mu = torch.abs(vals.mean(dim=0))
        sigma = vals.std(dim=0)
        return 1 + mu + 2*sigma

    def loss(self, X, R):
        """ 
        Computation of the loss function

        Args:
            X (torch.Tensor): The input tensor [shape [batch_size, T, K]]
            R (torch.Tensor): The output of the g functions [shape [batch_size, T, R]]

        Returns:
            torch.Tensor: The output of the loss function [shape [batch_size]]
        """
        X_T = X[:, -1] # X at time T
        gX = self.g(X_T)
        inner = gX - self.SI(X, R) - self.Y
        delta_threshold = self.compute_delta_threshold(gX)
        return self.L(inner, delta_threshold).mean()


def phi(x, delta_threshold):
    """ 
    The phi function in the paper, so the quadratic to absolute value loss function.

    Args:
        x (torch.Tensor): The input tensor [shape [batch_size, R]]
        delta_threshold (torch.Tensor): The delta threshold tensor (normalization) [shape [R]]

    Returns:
        torch.Tensor: The output of the phi function [shape [batch_size, R]]
    """
    x = x/delta_threshold # [bs, R] / [R] = [bs, R]
    # we distingush f values below and above 1
    mask = torch.abs(x) <= 1
    X_upper = (x**2)*(mask)
    X_lower = (torch.abs(x)*2 - 1)*(~mask)
    # assert torch.all(X_upper >= 0)
    # assert torch.all(X_lower >= 0)
    out = (X_upper + X_lower)
    return out # [bs, R]

def compute_delta_threshold(vals):
    """ 
    Compute the delta threshold for the loss function
    (this is a normalization factor)

    Args:
        vals (torch.Tensor): The input tensor [shape [batch_size, R]]

    Returns:
        torch.Tensor: The output of the delta threshold [shape [R]]

    Note that delta = 1 + mu + 2*sigma where mu is the mean and sigma is the standard deviation (for the considered batch!)
    """
    mu = torch.abs(vals.mean(dim=0))
    sigma = vals.std(dim=0)
    return 1 + mu + 2*sigma

class FilteringDeepCME(torch.nn.Module):
    """ 
    FilteringDeepCME implementation with vectorization and batching.
    Ve use the delta NN + baseline strategy

    Usage:
        Train the model on a specific initial condition X0 and a list of time points T.

        The output is the given by the value of self.Y which corresponds to:

        self.baseline(X_t_i) = E(g_1(X_t_i+1), ..., g_R(X_t_i+1)) 

        where X_T is the value of the state at time T. 
    """

    def __init__(self, backbone, X_encoder, Y_encoder, baseline_net, tau_times, measurement_times, g_functions, temporal_feature_extractor, R, K, O, position_in_the_chain, n_NN_in_chain, device="cpu"):
        """ 
        Default constructor

        Args:
            
        """
        super(FilteringDeepCME, self).__init__()

        # NN components
        self.backbone = backbone
        self.X_encoder = X_encoder
        self.Y_encoder = Y_encoder
        self.baseline_net = baseline_net # this corresponds to Y in the original deepCME formulation (mathcal Y)

        # list of taus (in t0 t1)
        self.tau_times = tau_times.to(device)
        # list of measurement times (in t0, tn)
        self.measurement_times = measurement_times.to(device)
        
        self.g_functions = g_functions

        self.temporal_feature_extractor = temporal_feature_extractor

        self.R = R # number of g functions
        self.K = K # dimensionality of the output
        self.O = O # dimensionality of the observations
        self.device = device

        self.position_in_the_chain = position_in_the_chain
        self.n_NN_in_chain = n_NN_in_chain

        # use xavier initialization for the backbone
        for m in self.backbone.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
        # and for the encoders
        for m in self.X_encoder.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
        for m in self.Y_encoder.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
        # and for the baseline
        for m in self.baseline_net.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
        

    def forward(self, t, X, Y):
        """ 
        Forward pass of the model

        Args:
            t (float): The time point to condition the model on
            X (torch.Tensor): The input tensor (hidden)   [shape [batch_size, K]]
            Y (torch.Tensor): The input tensor (observed) [shape [batch_size, time_horizon, O]]

        Returns:
            torch.Tensor: The output of the model [shape [batch_size, R, K]]
        """
        eY = self.Y_encoder(Y)
        return self.forward_with_Y_embeddings(t, X, eY)

    def forward_with_Y_embeddings(self, t, X, eY):
        """ 
        Forward pass of the model 
        (this version is needed to avoid recomputing the embeddings, which can be extracted in a single iteration)

        Args:
            t (float): The time point to condition the model on
            X (torch.Tensor): The input tensor (hidden)   [shape [batch_size, K]]
            eY (torch.Tensor): The input tensor (observed) [shape [e(O)]]

        Returns:
            torch.Tensor: The output of the model [shape [batch_size, R, K]]
        """
        current_time_features = self.temporal_feature_extractor(t).to(X.device).repeat(X.shape[0], 1)
        eX = self.X_encoder(X)
        repey = eY.repeat(X.shape[0], 1)
        #print("eX, repey, ctf: ", eX.shape, repey.shape, current_time_features.shape)
        XY_with_temporal_features = torch.cat([current_time_features, eX, repey], dim=1)
        out = self.backbone(XY_with_temporal_features) # --- this is the V function
        # reshape the output to be of shape (batch_size, R, K)
        return out.view(-1, self.R, self.K) # --- from here analogous to DeepCME

    def forward_baseline(self, t, X, eY):
        #print("baseline eX, eY: ", eX.shape, eY.shape)
        eX = self.X_encoder(X)
        current_time_features = self.temporal_feature_extractor(t).to(eX.device).repeat(eX.shape[0], 1)
        repey = eY.repeat(eX.shape[0], 1)
        #print("baseline eX, repey, ctf: ", eX.shape, repey.shape, current_time_features.shape)
        XY_with_temporal_features = torch.cat([current_time_features, eX, repey], dim=1)
        #print("baseline XY_with_temporal_features: ", XY_with_temporal_features.shape)
        out = self.baseline_net(XY_with_temporal_features) # --- this is the mathcal Y function
        # no reshape needed
        return out
    
    def eval_baseline(self, t, X, Y):
        """ 
        Compute the baseline function for the loss computation

        Args:
            t (float): The time point to condition the model on
            X (torch.Tensor): The input tensor (hidden)   [shape [batch_size, K]]
            Y (torch.Tensor): The input tensor (observed) [shape [batch_size, time_horizon, O]]
            R (torch.Tensor): The output of the g functions [shape [batch_size, time_horizon, R]]

        Returns:
            torch.Tensor: The output of the baseline function [shape [batch_size, R]]
        """
        eY = self.Y_encoder(Y)
        return self.forward_baseline(t, X, eY)


    def L(self, x, delta_threshold):
        """ 
        Computation of the inner part of the loss function

        Args:
            x (torch.Tensor): The input tensor [shape [batch_size, R]]
            delta_threshold (torch.Tensor): The delta threshold tensor (normalization) [shape [R]]

        Returns:
            torch.Tensor: The output of the L function [shape [batch_size]]
        """
        return phi(x, delta_threshold).sum(dim=-1)

    def g(self, X):
        """ 
        Compute the output (g) functions

        Args:
            X (torch.Tensor): The input tensor [shape [batch_size, K]]

        Returns:
            torch.Tensor: The output of the g functions [shape [batch_size, R]]

        Note: I don't think this is further vecorizable, anyway the number of g functions is small (at most 4, usually)
        """
        out = []
        for i in range(len(self.g_functions)):
            out.append(self.g_functions[i](X))
        return torch.stack(out, dim=1)
    
    def SI(self, X, eY, dR):
        """ 
        Compute the stochastic integral

        Args:
            t_offset (float): The time offset of the considered interval
            X (torch.Tensor): The input tensor [shape [batch_size, T, K]]
            dR (torch.Tensor): The centered poisson process difference (sliced in the right range)
            eY (torch.Tensor): The incremental embeddings of the observed input tensor [shape [batch_size, T, hidden_size]]
            
        Returns:
            torch.Tensor: The output of the stochastic integral [shape [batch_size, R]]
        """
        # -- iterative version
        # out_old = None
        # for j in range(1, self.time_list.shape[0]):
        #     if out_old is None:
        #         out_old = torch.bmm(self.forward(self.time_list[j-1], X[:, j-1]), (R[:, j] - R[:, j-1]).unsqueeze(-1)).squeeze(-1)
        #     else:
        #         out_old += torch.bmm(self.forward(self.time_list[j-1], X[:, j-1]), (R[:, j] - R[:, j-1]).unsqueeze(-1)).squeeze(-1) 

        # compute the forward pass for all time points
        #print("before stacked_fwd X, eY, dR: ", X.shape, eY.shape, dR.shape)
        stacked_fwd = torch.vmap(lambda t, x, ey: self.forward_with_Y_embeddings(t, x, ey), in_dims=(0, 1, None))(self.tau_times[:-1]+self.measurement_times[self.position_in_the_chain], X[:,:-1], eY)
        # compute the product in the stochastic integral, then sum over time points
        out = torch.vmap(lambda v, dr: torch.bmm(v, dr.unsqueeze(-1)), in_dims=(0, 1))(stacked_fwd, dR).squeeze(-1).sum(dim=0)
        # return the sum
        return out
    

    def G_last(self, X):
        # likelihood = ... # let's start without
        X_T = X[:, -1] 
        return self.g(X_T)


    def compute_local_loss(self, X, eY, dR):
        
        if self.position_in_the_chain == self.n_NN_in_chain-1:
            gX = self.G_last(X)
        else:
            raise NotImplementedError("Not implemented for intermediate NNs")

        baseline = self.forward_baseline(self.measurement_times[self.position_in_the_chain], X[:,0], eY)
        #print("baseline: ", baseline.shape)
        inner = gX - self.SI(X, eY, dR) - baseline
        delta_threshold = compute_delta_threshold(gX)
        return self.L(inner, delta_threshold) # no mean here
    

    def loss(self, X, Y, R):
        """ 
        Computation of the loss function

        Args:
            X (torch.Tensor): The input tensor hid. [shape [batch_size, T, K]]
            Y (torch.Tensor): The input tensor obs. [shape [batch_size, T, O]]
            R (torch.Tensor): The centered poisson process [shape [batch_size, T, R]]

        Returns:
            torch.Tensor: The output of the loss function [shape [batch_size]]
        """

        # precompute the delta in the centered poisson process
        dR = R[:, 1:] - R[:, :-1]
        eY = self.Y_encoder(Y[:, self.position_in_the_chain+1:])
        # split view of X to divide in Y related intervals
        X = split_overalp(X, Y.shape[1])
        dR = split_overalp(dR, Y.shape[1])
        # # observation: eY needs to be done in reverse fashon, which is then quite efficient to compute

        #print("before vmapping X, eY, dR: ", X.shape, eY.shape, dR.shape)

        def inner_loop(x, ey, dr): # --- loop over k_prime
            #print("inner_loop: ", x.shape, ey.shape, dr.shape)
            out = self.compute_local_loss(x, ey, dr)
            #print("inner out: ", out.shape)
            return out

        def extern_loop(x, ey, dr): # --- loop over q
            #print("extern_loop: ", x.shape, ey.shape, dr.shape)
            return torch.vmap(inner_loop, in_dims=(1, None, 1))(x, ey, dr).sum(dim=0) # sum over k_prime

        L = torch.vmap(extern_loop, in_dims=(None, 0, None))(X, eY, dR).sum(dim=0).sum(dim=0) # sum over q and then sum over q_prime

        #L = torch.vmap(lambda x, ey, dr: torch.vmap(lambda t_k_prime, x_k_prime, ey, dr : self.compute_local_loss(t_k_prime, x_k_prime, ey, dr), in_dims=(0, 1, None, None))(self.measurement_times[:-1], x, ey, dr).sum(dim=0), in_dims=(None, 0, None))(X, eY, dR).sum(dim=0).sum(dim=0)
        L = L/((X.shape[0]**2)*X.shape[1]) # twice batch size, and then sum over time points (of y measurements)
        return L



def split_overalp(X, size_y):
    """ 
    Swap form X with absolute time to X with relative time (an extra dimension with overlap on the last element)

    NOTE: this should be done at dataset creation time
    """
    lastelement = X[:, -1, :]
    X = X[:, :-1].view(X.shape[0], size_y-1, -1, X.shape[-1])
    firstelements = torch.cat([X[:, 1:, 0, :], lastelement.unsqueeze(1)], dim=1)
    X = torch.cat([X, firstelements.unsqueeze(-2)], dim=2)
    return X

        

                





