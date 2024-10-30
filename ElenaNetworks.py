import torch 
from copy import deepcopy        
from OtherNetworks import RNNEncoder, MLP


class NeuralMartingale(torch.nn.Module):

    def __init__(self, X_encoder, Y_encoder, backbone, likelihood_function, likelihood_function_parameters, h_transform, g_function, stochiometry_matrix, discretization_size, output_sampling_size, times_tau, times_t, g_output_size, device='cpu', paired_batch_size=1, batch_size=1):
        """
        Implementation of the martingale approximating neural network.

        Args:
            X_encoder (_type_): X_encoder is the neural network that encodes the input X. (an MLP usually)
            Y_encoder (_type_): Y_encoder is the neural network that encodes the input Y. (an RNN usually)
            backbone (_type_): backbone is the neural network that computes the output of the martingale. (an MLP usually)
            likelihood_function (_type_): likelihood_function is the function that computes the likelihood of the data, namely the noise model.
            likelihood_function_parameters (_type_): likelihood_function_parameters is the parameters of the likelihood function. (if gaussian noise, the covariance matrix)
            h_transform (_type_): h_transform is the function that transforms the hidden process to into the observed process (X to Y).
            g_function (_type_): g_function is the function associated to E[g(X)|Y]. It reflects quantity of interest. 
            stochiometry_matrix (_type_): stochiometry_matrix is the matrix that describes the stochastic CRN.
            discretization_size (_type_): discretization_size is the number of points between t0 and t1 excluded.
            output_sampling_size (_type_): Number of measurements on Y.
            times_tau (_type_): sequence of tau times between t0 and t1 included. (this is the time of the hidden process, relative to the observed process)
            times_t (_type_): sequence of t times between t0 and tn included. (this is the time of the observed process)
            g_output_size (_type_): output size of the g_function.
            device (_type_): device is the device on which the computations are done. Defaults to 'cpu'.
        """
        super(NeuralMartingale, self).__init__()

        self.X_encoder = X_encoder
        self.Y_encoder = Y_encoder

        self.backbone = backbone

        self.likelihood_function = likelihood_function
        self.likelihood_function_parameters = likelihood_function_parameters
        self.h_transform = h_transform
        self.g_function = g_function
        self.stochiometry_matrix = stochiometry_matrix

        # maybe
        self.discretization_size = discretization_size # this is the number points between t0 and t1 excluded. 
        self.M_bar = discretization_size + 1 # as it must be 0 to mbar 
        self.output_sampling_size = output_sampling_size

        self.times_tau = times_tau
        self.times_t = times_t

        self.device=device
        self.g_output_size = g_output_size

        # nested vmap as double for loop (on batches and on times)
        self.vectorized_forward = torch.vmap(self.forward, in_dims=(None, 0, None))
        self.vectorized_forward_through_time = torch.vmap(self.forward, in_dims=(0, None, 0))
        self.vectorized_forward_through_time_and_batches = torch.vmap(self.vectorized_forward_through_time, in_dims=(None, None, 0))

        self.integral_fun = torch.vmap(lambda nn_delta, R_delta_for_each_j: nn_delta * R_delta_for_each_j, in_dims=(0, 0))

        self.next_network_in_chain = None

        self.history = {"G" : [], "NN" : [], "SI" : [], "Gs" : [], "NNs" : [], "SIs" : []}

        self.paired_batch_size = paired_batch_size
        self.batch_size = batch_size
        self.index_pairs = torch.tensor([(i, j) for i in range(batch_size) for j in range(batch_size)])
        self.index_pairs_distribution = torch.ones(batch_size**2)/(batch_size**2)

        # initialize all weights with xavier_uniform
        for param in self.parameters():
            if len(param.shape) >= 2:
                torch.nn.init.xavier_uniform_(param)

        assert self.g_output_size == self.backbone.output_size, "g_output_size must be equal to the output size of the martingale (backbone)"
        

    def set_next_network_in_chain(self, next_network_in_chain):
        self.next_network_in_chain = next_network_in_chain

    # TODO check broadcasting
    # def G_n_minus_1(self, t, Xstate, Yslice):
    #     """ 

    #     # NOTE: checked 

    #     Args:
    #         t: time at the end of the interval (so at t_n).
    #         Xstate: state at the end of k'-th interval.
    #         Yslice: Values of Y at t_n (note: this will have still an additional dimension as a proper tensor slice).
    #     """ 
    #     batch_size = Yslice.shape[0]
    #     b = self.g_function(Xstate).repeat(batch_size, 1)
    #     #a = self.likelihood_function(Yslice[:, -1, :], Xstate, self.h_transform, self.likelihood_function_parameters).repeat(1,b.shape[1])
    #     # print(a.shape, b.shape)
    #     return b#a*b
    
    # def G(self, t, Xstate, Yslice):
    #     """ 

    #     # NOTE: checked

    #     Args:
    #         t: time at the end of the interval.
    #         Xstate: state at the end of k'-th interval.
    #         Yslice: Value of Y from the end of k'-th interval to t_n.
    #     """
    #     return self.likelihood_function(Yslice[:, 0, :], Xstate, self.h_transform, self.likelihood_function_parameters)*self.next_network_in_chain.forward(t, Xstate, Yslice[:, 1:, :]) 
    
    # def stochastic_integral(self, t_k, k, k_prime, X, Y, R_delta):
    #     """ 
    #     Compute the stochastic integral of the loss function.

    #     This is computed in the interval (t_k, t_k+1) for all the trajectories in the batch.

    #     # NOTE: checked

    #     Args:

    #         t_k: the time at which the integral is computed.
    #         k: the index of the time in the sequence of times. (current)
    #         k_prime: the index of the time in the sequence of times. (virtual)
    #         X: the hidden process tensor.
    #         Y: the output process tensor.
    #         R_delta: the delta of the centered poission process.

    #     Returns:
    #         The value of the stochastic integral in the interval (t_k, t_k+1) for all the trajectories in the batch.
    #     """
    #     # precompute Yslice
    #     # --- the code was checked --- 
    #     Yslice = Y[:, k+1:, :]
    #     integral = torch.zeros(Y.shape[0], self.g_output_size).to(X)
    #     # naive implementation
    #     M = self.stochiometry_matrix.shape[1]
    #     for j in range(M):
    #         # i moves within the interval (t_k, t_k+1)
    #         for m in range(self.M_bar):
    #             Xslice = X[k_prime*(self.M_bar)+m, :]
    #             current_time = self.times_tau[m] + t_k 
    #             # before and after displacement
    #             nn_before = self.forward(current_time, Xslice, Yslice)
    #             nn_after = self.forward(current_time, Xslice + self.stochiometry_matrix[:, j], Yslice)
    #             nn_delta = nn_after - nn_before
    #             R_delta_j = R_delta[k_prime*(self.M_bar)+m, j].repeat(nn_delta.shape[0], 1) # now batched
    #             # print(nn_delta.shape, R_delta_j.shape, R_delta.shape, integral.shape)
    #             integral += nn_delta * R_delta_j # TODO check index order

    #     return integral

    # def vectorized_stochastic_integral(self, t_k, k, k_prime, X, Y, R_delta):
    #     """ 
    #     Vectorized implementation of the stochastic integral.
    #     """
    #     # precompute Yslice
    #     Yslice = Y[:, k+1:, :]

    #     M = self.stochiometry_matrix.shape[1]

    #     perturbations = self.stochiometry_matrix.to(X).T #torch.stack([self.stochiometry_matrix[:, j] for j in range(M)], dim=0)
        
    #     X_at_times = X[k_prime*self.M_bar:(k_prime+1)*self.M_bar, :]
    #     X_perturbed = torch.cat(((X_at_times.unsqueeze(1).repeat(1, M, 1) + perturbations.unsqueeze(0).repeat(self.M_bar, 1, 1))).unbind()) # n_perturbations, state_dim
    #     # print(X_at_times.shape, X_perturbed.shape)
    #     # print(X_perturbed.shape)
    #     # vectorized forward pass with vmap
    #     # n_times*M, batch_size, g_output_size
    #     # print(X_at_times.shape)
    #     # print(self.vectorized_forward(t_k, X_at_times, Yslice).shape)
    #     # print(torch.repeat_interleave(self.vectorized_forward(t_k, X_at_times, Yslice), M, dim=0).shape)
    #     # print(self.vectorized_forward(t_k, X_perturbed, Yslice).shape)


    #     nn_delta = self.vectorized_forward(t_k, X_perturbed, Yslice) - self.vectorized_forward(t_k, torch.cat(X_at_times.unsqueeze(1).repeat(1,M,1).unbind()), Yslice)
    #     #print(nn_delta.shape)

    #     R_delta_for_each_j = torch.stack([
    #         R_delta[k_prime*self.M_bar:(k_prime+1)*self.M_bar, j] for j in range(M)
    #     ], dim=0).T.flatten()

    #     # print all shap
    #     #print(nn_delta.shape, R_delta_for_each_j.shape)
    #     # vmap 
    #     # print(self.integral_fun(nn_delta, R_delta_for_each_j).shape)
    #     integral = self.integral_fun(nn_delta, R_delta_for_each_j).sum(dim=0)
    #     #print(integral.shape)
        
    #     return integral

    # def improved_vectorized_stochastic_integral(self, k_prime, delta_NN_k_q_prime, R_delta_q_prime):
    #     """ 
    #     Improved vectorized implementation of the stochastic integral.
    #     """
    #     M = self.stochiometry_matrix.shape[1]
    #     # we can use standard broadcasting
    #     #print(delta_NN_k_q_prime.shape, R_delta_q_prime.shape)
    #     R_tilde = R_delta_q_prime[k_prime*self.M_bar:(k_prime+1)*self.M_bar, :].unsqueeze(-1).repeat(1, 1, self.g_output_size).flatten(0,1)
    #     # print(R_tilde.max(), R_tilde.min())
    #     # print(delta_NN_k_q_prime.max(), delta_NN_k_q_prime.min())
    #     temp = delta_NN_k_q_prime * R_tilde
    #     # print(temp.max(), temp.min())
    #     # print(temp.sum(dim=1).max(), temp.sum(dim=1).min())
    #     return temp.sum(dim=1)

    # def precompute_delta_NN_k(self, k, k_prime, q_prime, X, Y):
    #     M = self.stochiometry_matrix.shape[1]
    #     timelist = torch.linspace(self.times_t[k], self.times_t[k+1], self.M_bar).to(X)
    #     X_slice = X[q_prime, k_prime*self.M_bar:(k_prime+1)*self.M_bar, :]
    #     Y_slice = Y[:, k+1:, :]
    
    #     NN_k = torch.repeat_interleave(self.vectorized_forward_through_time(timelist, X_slice, Y_slice).permute(1,0,2), M, dim=1)
    #     NN_k_prime = self.vectorized_forward_through_time(torch.repeat_interleave(timelist, M, dim=0), torch.repeat_interleave(X_slice, M, dim=0) + self.stochiometry_matrix.T.repeat(X_slice.shape[0], 1), Y_slice).permute(1,0,2)
        
    #     return NN_k_prime - NN_k # [BS, m_bar*M, g_output_size]

    # def loss(self, X, Y, R, k):
    #     # NOTE: checked
    #     # for a single time k, compute the loss function.

    #     # compute the delta of the centered poisson process
    #     R_delta = R[:, 1:] - R[:, :-1]

    #     loss = None # initialize the loss for the batch.

    #     # rename the variables to be closer to the mathematical notation
    #     batch_size = X.shape[0]
    #     n = self.times_t.shape[0] - 1
    #     #print(self.times_t.shape, n)

    #     for k_prime in range(n): # for all the times in the sequence [n]

    #         for q_prime in range(batch_size): # and for all other sequences in the batch
    #             #print(k, k_prime, q_prime)

    #             delta_NN_k = self.precompute_delta_NN_k(k, k_prime, q_prime, X, Y)

    #             # indexes on X
    #             tau_k_prime_index_abs_begin = k_prime * self.M_bar 
    #             tau_k_prime_index_abs_end = (k_prime+1) * self.M_bar
    #             #print("tauindex", tau_k_prime_index_abs_begin, tau_k_prime_index_abs_end, self.M_bar)

    #             # print(k, k_prime, q_prime, self.times_t[k], self.times_t[k+1])

    #             if  k == n - 1:
    #                 # don't loose the dimensionality of Y
    #                 Gm1 = self.G_n_minus_1(self.times_t[k+1], X[q_prime, tau_k_prime_index_abs_end, :], Y[:, k+1:, :])
    #                 NN = self.forward(self.times_t[k], X[q_prime, tau_k_prime_index_abs_begin, :], Y[:, k+1:, :])
    #                 #SI_old = self.stochastic_integral(self.times_t[k], k, k_prime, X[q_prime, :, :], Y, R_delta[q_prime, :, :])
    #                 #SI = self.vectorized_stochastic_integral(self.times_t[k], k, k_prime, X[q_prime, :, :], Y, R_delta[q_prime, :, :])

    #                 SI = self.improved_vectorized_stochastic_integral(k_prime, delta_NN_k, R_delta[q_prime, :, :])

    #                 #print(SI.max(), SI.min())

    #                 self.history["G"].append(Gm1.sum())
    #                 self.history["NN"].append(NN.sum())
    #                 self.history["SI"].append(SI.sum())
    #                 # save also the SD
    #                 self.history["Gs"].append(Gm1.std())
    #                 self.history["NNs"].append(NN.std())
    #                 self.history["SIs"].append(SI.std())

    #                 # print(Gm1.shape, NN.shape)

    #                 #print(Gm1.shape, NN.shape, SI.shape)
    #                 if loss is None:
    #                     loss = ((Gm1 - NN + SI)**2)
    #                 else:
    #                     loss += ((Gm1 - NN + SI)**2)
    #                 # print(((Gm1 - NN + SI).abs()).sum())
    #                 # print(Gm1.sum(), NN.sum(), SI.sum())
    #                 # print(SI.sum() - SI_old.sum())
    #                 # print("SI", SI)
    #                 # print("SI_old", SI_old)
    #                 # raise Exception("stop")
    #             else:
    #                 loss += (self.G(self.times_t[k+1], X[q_prime, tau_k_prime_index_abs_end, :], Y[:, k+1:, :]) - self.forward(self.times_t[k], X[q_prime, tau_k_prime_index_abs_begin, :], Y[:, k+1:, :]) + self.stochastic_integral(self.times_t[k], k, k_prime, X[q_prime, :, :], Y, R_delta[q_prime, :, :]))**2
    #
    #      return loss.sum()/(n*batch_size*batch_size*self.M_bar) # batch average (per interval)
            
    def G_n_minus_1(self, Xstate, Yslice):
        likelihood = self.likelihood_function(Yslice[-1, :], Xstate, self.h_transform, self.likelihood_function_parameters)
        gx = self.g_function(Xstate)

        # TODO remove
        # likelihood = 1.0
        return likelihood*gx

    def stochastic_integral(self, k, k_prime, q, q_prime, X, Y, R):
        M = self.stochiometry_matrix.shape[1]
        si = torch.zeros(self.g_output_size).to(X)
        for j in range(M):
            for m in range(self.M_bar):
                X_displaced = X[q_prime, k_prime*self.M_bar + m, :] + self.stochiometry_matrix[:, j]
                if torch.all(X_displaced >= 0):
                    nn = self.forward(self.times_t[k]+self.times_tau[m], X[q_prime, k_prime*self.M_bar + m, :], Y[q:q+1, k+1:, :])
                    nn_displaced = self.forward(self.times_t[k]+self.times_tau[m], X_displaced, Y[q:q+1, k+1:, :])
                    nn_delta = nn_displaced - nn
                    R_delta_j = R[q_prime, k_prime*self.M_bar + m + 1, j] - R[q_prime, k_prime*self.M_bar + m, j]
                    si += (nn_delta*R_delta_j).squeeze()
                else:
                    continue
                   
        return si

    def stochastic_integral_vectorized(self, k, k_prime, q, q_prime, X, Y, R):
        M = self.stochiometry_matrix.shape[1]
        si = torch.zeros(self.M_bar, self.g_output_size).to(X)
        for j in range(M):
            # for m in range(self.M_bar): # <--- vectorize on m
            m = torch.arange(self.M_bar).to(X).int()
            # print(m)
            X_displaced = X[q_prime, k_prime*self.M_bar + m, :] + self.stochiometry_matrix[:, j]
            self.times_t = self.times_t.to(X)
            self.times_tau = self.times_tau.to(X)

            # verify stochoimetry
            validity_mask = torch.sum(X_displaced < 0, dim=-1) == 0

            vectorized_forward = torch.vmap(lambda t, x, y: self.forward(t, x, y), in_dims=(0, 0, None))

            nn = vectorized_forward(self.times_t[k]+self.times_tau[m], X[q_prime, k_prime*self.M_bar + m, :], Y[q:q+1, k+1:, :])
            nn_displaced = vectorized_forward(self.times_t[k]+self.times_tau[m], X_displaced, Y[q:q+1, k+1:, :])
            nn_delta = nn_displaced - nn
            R_delta_j = R[q_prime, k_prime*self.M_bar + m + 1, j] - R[q_prime, k_prime*self.M_bar + m, j]
            # print(nn_delta.squeeze(1).shape, R_delta_j.unsqueeze(1).repeat(1,self.g_output_size).shape, validity_mask.unsqueeze(1).repeat(1,self.g_output_size).shape)
            si += (nn_delta.squeeze(1)*R_delta_j.unsqueeze(1).repeat(1,self.g_output_size)).squeeze()*validity_mask.unsqueeze(1).repeat(1,self.g_output_size)

        return si.sum(dim=0)


    def loss(self, X, Y, R, k): # O(n b^2 M M_bar)

        index_pairs = self.index_pairs_distribution.multinomial(num_samples=self.paired_batch_size, replacement=False)

        batch_size = X.shape[0]
        n = self.times_t.shape[0] - 1
        loss = torch.zeros(self.g_output_size).to(X)
        for pair_index in index_pairs:
        #for q in range(batch_size):
        #    for q_prime in range(batch_size):
            q, q_prime = self.index_pairs[pair_index]
            
            #for q in range(batch_size):
            #    for q_prime in range(batch_size):
            k_prime = k
            # for k_prime in range(n): 
            g = self.G_n_minus_1(X[q_prime, (k_prime+1)*self.M_bar, :], Y[q, k+1:, :])
            nn = self.forward(self.times_t[k], X[q_prime, k_prime*self.M_bar, :], Y[q:q+1, k+1:, :]).squeeze()
            #si = self.stochastic_integral(k, k_prime, q, q_prime, X, Y, R)
            si_vectorized = self.stochastic_integral_vectorized(k, k_prime, q, q_prime, X, Y, R)

            #print("g", g.shape, "nn", nn.shape, "si", si_vectorized.shape)
            #print("si", si)
            #print("si_vectorized", si_vectorized)
            loss += (g - nn - si_vectorized)**2

        return loss.sum()/(self.paired_batch_size*n)




    # def loss(self, X, Y, R, k):
    #     # NOTE: checked
    #     # for a single time k, compute the loss function.

    #     # compute the delta of the centered poisson process
    #     R_delta = R[:, 1:] - R[:, :-1]
    #     t_bar = self.times_t[k] + self.times_tau

    #     # compute the forward pass through the network at the times t_bar
    #     M = self.stochiometry_matrix.shape[1]
    #     NN_fwd = self.vectorized_forward_through_time_and_batches(t_bar, X, Y[:, k+1:, :]).repeat(M, 1, 1)
    #     NN_fwd_perturbed = NN_fwd.clone()
    #     for i in range(M):
    #         NN_fwd_perturbed[i, :, :] += self.stochiometry_matrix[i, :]

    #     NN_delta = NN_fwd_perturbed - NN_fwd

    #     SI = (NN_delta[:, :, :-1, :] * R_delta).sum(dim=0) # sum over the reactions

    #     batch_size = X.shape[0]
    #     n = self.times_t.shape[0] - 1

    #     for k_prime in range(n): # for all the times in the sequence [n] # TODO revert to range(n)

    #         # for q in range(batch_size): # and for all other sequences in the batch

    #         for q_prime in range(batch_size):
    #             tau_k_prime_index_abs_begin = k_prime * self.M_bar 
    #             tau_k_prime_index_abs_end = (k_prime+1) * self.M_bar
                
    #             if k == n - 1:
    #                 Gm1 = self.G_n_minus_1(self.times_t[k+1], X[q_prime, tau_k_prime_index_abs_end, :], Y[q:q+1, k+1:, :])
    #                 # NN = self.forward(self.times_t[k], X[q_prime, tau_k_prime_index_abs_begin, :], Y[q:q+1, k+1:, :])
    #                 NN = NN_fwd[0, q_prime, tau_k_prime_index_abs_begin, :]
    #                 SI_q_prime = SI[q_prime, k_prime, :]
    #                 loss = (Gm1 - NN + SI_q_prime)**2

    #             else:
    #                 # TODO case k < n-1
    #                 pass

        



        # loss = None # initialize the loss for the batch.

        # # rename the variables to be closer to the mathematical notation
        # batch_size = X.shape[0]
        # n = self.times_t.shape[0] - 1
        # #print(self.times_t.shape, n)

        # for k_prime in range(n-1, n): # for all the times in the sequence [n] # TODO revert to range(n)

        #     for q in range(batch_size): # and for all other sequences in the batch

        #         #for q_prime in range(batch_size):
        #             q_prime = q

        #             delta_NN_k = self.precompute_delta_NN_k(k, k_prime, q_prime, X, Y[q:q+1, :, :])

        #             # indexes on X
        #             tau_k_prime_index_abs_begin = k_prime * self.M_bar 
        #             tau_k_prime_index_abs_end = (k_prime+1) * self.M_bar
                    

        #             if  k == n - 1:
        #                 # don't loose the dimensionality of Y

        #                 Gm1 = self.G_n_minus_1(self.times_t[k+1], X[q_prime, tau_k_prime_index_abs_end, :], Y[q:q+1, k+1:, :])
        #                 NN = self.forward(self.times_t[k], X[q_prime, tau_k_prime_index_abs_begin, :], Y[q:q+1, k+1:, :])
        #                 #SI_new = self.improved_vectorized_stochastic_integral(k_prime, delta_NN_k, R_delta[q_prime, :, :])
        #                 SI = self.stochastic_integral(self.times_t[k], k, k_prime, X[q_prime, :, :], Y[q:q+1, :, :], R_delta[q_prime, :, :])

        #                 self.history["G"].append((Gm1).sum())
        #                 self.history["NN"].append((NN).sum())
        #                 self.history["SI"].append((SI).sum())
        #                 # save also the SD
        #                 self.history["Gs"].append((Gm1).std())
        #                 self.history["NNs"].append((NN).std())
        #                 self.history["SIs"].append((SI).std())

        #                 if loss is None:
        #                     loss = (Gm1 - NN + SI)**2
        #                 else:
        #                     loss += (Gm1 - NN + SI)**2


        #                 #print((SI-SI_new).abs().sum())
        #                 #raise Exception("stop")
        #             else:
        #                 pass
        #                 #loss += (self.G(self.times_t[k+1], X[q_prime, tau_k_prime_index_abs_end, :], Y[:, k+1:, :]) - self.forward(self.times_t[k], X[q_prime, tau_k_prime_index_abs_begin, :], Y[:, k+1:, :]) + self.stochastic_integral(self.times_t[k], k, k_prime, X[q_prime, :, :], Y, R_delta[q_prime, :, :]))**2

        # return loss.sum()/(n*batch_size*batch_size) # batch average (per interval)

    def forward(self, t, XatT, Yslice):
        # NOTE: checked
        # encode X and Y
        
        #Yslice = torch.zeros_like(Yslice) # TODO remove
        #XatT = XatT*torch.tensor([1.,1.,0.]).to(XatT) # TODO remove
        eX = self.X_encoder(XatT).repeat(Yslice.shape[0], 1)
        eY = self.Y_encoder(Yslice)

        if type(t) != torch.Tensor:
            t = torch.tensor([t], dtype=torch.float32).to(eX).repeat(eX.shape[0], 1)
        else:
            t = t.to(eX).repeat(eX.shape[0], 1)

        # concatenate embeddings and predict
        # out = self.backbone( torch.cat([t, eX, eY], dim=1) )
        out = self.backbone( torch.cat([t, eX, eY], dim=1) )

        return out



class NeuralMartingaleChain(torch.nn.Module):

    def __init__(self, N, neural_martingale):
        super(NeuralMartingaleChain, self).__init__()
        self.N = N
        # create N NeuralMartingale objects
        self.chain = torch.nn.ModuleList([deepcopy(neural_martingale) for _ in range(N)])
        self.device = neural_martingale.device

        for i in range(len(self.chain)-1):
            self.chain[i].set_next_network_in_chain(self.chain[i+1])

    def forward(self, t, XatT, Yslice, i):    
        return self.chain[i](t, XatT, Yslice)
    


