from OtherNetworks import MLP 
import torch


class TemporalFeatureExtractor(torch.nn.Module):

    def __init__(self, n_features, T, device="cpu"):
        # n_features is r in the paper 
        super(TemporalFeatureExtractor, self).__init__()
        self.n_features = n_features
        self.lambda_1 = torch.ones(n_features).to(device)
        self.lambda_2 = torch.ones(n_features).to(device)
        self.phi_1 = torch.ones(n_features).to(device)
        self.lambda_1.requires_grad = True
        self.lambda_2.requires_grad = True
        self.phi_1.requires_grad = True
        self.device = device
        self.T = T

    def forward(self, t):
        exp_vec = torch.exp(self.lambda_1 * (self.T-t))
        sin_vec = torch.sin(self.lambda_2 * (self.T-t) + self.phi_1)
        return torch.cat([exp_vec, sin_vec], dim=1)
    




class DeepCME(torch.nn.Module):

    def __init__(self, backbone, time_list, delta_threshold, g_function, temporal_feature_extractor, R, K):
        super(DeepCME, self).__init__()
        self.V = backbone
        self.time_list = time_list
        # add trainable parameter Y
        self.Y = torch.random(1).to(backbone.device)
        self.Y.requries_grad = True
        
        self.delta_threshold = delta_threshold
        self.g_function = g_function

        self.temporal_feature_extractor = temporal_feature_extractor

        self.R = R
        self.K = K



    def forward(self, t, X):
        current_time_features = self.temporal_feature_extractor(t).repeat(X.shape[0], 1).to(X.device)
        X_with_temporal_features = torch.cat([current_time_features, X], dim=1)
        out = self.V(X_with_temporal_features)
        




    def loss(self, X):
        X_T = X[:, -1] # X at time T








