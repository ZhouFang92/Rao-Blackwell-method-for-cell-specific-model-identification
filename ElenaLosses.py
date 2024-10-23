import torch

# def likelihood_GaussianNoise(Y, X, h_transform, parameters):
#     # define parameters as a dictionary when defining likelihood functions
#     # the parameters are X, h_transform, noise_covariance
#     hx = h_transform(X)
    
#     # repeat hx to match the size of Y
#     hx = hx.repeat(Y.shape[0], 1)

#     Sigma = parameters["noise_covariance"].to(hx.device)

#     #print(Y.shape, hx.shape, Sigma.shape)

#     # out = torch.nn.functional.gaussian_nll_loss(Y, hx, Sigma, full=True, reduction='none')

#     gaussian = torch.distributions.MultivariateNormal(hx, Sigma)
#     out = gaussian.log_prob(Y)


#     return torch.exp(out.unsqueeze(1))

def likelihood_GaussianNoise_unbatched(Y, X, h_transform, parameters):
    hx = h_transform(X).unsqueeze(0)
    Sigma = parameters["noise_covariance"].to(hx.device)
    gaussian = torch.distributions.MultivariateNormal(hx, Sigma)
    out = gaussian.log_prob(Y)
    return torch.exp(out)


def likelihood_constant_one(Y, X, h_transform, parameters):
    return torch.tensor([1.0]).to(X)