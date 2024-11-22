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

def likelihood_GaussianNoise(Y, X, h_transform, parameters):
    # define parameters as a dictionary when defining likelihood functions
    # the parameters are X, h_transform, noise_covariance
    hx = h_transform(X)
    
    # repeat hx to match the size of Y
    # hx = hx.repeat(Y.shape[0], 1)

    Sigma = parameters["noise_covariance"].to(hx.device)

    #print(Y.shape, hx.shape, Sigma.shape)

    # out = torch.nn.functional.gaussian_nll_loss(Y, hx, Sigma, full=True, reduction='none')

    #print(Y.shape, hx.shape, Sigma.shape)
    gaussian = torch.distributions.MultivariateNormal(Y, Sigma) # <-- invert role of Y and hx
    out = gaussian.log_prob(hx)

    return torch.exp(out)

def likelihood_GaussianNoise_vmap_compatible(Y, X, h_transform, parameters):

    # define parameters as a dictionary when defining likelihood functions
    # the parameters are X, h_transform, noise_covariance
    hx = h_transform(X).unsqueeze(-1)

    Sigma = parameters["noise_covariance"].to(hx.device)
    s_det = parameters["noise_covariance_determinant"].to(hx.device)
    Sigma_inv = parameters["noise_covariance_inverse"].to(hx.device)

    # compute the likelihood from the gaussian equation
    #print(Y.shape, hx.shape, Sigma.shape, (Y - hx).shape, Sigma_inv.shape)
    
    #out = torch.vmap(lambda hx, Sigma, s_det, Sigma_inv, Y : (2*torch.pi)**(-Sigma.shape[0]/2) * s_det**(-1/2) * torch.exp(-1/2 * (Y - hx) @ Sigma_inv @ (Y - hx).T), in_dims=(0, None, None, None, None))(hx, Sigma, s_det, Sigma_inv, Y)
    out = torch.vmap(lambda hx, Sigma, s_det, Sigma_inv, Y : torch.exp(-1/2 * (Y - hx) @ Sigma_inv @ (Y - hx).T), in_dims=(0, None, None, None, None))(hx, Sigma, s_det, Sigma_inv, Y)

    #print("Y : ", Y.shape, "hx : ", hx.shape, "Sigma : ", Sigma.shape, "out : ", out.shape)
    #closeness_mask = ((Y-hx).abs() < 1.0).all(dim=-1)

    # print("closeness_mask : ", closeness_mask.shape)
    # print("out : ", out.shape)

    #out = closeness_mask * out.squeeze(-1)  
    out = out.squeeze(-1)  

    # effective_sample_size = closeness_mask.sum(dim=0)
    # print("effective_sample_size : ", effective_sample_size)

    return out



def likelihood_GaussianNoise_unbatched(Y, X, h_transform, parameters):
    hx = h_transform(X).unsqueeze(0)
    Sigma = parameters["noise_covariance"].to(hx.device)
    gaussian = torch.distributions.MultivariateNormal(hx, Sigma)
    out = gaussian.log_prob(Y)
    return torch.exp(out)


def likelihood_constant_one(Y, X, h_transform, parameters):
    return torch.tensor([1.0]).to(X)