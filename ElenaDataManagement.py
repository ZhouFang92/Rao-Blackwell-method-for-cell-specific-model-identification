# helper functions for defining the sampling times for the X and Y processes
import numpy as np
import torch
from tqdm import tqdm

def count_samples_for_supersampling(original, intensity):
    """ 
    Prepare supersampling for X ranges
    """
    x = original
    n_holes = x - 1
    x = intensity * n_holes + x
    return x

def get_X_Y_sampling_times(t0, tn, number_of_Y_samples, X_supersampling_intensity):
    """ 
    helper function to get sampling times for the X and Y processes
    """
    assert (number_of_Y_samples % 2 == 1 and number_of_Y_samples > 1) or number_of_Y_samples==2, "the number of mesurements must be odd and greater than 1 to include the extremes"
    assert X_supersampling_intensity > 0, "we require some supersampling on the X"
    forY = np.linspace(t0, tn, number_of_Y_samples)
    forX = np.linspace(t0, tn, count_samples_for_supersampling(number_of_Y_samples, X_supersampling_intensity))
    return forX, forY

def sample_trajectory_on_times(sampling_points, time_list, state_list):
    """ 
    sample a trajectory on specific time points
    """

    sampling_time_index = 0
    real_time_index = 0

    sampled_state_list = []

    while sampling_time_index < len(sampling_points):
        if real_time_index == len(time_list):
            sampled_state_list.append(state_list[real_time_index-1].copy())
            sampling_time_index += 1
        elif time_list[real_time_index] <= sampling_points[sampling_time_index]:
            real_time_index += 1
        else:
            sampled_state_list.append(state_list[max(0, real_time_index-1)].copy())
            sampling_time_index += 1

    return sampled_state_list


def CRN_simulations_to_dataloaders(data, batch_size, test_split=0.2, shuffle_dataset=True):
    """ 
    convert a dataset to a batched torch dataset
    """

    # torchify
    tX = torch.tensor(data["times_X"]).float()
    tY = torch.tensor(data["times_Y"]).float()
    Xs = torch.stack([torch.tensor(np.array(X)) for X in data["X"]]).float()
    Ys = torch.stack([torch.tensor(np.array(Y)) for Y in data["Y"]]).float()
    Rs = torch.stack([torch.tensor(np.array(R)) for R in data["R"]]).float()

    dataset = torch.utils.data.TensorDataset(Xs, Ys, Rs)
    train_set, test_set = torch.utils.data.random_split(dataset, [int(len(dataset)*(1-test_split)), int(len(dataset)*test_split)])

    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=shuffle_dataset)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, tX, tY

import random
import math

def run_SSA_for_filtering(MI, initial_states, parameter_values, t_fin, number_of_Y_samples, X_supersampling_intensity, t0=0, n_samples=1):
    """ 
    run multiple SSA simulations with the specified sampling strategy
    """

    X_sampling_times, Y_sampling_times = get_X_Y_sampling_times(t0, t_fin, number_of_Y_samples, X_supersampling_intensity)

    state_lists = []
    R_lists = []
    Y_lists = []

    for _ in tqdm(range(n_samples)):

        time_list, state_list, cPP = MI.SSA(initial_states[math.floor(random.random()*len(initial_states))], parameter_values, t0, t_fin, compute_centered_poisson_process=True)
        
        sampled_state_list = sample_trajectory_on_times(X_sampling_times, time_list, state_list)
        state_lists.append(sampled_state_list)
        # TODO check if R+ and R- are needed
        R = cPP.sample_at_times(X_sampling_times)
        R_lists.append(R)

        # Generate the observations
        Observation_times_list = Y_sampling_times
        Y_list = MI.generate_observations(state_list, time_list, parameter_values, Observation_times_list)
        Y_lists.append(Y_list)

    data = {"times_X" : X_sampling_times, "times_Y" : Y_sampling_times, "X" : state_lists, "Y" : Y_lists, "R" : R_lists}

    return data