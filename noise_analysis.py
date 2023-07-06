import bisect
# go to the parent folder
import os

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import time

# Our package
from RB_method_for_model_identification.RBForModelIdentification import RBForModelIdentification


########################################################################################################################
# The network structure
species_names = ['G0', 'G1', 'G2', 'mRNA']
stoichiometric_matrix = [[-1, 1, 0, 0, 0, 0],
                         [1, -1, -1, 1, 0, 0],
                         [0, 0, 1,-1, 0, 0],
                         [0, 0, 0, 0, 1,-1]]
parameters_names = ['k1','k2', 'k3', 'k4', 'kp1', 'kp2']
reaction_names = ['G Act. 1', 'G Deg. 1', 'G Act. 2', 'G Deg. 2', 'mRNA prod.', 'mRNA deg.']
propensities = [
    lambda k1, G0: k1*G0,
    lambda k2, G1: k2*G1,
    lambda k3, G1: k3*G1,
    lambda k4, G2: k4*G2,
    lambda kp1, kp2, G1, G2: kp1*G1 + (kp1+kp2)*G2,
    lambda mRNA: mRNA
]

range_of_species = \
    pd.DataFrame([[0, 1], [0, 1], [0, 1], [0, 150] ], index=species_names, columns=['min', 'max'])
range_of_parameters= \
    pd.DataFrame([[0, 1], [0, 1], [0, 2], [0, 2], [0, 80], [20, 120]],index=parameters_names,columns=['min', 'max'])
discretization_size_parameters = \
    pd.DataFrame([21, 21, 21, 21, 11, 11], index=parameters_names) #index=parameters_names

# The observation related information
# h_function = [
#     lambda Protein: Protein
# ]
h_function = [
    lambda mRNA: np.where(mRNA > 7, mRNA, 0)
]
observation_noise_intensity = [
    lambda : 1
]
#observation_noise_intensity = {'sigma1': 0.1}

maximum_size_of_each_follower_subsystem = 30000 #800 # 1000


MI = RBForModelIdentification(
    species_names=species_names,
    stoichiometric_matrix=stoichiometric_matrix,
    parameters_names=parameters_names,
    reaction_names=reaction_names,
    propensities=propensities,
    range_of_species=range_of_species,
    range_of_parameters=range_of_parameters,
    observation_noise_intensity=observation_noise_intensity,
    discretization_size_parameters=discretization_size_parameters,
    h_function=h_function,
    maximum_size_of_each_follower_subsystem=maximum_size_of_each_follower_subsystem)

print('leader species: ', MI.leader_species_time_course_data)
print('follower species: ', MI.get_follower_species_time_course_data())
print('follower parameters: ', MI.get_follower_parameters_time_course_data())

########################################################################################################################
# Measured cells

import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.stats import poisson

file_name = 'Example_of_model_identification/real_data/data45.mat'
real_data = sio.loadmat(file_name)

measured_cell_indexes = []

for cell_index in range(len(real_data['cellTrajectory'][0])):
    mRNA_trjectatory = real_data['cellTrajectory'][0,cell_index][0,0][0]/73
    Observation_times_list= list(range(2, len(mRNA_trjectatory)*2, 2))
    if len(mRNA_trjectatory) > 5:
        measured_cell_indexes.append(cell_index)

print('Number of measured cells:', len(measured_cell_indexes))
print('Cell Indexes:', measured_cell_indexes)

########################################################################################################################
# compute the mean and variance of the real data and inferred models

from tqdm import tqdm
import pickle

parameters_MAP_list = []
stationary_mean_real_cell_list = []
stationary_mean_inferred_mode_list = []
stationary_variance_real_cell_list = []
stationary_variance_inferred_mode_list = []

time = [2, 4, 6, 10, 20, 40, 80, 120] #[2*i for i in range(1, 120)]
mean_inferred_model_over_time_list = []
variance_inferred_model_over_time_list = []
mRNA_real_cell_over_time_list = []

# Initial distribution
Initial_marginal_distributions = MI.generate_uniform_marginal_distributions_via_speceis_range(MI.range_of_species)
delta_distribution = np.zeros((len(Initial_marginal_distributions['mRNA'].states), 1))
delta_distribution[0] = 1
Initial_marginal_distributions['mRNA'].adjust_distribution(delta_distribution)
# initial distribution of G0, G1, G2
Initial_marginal_distributions['G0'].adjust_distribution(np.array([[0], [1]]))
Initial_marginal_distributions['G1'].adjust_distribution(np.array([[1], [0]]))
Initial_marginal_distributions['G2'].adjust_distribution(np.array([[1], [0]]))

for cell_index in tqdm(measured_cell_indexes):
    with open(f"Example_of_model_identification/results_in_paper/real_data_cell_index_{cell_index}_particle_number_3000_larger_space.pkl", 'rb') as f:
            data=pickle.load(f)
    Marginal_distributions_over_time = data['margin_distributions_over_time']
    Y_list = data['Y_list']

    # parameters
    MAP_parameter_dict = {}
    for i in range(len(Marginal_distributions_over_time[-1])):
        element = list(Marginal_distributions_over_time[-1][i].parameter_species_ordering.keys())[0]
        if element in MI.parameters_names:
            map_index = np.argmax(Marginal_distributions_over_time[-1][i].distribution_list[-1])
            map_state = Marginal_distributions_over_time[-1][i].states[map_index, 0]
            MAP_parameter_dict.update({element: map_state})
    parameters_MAP_list.append(MAP_parameter_dict)

    # stationary mean and variance of real cells
    stationary_mean_real_cell_list.append(np.mean(Y_list))
    stationary_variance_real_cell_list.append(np.var(Y_list))

    # stationary mean and variance of MAP estimates
    Tf = 120
    distribution_inferred_model = MI.FSP(Initial_marginal_distributions, MI.range_of_species, MAP_parameter_dict, 0, Tf, normalization=True)
    probability = distribution_inferred_model.distribution_list[-1].T
    states = distribution_inferred_model.states[:,distribution_inferred_model.species_ordering['mRNA']]
    mean = np.sum(states*probability)
    variance = np.sum((states-mean)**2*probability)
    stationary_mean_inferred_mode_list.append(mean)
    stationary_variance_inferred_mode_list.append(variance)

    # mean and variance of inferred model over time
    if len(Marginal_distributions_over_time) < 120:
        continue
    mRNA_trjectatory = [Y_list[round(i/2)][0] for i in time]
    mRNA_real_cell_over_time_list.append(mRNA_trjectatory)

    # mean and variance of inferred model over time
    mean_over_time = []
    variance_over_time = []
    t_previous = 0
    for t in time:
        Tf = t
        distribution_inferred_model = MI.FSP(Initial_marginal_distributions, MI.range_of_species, MAP_parameter_dict, 0, Tf, normalization=True)
        probability = distribution_inferred_model.distribution_list[-1].T
        states = distribution_inferred_model.states[:,distribution_inferred_model.species_ordering['mRNA']]
        mean = np.sum(states*probability)
        variance = np.sum((states-mean)**2*probability)
        mean_over_time.append(mean)
        variance_over_time.append(variance)
    mean_inferred_model_over_time_list.append(mean_over_time)
    variance_inferred_model_over_time_list.append(variance_over_time)

########################################################################################################################
# save the result

data_to_save={
    # 'Identification Object': MI,
    'measured_cell_indexes': measured_cell_indexes,
    'parameters_MAP_list': parameters_MAP_list,
    'stationary_mean_real_cell_list': stationary_mean_real_cell_list,
    'stationary_mean_inferred_mode_list': stationary_mean_inferred_mode_list,
    'stationary_variance_real_cell_list': stationary_variance_real_cell_list,
    'stationary_variance_inferred_mode_list': stationary_variance_inferred_mode_list,
    'time': time,
    'mean_inferred_model_over_time_list': mean_inferred_model_over_time_list,
    'variance_inferred_model_over_time_list': variance_inferred_model_over_time_list,
    'mRNA_real_cell_over_time_list': mRNA_real_cell_over_time_list
}


with open('noise_analysis.pkl', 'wb') as f:
    pickle.dump(data_to_save, f)


