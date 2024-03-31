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
# The network structure
species_names = ['G0', 'G1', 'G2', 'mRNA']
stoichiometric_matrix = [[-1, 1, 0, 0, 0, 0],
                         [1, -1, -1, 1, 0, 0],
                         [0, 0, 1,-1, 0, 0],
                         [0, 0, 0, 0, 1,-1]]
parameters_names = ['k1','k2', 'k3', 'k4', 'kp1', 'kp2', 'sigma']
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
    pd.DataFrame([[0, 1], [0, 1], [0, 1], [0, 2], [20, 40], [40, 80], [0.1, 2]],index=parameters_names,columns=['min', 'max'])
discretization_size_parameters = \
    pd.DataFrame([21, 21, 21, 21, 11, 11, 20], index=parameters_names) #index=parameters_names

# The observation related information
# h_function = [
#     lambda Protein: Protein
# ]
h_function = [
    lambda mRNA: np.where(mRNA > 7, mRNA, 0)
]
observation_noise_intensity = [
    lambda sigma: sigma
]


maximum_size_of_each_follower_subsystem = 300000 #800 # 1000


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
# load real data
import scipy.io as sio
import sys
import matplotlib.pyplot as plt
from scipy.stats import poisson

file_name = 'Example_of_model_identification/real_data/data45.mat'
real_data = sio.loadmat(file_name)
cell_index = 78 # int(sys.argv[1]) #78
print('cell index: ', cell_index)
mRNA_trjectatory = real_data['cellTrajectory'][0,cell_index][0,0][0]/73 # 73 the ration between the fluorescent readout and the mRNA copy number
Observation_times_list= list(range(2, len(mRNA_trjectatory)*2, 2))
Y_list = [np.array(np.minimum(element, 150)) for element in mRNA_trjectatory[1:]] # the range of the measurement is 0-150


########################################################################################################################
# Inference

# initial distribution: all the species are assumed to be independent
Initial_marginal_distribution = MI.generate_uniform_marginal_distributions()
# initial distribution of mRNA
delta_distribution = np.zeros((len(Initial_marginal_distribution['mRNA'].states), 1))
delta_distribution[0] = 1
Initial_marginal_distribution['mRNA'].adjust_distribution(delta_distribution)
# initial distribution of G0, G1, G2
Initial_marginal_distribution['G0'].adjust_distribution(np.array([[0], [1]]))
Initial_marginal_distribution['G1'].adjust_distribution(np.array([[1], [0]]))
Initial_marginal_distribution['G2'].adjust_distribution(np.array([[1], [0]]))
# initial distribution of k3, with half of the probability the value is 0
Initial_marginal_distribution['k3'].distribution[0] = sum(Initial_marginal_distribution['k3'].distribution[1:])
Initial_marginal_distribution['k3'].distribution = Initial_marginal_distribution['k3'].distribution / sum(Initial_marginal_distribution['k3'].distribution)

# inference
particle_size = 3000 #int(sys.argv[1]) #100
Marginal_distributions_over_time, final_particles, time_result, mean_result, std_result \
    = MI.RB_inference_time_course_data(
        time_points_for_observations = Observation_times_list,
        Y_list=Y_list,
        particle_size=particle_size,
        marginal_distributions_dic= Initial_marginal_distribution)

########################################################################################################################
# save the results

import pickle

data_to_save={
    # 'Identification Object': MI,
    'Observation_times_list': Observation_times_list,
    'Y_list': Y_list,
    #'Particles_list_returned': Particles_list_returned,
    'time_result': time_result,
    'mean_result': mean_result,
    'std_result': std_result,
    #'final_particles': final_particles,
    'margin_distributions_over_time': Marginal_distributions_over_time
}

file_name = 'real_data'+'_cell_index_'+str(cell_index)+'_particle_number_'+ str(particle_size) + '_sigma_uncertainty.pkl'

with open(file_name, 'wb') as f:
    pickle.dump(data_to_save, f)