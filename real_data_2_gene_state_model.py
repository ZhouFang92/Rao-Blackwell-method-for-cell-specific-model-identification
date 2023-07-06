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
species_names = ['G0', 'G1', 'mRNA']
stoichiometric_matrix = [[-1, 1, 0, 0],
                         [1, -1, 0, 0],
                         [0, 0, 1,-1]]
parameters_names = ['k1','k2', 'kp1']
reaction_names = ['G Act. 1', 'G Deg. 1', 'mRNA prod.', 'mRNA deg.']
propensities = [
    lambda k1, G0: k1*G0,
    lambda k2, G1: k2*G1,
    lambda kp1, G1: kp1*G1,
    lambda mRNA: mRNA
]

range_of_species = \
    pd.DataFrame([[0, 1], [0, 1], [0, 120] ], index=species_names, columns=['min', 'max'])
range_of_parameters= \
    pd.DataFrame([[0, 1], [0, 1], [0, 150]],index=parameters_names,columns=['min', 'max'])
discretization_size_parameters = \
    pd.DataFrame([21, 21, 151], index=parameters_names) #index=parameters_names

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

maximum_size_of_each_follower_subsystem = 20000 #800 # 1000


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
# load the data
import scipy.io as sio

file_name = 'Example_of_model_identification/real_data/data45.mat'
real_data = sio.loadmat(file_name)
cell_index = 78
mRNA_trjectatory = real_data['cellTrajectory'][0,cell_index][0,0][0]/73
Observation_times_list= list(range(2, len(mRNA_trjectatory)*2, 2))
Y_list = [np.array(element) for element in mRNA_trjectatory[1:]]

########################################################################################################################
# Do the inference

# initial distribution: all the species are assumed to be independent
Initial_marginal_distribution = MI.generate_uniform_marginal_distributions()
# initial distribution of mRNA
delta_distribution = np.zeros((len(Initial_marginal_distribution['mRNA'].states), 1))
delta_distribution[0] = 1
Initial_marginal_distribution['mRNA'].adjust_distribution(delta_distribution)
# initial distribution of G0, G1, G2
Initial_marginal_distribution['G0'].adjust_distribution(np.array([[0], [1]]))
Initial_marginal_distribution['G1'].adjust_distribution(np.array([[1], [0]]))

# inference
particle_size = 10
Marginal_distributions_over_time, final_particles, time_result, mean_result, std_result \
    = MI.RB_inference_time_course_data(
        time_points_for_observations = Observation_times_list,
        Y_list=Y_list,
        particle_size=particle_size,
        marginal_distributions_dic= Initial_marginal_distribution)

########################################################################################################################
# Save the results

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


file_name = 'real_data_2_gene_state_model_'+'_cell_index_'+str(cell_index)+'_particle_number_'+ str(particle_size) + '.pkl'

with open(file_name, 'wb') as f:
    pickle.dump(data_to_save, f)


