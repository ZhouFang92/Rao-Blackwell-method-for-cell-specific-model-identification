import bisect
# go to the parent folder
import os

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import time

# Our package
from RB_method_for_model_identification.RBForModelIdentification import RBForModelIdentification

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
    lambda kp1, kp2, G1, G2: kp1*(G1+G2) + kp2*G2,
    lambda mRNA: 1*mRNA
]

range_of_species = \
    pd.DataFrame([[0, 1], [0, 1], [0, 1], [0, 20] ], index=species_names, columns=['min', 'max'])
range_of_parameters= \
    pd.DataFrame([[0, 1], [0, 1], [0, 1], [0, 1], [1, 10], [1, 10]],index=parameters_names,columns=['min', 'max'])
discretization_size_parameters = \
    pd.DataFrame([21, 21, 21, 21, 10, 10], index=parameters_names) #index=parameters_names

# The observation related information
# h_function = [
#     lambda Protein: Protein
# ]
h_function = [
    lambda mRNA: mRNA
]
observation_noise_intensity = [
    lambda : 0.1
]
#observation_noise_intensity = {'sigma1': 0.1}

maximum_size_of_each_follower_subsystem = 30000 #800 #10000 # 1000


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



# Get a trajectory of the system
parameter_values = {'k1': 0.3, 'k2': 0.4, 'k3': 0, 'k4': 0.3, 'kp1': 3, 'kp2': 0}
initail_state = {'G0': 1, 'G1': 0, 'G2':0, 'mRNA': 0}
tf = 480
time_list, state_list = MI.SSA(initail_state, parameter_values, 0, tf)
MI.plot_trajectories(time_list, state_list)


# Generate the observations
dt = 1
Observation_times_list = np.arange(dt, tf+dt, dt)
Y_list = MI.generate_observations(state_list, time_list, parameter_values, Observation_times_list)

########################################################################################################################
# inference

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
Initial_marginal_distribution['k3'].distribution = Initial_marginal_distribution['k3'].distribution / sum(
    Initial_marginal_distribution['k3'].distribution)


# inference
particle_size = 10000
Marginal_distributions_over_time, final_particles, time_result, mean_result, std_result \
    = MI.RB_inference_time_course_data(
        time_points_for_observations = Observation_times_list,
        Y_list=Y_list,
        particle_size=particle_size,
        marginal_distributions_dic= Initial_marginal_distribution)


########################################################################################################################
import pickle

data_to_save={
    # 'Identification Object': MI,
    'time_list': time_list,
    'state_list': state_list,
    'Observation_times_list': Observation_times_list,
    'Y_list': Y_list,
    #'Particles_list_returned': Particles_list_returned,
    'parameter_values': parameter_values,
    'time_result': time_result,
    'mean_result': mean_result,
    'std_result': std_result,
    'final_particles': final_particles,
    'margin_distributions_over_time': Marginal_distributions_over_time
}

file_name = 'data_example_4'+'_tf_'+str(tf)+'_particle_number_'+ str(particle_size) + '_two_gene_states.pkl'

with open(file_name, 'wb') as f:
    pickle.dump(data_to_save, f)
