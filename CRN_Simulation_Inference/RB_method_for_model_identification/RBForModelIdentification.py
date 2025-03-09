import numpy as np
import pandas as pd
import igraph as ig
import matplotlib.pyplot as plt
import math
import scipy
from tqdm import tqdm
import inspect
import bisect
from joblib import Parallel, delayed
from multiprocessing import get_context
import time
import random
import copy

# pathos
import pathos.multiprocessing as mp

import os

from CRN_Simulation_Inference.CRN_Simulation.CRN import CRN
from CRN_Simulation_Inference.CRN_ContinuousTimeFiltering.CRNForContinuousTimeFiltering import CRNForContinuousTimeFiltering
from CRN_Simulation_Inference.CRN_ContinuousTimeFiltering.DistributionOfSubsystems import DistributionOfSubsystems
from CRN_Simulation_Inference.RB_method_for_model_identification.MarginalDistribution import MarginalDistribution
from CRN_Simulation_Inference.RB_method_for_model_identification.Particle import Particle

# This class only contains 2 parts: 1. leader follower decomposition 2. RB-algorithm for model identification


class RBForModelIdentification(CRN):

    def __init__(self, stoichiometric_matrix, species_names, parameters_names,
                 reaction_names, propensities, h_function, observation_noise_intensity,
                 range_of_species, range_of_parameters, discretization_size_parameters,
                 maximum_size_of_each_follower_subsystem=1000):
        """

        :param stoichiometric_matrix:
        :param species_names:
        :param parameters_names:
        :param reaction_names:
        :param propensities:
        :param h_function: lambda function for observations
        """
        super().__init__(stoichiometric_matrix, species_names, parameters_names, reaction_names, propensities)
        self.range_of_species = range_of_species
        self.range_of_parameters = range_of_parameters
        self.discretization_size_parameters = discretization_size_parameters

        self.observation_noise_intensity = observation_noise_intensity  # a dictionary
        self.observation_noise_intensity_dependencies = [inspect.getfullargspec(p).args for p in self.observation_noise_intensity]

        self.h_function = h_function
        self.h_function_dependencies = [inspect.getfullargspec(p).args for p in self.h_function]

        # TODO: check if the input are eglibible
        # check whether the length of h_function and observation_noise_intensity are the same
        if len(self.h_function) != len(self.observation_noise_intensity):
            raise ValueError('The length of h_function and observation_noise_intensity are not the same!')
            return
        # the union of self.h_function_dependencies and self.observation_noise_dependecies should be subset of self.species_names and self.parameters_names
        union_of_h_dependencies = set().union(*self.h_function_dependencies)
        union_of_observation_noise_dependencies = set().union(*self.observation_noise_intensity_dependencies)
        union_of_dependencies = union_of_h_dependencies.union(union_of_observation_noise_dependencies)
        if not union_of_dependencies.issubset(set(self.species_names).union(self.parameters_names)):
            raise ValueError('The h function or observation noise has undefined species or parameters!')
            return

        # Do the leader follower decomposition for time course data
        self.maximum_size_of_each_follower_subsystem = maximum_size_of_each_follower_subsystem
        self.leader_species_time_course_data, self.leader_follower_decomposition_result_time_course_data =\
            self.first_level_decomposition_time_course_data(max_size_of_subsystems=maximum_size_of_each_follower_subsystem)


        # prepare the filtered CME
        self.leader_follower_decomposition_result_time_course_data.set_up_preparations_for_filtering()

    ###############################
    # get methods for the RB class
    ###############################

    def get_follower_species_time_course_data(self):
        return self.leader_follower_decomposition_result_time_course_data.hidden_species

    def get_follower_parameters_time_course_data(self):
        return self.leader_follower_decomposition_result_time_course_data.unkown_parameters

    def get_observation_membership(self, observation_index):
        # return the index of the subsystem that the observation belongs to
        subsystems = self.leader_follower_decomposition_result_time_course_data.subsystems
        species_parameters_in_this_observation = self.h_function_dependencies[observation_index] + \
                                                 self.observation_noise_intensity_dependencies[observation_index]
        species_parameters_in_this_observation_no_leader = \
            set(species_parameters_in_this_observation).difference(self.leader_species_time_course_data)
        if len(species_parameters_in_this_observation_no_leader) == 0:
            return 0

        for i in range(1, len(subsystems)):
            if species_parameters_in_this_observation_no_leader.issubset(subsystems[i].parameter_species_ordering):
                return i

    def get_observation_membership_for_all_observations(self):
        return [self.get_observation_membership(i) for i in range(len(self.h_function))]

    def get_number_of_follower_subsystems(self):
        return self.leader_follower_decomposition_result_time_course_data.get_number_of_subsystem()

    def get_size_of_follower_subsystems(self):
        return self.leader_follower_decomposition_result_time_course_data.get_size_of_subsystems()

    def get_size_of_space_for_each_species(self):
        df_size_species_space = self.range_of_species['max'].sub(self.range_of_species['min'])
        df_size_species_space = df_size_species_space.add(1)
        return df_size_species_space.to_frame()

    def get_space_for_species(self, species):
        state_space = []
        for i in range(self.range_of_species.loc[species]['min'], self.range_of_species.loc[species]['max']+1):
            state_space.append([i])
        return np.array(state_space)

    ###############################
    # leader-follower decomposition algorithms
    ###############################

    def second_level_decomposition_time_course_data(self, leader_species):
        observable_species = leader_species
        range_of_hidden_species  = self.range_of_species.drop(leader_species)
        group_hidden_species_and_parameters = \
            [ list(set(self.h_function_dependencies[i]).union(self.observation_noise_intensity_dependencies[i]))
                                 for i in range(len(self.h_function_dependencies))]

        CF = CRNForContinuousTimeFiltering(stoichiometric_matrix=self.stoichiometric_matrix,
                                           species_names=self.species_names,
                                           parameters_names=self.parameters_names,
                                           reaction_names=self.reaction_names,
                                           propensities=self.propensities,
                                           observable_species=observable_species,
                                           range_of_species=range_of_hidden_species,
                                           range_of_parameters=self.range_of_parameters,
                                           discretization_size_parameters=self.discretization_size_parameters,
                                           group_hidden_species_and_parameters=group_hidden_species_and_parameters
                                           )

        return CF

    def first_level_decomposition_time_course_data(self, max_size_of_subsystems=1000):
        # initialization
        optimal_size = 0
        optimal_leader_species = []

        # traverse all choices of leader species
        leader_system_choices = self.generate_subsets(set(self.species_names))
        for leader_species in leader_system_choices:
            # calculate the size of the second level decomposition
            CF = self.second_level_decomposition_time_course_data(list(leader_species))
            size_of_subsystems = np.prod(CF.get_size_of_subsystems()[1:])
            size_of_the_largest_subsystem = CF.get_size_of_the_largest_subsystem()

            # update the optimal leader species
            if size_of_subsystems > optimal_size and size_of_the_largest_subsystem < max_size_of_subsystems:
                optimal_size = size_of_subsystems
                optimal_leader_species = leader_species

        if optimal_size == 0:
            raise ValueError("No egligble leader-follower decomposition is found with the given maximum size of systems.")

        return list(optimal_leader_species), self.second_level_decomposition_time_course_data(list(optimal_leader_species))

    def generate_subsets(self, s):
        """
        s is a set
        :return:
        """
        # If the set is empty, return a set with the empty set
        if len(s) == 0:
            return [set([])]
        # Get the first element of the set
        x = next(iter(s))
        # Recursively generate subsets of the remaining elements
        subsets = self.generate_subsets(s - {x})
        # Add x to each subset to get new subsets that include x
        new_subsets = [subset.union({x}) for subset in subsets]
        # Return the union of the original and new subsets
        subsets.extend(new_subsets)
        return subsets

    ###############################
    # Generate observations
    ###############################

    def get_args(self, function, state_parameters_dic):
        args = inspect.getfullargspec(function).args
        return [state_parameters_dic[arg] for arg in args]

    def _eval_h_function(self, state_parameters_dic):
        h = np.zeros(len(self.h_function))
        for i, h_i in enumerate(self.h_function):
            h[i] = h_i(* self.get_args(h_i, state_parameters_dic))
        return h

    def _eval_observation_noise_intensity(self, state_parameters_dic):
        noise = np.zeros(len(self.observation_noise_intensity))
        for i, noise_i in enumerate(self.observation_noise_intensity):
            noise[i] = noise_i(* self.get_args(noise_i, state_parameters_dic))
        return noise

    def generate_observation_noise(self, state_parameters_dic):
        intensity = self._eval_observation_noise_intensity(state_parameters_dic)
        return np.random.normal(0, intensity)

    def generate_observations(self, state_list, time_list, parameter_dic, time_points_for_observations,
                              species_ordering=None):
        """

        :param state_list: a list of states for the system indicating the trajectories
        :param time_list: a list of jumping times
        :param time_points_for_observations: the time points for measurements
        :return: a list of measurements

        TODO: this can be implemented in O(n)
        """

        if species_ordering is None:
            species_ordering = self.species_ordering

        # generate measurements for each observation time point
        measurements = []
        for t in time_points_for_observations:
            # find the index of the state at time t
            index = bisect.bisect_right(time_list, t)
            index = index - 1

            # get the state at time t
            state = state_list[index]

            # generate the measurement
            state_parameters_dic = dict(zip(species_ordering, state))
            state_parameters_dic.update(parameter_dic)
            measurement = self._eval_h_function(state_parameters_dic) + self.generate_observation_noise(state_parameters_dic)
            measurements.append(measurement)

        return measurements



    ###############################
    #  Inference algorithm for time course data
    ###############################

    def RB_inference_time_course_data(self, time_points_for_observations, Y_list,  particle_size,
                                      marginal_distributions_dic, tqdm_disable=False):
        # initialization, the decomposition has been obtained in self.leader_follower_decomposition_result_time_course_data
        Particles_list_to_return = []

        CF = self.leader_follower_decomposition_result_time_course_data
        particles = self.sample_particles_from_marginal_distributions(marginal_distributions_dic, particle_size)

        Marginal_distributions_over_time = [] # a list of marginal distributions over time
        mean_out = {element: [] for element in self.species_names + self.parameters_names}  # a dictionary of list
        std_out = {element: [] for element in self.species_names + self.parameters_names}  # a dictionary of list

        # do the inference for each time point
        t_current = 0
        for t, Y in tqdm( list(zip(time_points_for_observations, Y_list)), desc="Progress", unit="Measurements", position=0, disable=tqdm_disable):
            # reconstruct the particles
            start_time = time.time()
            [particle.reconstruct_particle() for particle in particles]
            end_time = time.time()
            # print("reconstruct the particles time: ", end_time - start_time)

            # prediction step
            start_time = time.time()
            particles = self.prediction_step(CF, particles, t_current, t)
            end_time = time.time()
            # print("prediction step time: ", end_time - start_time)

            # update step
            start_time = time.time()
            self.update_step_time_course_data(t, particles, Y)
            end_time = time.time()
            # print("update step time: ", end_time - start_time)

            # save the result
            start_time = time.time()
            # Particles_list_to_return.append(copy.deepcopy(particles))
            marginal_distributions = self.extract_marginal_distribution(particles)
            Marginal_distributions_over_time.append(marginal_distributions)
            mean_out_time_t, std_out_time_t = self.extract_mean_std_esimtates_from_marginal_distributions(marginal_distributions)
            for element in self.species_names + self.parameters_names:
                mean_out[element].extend(mean_out_time_t[element])
                std_out[element].extend(std_out_time_t[element])
            end_time = time.time()
            # print("save the result time: ", end_time - start_time)

            # resample step
            if t == time_points_for_observations[-1]:
                continue
            start_time = time.time()
            particles = self.resample_particles(particles)
            end_time = time.time()
            # print("resample step time: ", end_time - start_time)

            t_current = t

        #return Particles_list_to_return
        return Marginal_distributions_over_time, particles, time_points_for_observations, mean_out, std_out



    # generate uniform marginal distributions for every species and parameters
    def generate_uniform_marginal_distributions(self):
        Marginal_distributions = {}
        # transverse all species
        for species in self.species_names:
            states = list(range(self.range_of_species.loc[species, 'min'], self.range_of_species.loc[species, 'max'] + 1))
            uniform_distribution = np.ones(len(states)) / len(states)
            marginal_uniform_distribution = MarginalDistribution(species, states, uniform_distribution)
            Marginal_distributions.update({species: marginal_uniform_distribution})
        # transverse all parameters
        for parameter in self.parameters_names:
            states = np.linspace(self.range_of_parameters.loc[parameter, 'min'], self.range_of_parameters.loc[parameter, 'max'],
                           self.discretization_size_parameters.loc[parameter, 0])
            uniform_distribution = np.ones(len(states)) / len(states)
            marginal_uniform_distribution = MarginalDistribution(parameter, states, uniform_distribution)
            Marginal_distributions.update({parameter: marginal_uniform_distribution})
        # return a dictionary of uniform marginal distributions
        return Marginal_distributions

    # sample particles from the marginal distributions
    def sample_particles_from_marginal_distributions(self, marginal_distributions_dic, particle_size):
        # initialization
        particles = []

        weight = 1 / particle_size

        follower_distributions = [[]]
        # the distribution of follower systems
        for i in range(1, self.leader_follower_decomposition_result_time_course_data.get_number_of_subsystem()+1):
            states = self.leader_follower_decomposition_result_time_course_data.subsystems[i].states
            parameter_species_ordering = self.leader_follower_decomposition_result_time_course_data.subsystems[i].parameter_species_ordering
            # the distribution of follower systems
            distribution = np.zeros((len(states),1))
            for j in range(len(states)):
                distribution_value = 1
                for element in parameter_species_ordering:
                        state_of_the_element = states[j][parameter_species_ordering[element]]
                        index_of_state_value = np.where(marginal_distributions_dic[element].states == state_of_the_element)[0]
                        distribution_value = distribution_value * marginal_distributions_dic[element].distribution[index_of_state_value]
                distribution[j] = distribution_value
            # construct the distribution of hidden systems
            distribution_class = DistributionOfSubsystems(states, parameter_species_ordering)
            distribution_class.extend_distributions([0], [distribution])
            follower_distributions.append(distribution_class)

        # sample particles
        for i in range(particle_size):
            # constrcut the species state
            state_dic = {}
            for species in self.species_names:
                marginal_distribution = marginal_distributions_dic[species]
                state_dic.update({species: np.random.choice(marginal_distribution.states, p=marginal_distribution.distribution.flatten())})
            # constrcut the parameter state
            parameter_dic = {}
            for parameter in self.parameters_names:
                marginal_distribution = marginal_distributions_dic[parameter]
                parameter_dic.update({parameter: np.random.choice(marginal_distribution.states, p=marginal_distribution.distribution.flatten())})
            # construct the particle
            particle = Particle(states_dic= state_dic, parameter_dic=parameter_dic, weight=weight,
                                follower_distributions=follower_distributions)
            particles.append(particle)

        return particles


    def prediction_step(self, CF, particles, t_current, t_next):
        # simulation of the particles and denote the leader system
        time_leader = []
        leader_trajectories = []
        leader_ordering = []
        leader_species = particles[0].get_leader_species_names()
        # for particle in particles:
        #     # simulate the particles
        #     time_out, state_out = self.SSA(particle.states_dic, particle.parameter_dic, t_current, t_next)
        #     particle.update_states(
        #         {key: state_out[-1][self.species_ordering[key]] for key, i in zip(self.species_ordering, range(len(self.species_ordering)))})
        #     # denote the leader system
        #     Time_leader_temp, leader_trajectory_temp, leader_ordering_temp\
        #         = self.extract_trajectory(time_out, state_out, leader_species)
        #     time_leader.append(Time_leader_temp)
        #     leader_trajectories.append(leader_trajectory_temp)
        #     leader_ordering.append(leader_ordering_temp)

        # results = Parallel(n_jobs=-1, backend='loky')(
        #     delayed(self.SSA)(
        #         particle.states_dic,
        #         particle.parameter_dic,
        #         t_current,
        #         t_next
        #     ) for particle in particles
        # )
        # rewrite using pathos

        results = mp.ProcessingPool().map(
            lambda particle: self.SSA(particle.states_dic, particle.parameter_dic, t_current, t_next),
            particles
        )

        for j in range(len(results)):
            time_out, state_out = results[j]
            particle = particles[j]
            particle.update_states(
                {key: state_out[-1][self.species_ordering[key]] for key in self.species_names})
            # denote the leader system
            Time_leader_temp, leader_trajectory_temp, leader_ordering_temp\
                = self.extract_trajectory(time_out, state_out, leader_species)
            time_leader.append(Time_leader_temp)
            leader_trajectories.append(leader_trajectory_temp)
            leader_ordering.append(leader_ordering_temp)



        # compute the conditional distribution of the follower systems
        # record initial distributions
        Initial_distributions_list = []
        for particle in particles:
            Initial_Distributions = [[]]
            for i in range(1, len(particle.follower_distributions)):
                Initial_Distributions.append(particle.follower_distributions[i].distribution_list[-1])
            Initial_distributions_list.append(Initial_Distributions)
        # compute the conditional distribution
        results = Parallel(n_jobs=-1, backend='loky')(
            delayed(CF.filteringFFSP_return_final_distribution)(
                Y_trajectory=leader_trajectories[i],
                Y_ordering=leader_ordering[i],
                Time_Y=time_leader[i],
                Initial_Distributions=Initial_distributions_list[i],
                tqdm_disable=True
            ) for i in range(len(particles))
        )



        # update the follower distributions
        for i in range(len(particles)):
            particles[i].update_follower_distributions(results[i])

        return particles

    def update_step_time_course_data(self, t, particles, Y):
        # compute the likelihood of each particle
        Likelihood_results = Parallel(n_jobs=-1, backend='loky')(
            delayed(self.Likelihood_time_course_data)(Y, particle)
            for particle in particles
        )
        # Likelihood_results = []
        # for particle in particles:
        #     Likelihood_results.append(self.Likelihood_time_course_data(Y, particle))


        # update the weight of each particle
        # Parallel(n_jobs=-1)(
        #     delayed(self.update_particles_according_to_the_likelihood)(t, particle, likelihood_list)
        #     for particle, likelihood_list in zip(particles, Likelihood_results)
        # )
        for particle, likelihood_list in zip(particles, Likelihood_results):
            self.update_particles_according_to_the_likelihood(t, particle, likelihood_list)


    def Likelihood_time_course_data(self, Y, particle):
        """

        :param Y:
        :param particle:
        :return: the likelihood of follower_subsystems are a row vector
        """
        # compute the likelihood for each particle
        Likelihood_list = []
        leader_species = particle.get_leader_species_names()
        observation_membership = self.get_observation_membership_for_all_observations()

        # compute the likelihood corresponding to the leader system
        likelihood = 1
        if 0 in observation_membership: # if the there is an observation for the leader system only
            state_parameter_dic = particle.states_dic.copy()
            state_parameter_dic.update(particle.parameter_dic)
            for observation_index in range(len(self.h_function)): # transverse all observations
                if observation_membership[observation_index] == 0:
                    h_args = self.get_args(self.h_function[observation_index], state_parameter_dic)
                    h_value = self.h_function[observation_index](*h_args)
                    noise_intensity_args = self.get_args(self.observation_noise_intensity[observation_index], state_parameter_dic)
                    noise_intensity_value = self.observation_noise_intensity[observation_index](*noise_intensity_args)
                    likelihood = likelihood * np.exp(- (Y[observation_index] - h_value)**2/ (2 * noise_intensity_value**2))/(np.sqrt(2*np.pi)*noise_intensity_value)
        Likelihood_list.append(likelihood)


        # compute the likelihood corresponding to the follower systems
        for i in range(1, self.get_number_of_follower_subsystems()+1): # transverse all follower systems
            if i not in observation_membership: # if there is no observation for the follower system
                Likelihood_list.append(1)
                continue
            likelihood = np.ones(self.get_size_of_follower_subsystems()[i])
            # construct the state_parameter_dic
            state_parameter_dic = {species: np.full(self.get_size_of_follower_subsystems()[i], particle.states_dic[species])
                 for species in leader_species}  # vectors of states and parameters
            follower_distribution = particle.follower_distributions[i]
            state_parameter_dic.update(
                dict(zip(follower_distribution.parameter_species_ordering.keys(), follower_distribution.states.T)))
            for observation_index in range(len(self.h_function)): # transverse all observations
                if observation_membership[observation_index] == i:
                    h_args = self.get_args(self.h_function[observation_index], state_parameter_dic)
                    h_value = self.h_function[observation_index](*h_args)
                    noise_intensity_args = self.get_args(self.observation_noise_intensity[observation_index], state_parameter_dic)
                    noise_intensity_value = self.observation_noise_intensity[observation_index](*noise_intensity_args)
                    likelihood = likelihood * np.exp(- (Y[observation_index] - h_value)**2/ (2 * noise_intensity_value**2))/(np.sqrt(2*np.pi)*noise_intensity_value)
            Likelihood_list.append(likelihood) # record the result

        return Likelihood_list

    def update_particles_according_to_the_likelihood(self, t, particle, Likelihood_list):
        # update the weight of each particle

        observation_membership = self.get_observation_membership_for_all_observations()

        new_weight = 1 * Likelihood_list[0]
        for i in range(1, len(Likelihood_list)):
            if i not in observation_membership:
                continue
            new_distribution = Likelihood_list[i].reshape(-1,1) * particle.follower_distributions[i].distribution_list[-1]
            new_distribution_sum = np.sum(new_distribution)
            new_weight = new_weight * new_distribution_sum
            # update the distribution
            if new_distribution_sum >= 1.0e-10:
                new_distribution = new_distribution / new_distribution_sum
            else:
                new_distribution = np.ones(new_distribution.shape) / new_distribution.shape[0]
            particle.follower_distributions[i].replace_distributions([t], [new_distribution])

        # update the weight
        if new_weight == float('inf') or new_weight == float('-inf') or new_weight == float('nan'):
            new_weight = 0
        particle.update_weight(new_weight)

    def resample_particles(self, particles):
        new_particles = []
        weights = [particle.weight for particle in particles]
        index = self.resampling(weights)
        for i in index:
            new_particles.append(copy.deepcopy(particles[i]))
        # new weights
        for particle in new_particles:
            particle.update_weight(1/len(new_particles))

        return new_particles

    def resampling(self, weights):
        # the algorithm in the book fundamentals of stochastic filtering
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else: # if all weights are zero give the uniform weights
            weights = np.ones(len(weights)) / len(weights)
            print("All weights are zero, give the uniform weights at this step!")
        o = np.zeros(len(weights)) # offsprings
        n = len(weights)
        g = n
        h = n
        for j in range(n-1):
            u = random.uniform(0, 1)
            if self.fraction_part(n*weights[j]) + self.fraction_part(g-n*weights[j]) < 1 and self.fraction_part(g) > 0:
                if u < 1 - self.fraction_part(n*weights[j]) / self.fraction_part(g):
                    o[j] = math.trunc(n*weights[j])
                else:
                    o[j] = math.trunc(n*weights[j]) + h - math.trunc(g)
            else:
                if u < 1 - (1-self.fraction_part(n*weights[j])) / (1-self.fraction_part(g)):
                    o[j] = math.trunc(n*weights[j])+1
                else:
                    o[j] = math.trunc(n*weights[j]) + h - math.trunc(g)
            g = g - n*weights[j]
            h = h - o[j]
        o[-1] = h

        index = []
        for i in range(len(o)):
            index.extend([i]*int(o[i]))

        return index

    def fraction_part(self, number):
        return number - math.trunc(number) # return the fraction part of a number


    ##########################################
    # plot the results
    ##########################################

    def extract_marginal_distribution(self, particles):
        """
        :param particles:
        :return: a list of marginal distributions
        """
        marginal_distributions = []
        t = particles[0].follower_distributions[1].time_list[-1]
        leader_species = particles[0].get_leader_species_names()
        follower_components = particles[0].get_follower_parameter_species_names()

        # leader component
        for species in leader_species:
            # construct the marginal distribution class
            marginal_distribution = DistributionOfSubsystems(self.get_space_for_species(species),{species: 0})
            # compute the distribution
            distribution = np.zeros((self.get_size_of_space_for_each_species().loc[species, 0], 1))
            for particle in particles:
                index = np.where(marginal_distribution.states == particle.states_dic[species])[0] # find the index of the state in the states
                distribution[index, 0] = distribution[index, 0] + particle.weight
            marginal_distribution.replace_distributions([t], [distribution/np.sum(distribution)])
            # save the marginal distribution
            marginal_distributions.append(marginal_distribution)

        # follower species
        for subsystem_index in range(1, len(particles[0].follower_distributions)):
            # summarize the marginal distribution of the subsystem
            states = particles[0].follower_distributions[subsystem_index].states
            parameter_species_ordering = particles[0].follower_distributions[subsystem_index].parameter_species_ordering
            # compute the distribution
            distribution = np.zeros((len(states), 1))
            for particle in particles:
                distribution = distribution + particle.weight * particle.follower_distributions[subsystem_index].distribution_list[-1]
            distribution = distribution / np.sum(distribution)

            # extract the marginal distribution of each parameter and species
            for element in parameter_species_ordering.keys():
                element_index = parameter_species_ordering[element]
                element_states = np.unique(states[:, element_index], axis = 0).reshape(-1, 1)
                # construct the marginal distribution class
                marginal_distribution = DistributionOfSubsystems(element_states, {element: 0})
                # constrcut the distribution
                element_distribution = np.zeros((len(element_states), 1))
                for state, i in zip(states, range(len(states))):
                    index = np.where(element_states == state[element_index])[0]
                    element_distribution[index, 0] = element_distribution[index, 0] + distribution[i, 0]
                marginal_distribution.replace_distributions([t], [element_distribution])
                # save the marginal distribution
                marginal_distributions.append(marginal_distribution)

        return marginal_distributions

    def plot_marginal_distribution(self, marginal_distributions, parameter_real_values = None, species_real_values = None):

        rows = math.ceil(len(marginal_distributions) / 2)
        columns = 2
        fig, axs = plt.subplots(math.ceil(len(marginal_distributions) / 2), 2, figsize=(columns * 3, rows * 3))
        fig.subplots_adjust(wspace=0.5, hspace=1)

        # sort the distribution according to the parameters and names; it should follows the order of the parameters and species
        parameter_species_names = self.parameters_names + self.species_names
        marginal_distributions = sorted(marginal_distributions, key=lambda x: parameter_species_names.index(next(iter(x.parameter_species_ordering))))

        for i, ax in enumerate(axs.flatten()):
            if i >= len(marginal_distributions):
                break
            distribution = marginal_distributions[i].distribution_list[-1].copy()
            states = marginal_distributions[i].states.copy()
            name = next(iter(marginal_distributions[i].parameter_species_ordering))
            ax.bar(states.reshape(-1), distribution.reshape(-1), width=states[1] - states[0], label='Estimated distribution')
            # plot the real value
            if name in self.parameters_names and parameter_real_values is not None:
                ax.axvline(x=parameter_real_values[name], color='r', linestyle='--', label='real value')
            if name in self.species_names and species_real_values is not None:
                ax.axvline(x=species_real_values[name], color='r', linestyle='--', label='real value')
            ax.set_xlabel(name)
            ax.set_ylabel('Probability')



    def plot_marginal_distribution_from_particles(self, particles, parameter_real_values = None, species_real_values = None):
        """

        :param particles:
        :param parameter_real_values: a dictionary
        :param species_real_values: a dictionary
        :return:

        """
        marginal_distributions = self.extract_marginal_distribution(particles)

        self.plot_marginal_distribution(marginal_distributions, parameter_real_values, species_real_values)


    def extract_mean_std_estimates_over_time(self, particles_list):
        # to extract the mean and std of the estimates over time
        time_out = []
        mean_out = {element: [] for element in self.species_names + self.parameters_names} # a dictionary of list
        std_out = {element: [] for element in self.species_names + self.parameters_names} # a dictionary of list

        for particles in particles_list: # transverse all the time
            time_out.append(particles[0].follower_distributions[1].time_list[-1])
            marginal_distributions = self.extract_marginal_distribution(particles)
            for marginal_distribution in marginal_distributions:
                parameter_species_name = next(iter(marginal_distribution.parameter_species_ordering))
                mean = np.sum(marginal_distribution.distribution_list[-1] * marginal_distribution.states, axis=0)
                std = np.sqrt(np.sum(marginal_distribution.distribution_list[-1] * (marginal_distribution.states - mean) ** 2, axis=0))
                mean_out[parameter_species_name].append(mean)
                std_out[parameter_species_name].append(std)

        return time_out, mean_out, std_out


    def extract_mean_std_esimtates_from_marginal_distributions(self, marginal_distributions):
        mean_out = {element: [] for element in self.species_names + self.parameters_names}
        std_out = {element: [] for element in self.species_names + self.parameters_names}

        for marginal_distribution in marginal_distributions:
            parameter_species_name = next(iter(marginal_distribution.parameter_species_ordering))
            mean = np.sum(marginal_distribution.distribution_list[-1] * marginal_distribution.states, axis=0)
            std = np.sqrt(np.sum(marginal_distribution.distribution_list[-1] * (marginal_distribution.states - mean) ** 2, axis=0))
            mean_out[parameter_species_name].append(mean)
            std_out[parameter_species_name].append(std)

        return mean_out, std_out


    def plot_mean_std_estimates_over_time(self, time_out, mean_out, std_out, real_parameters=None):
        rows = math.ceil( len(mean_out) / 2)
        columns = 2
        fig, axs = plt.subplots(rows, 2, figsize=(columns * 3, rows * 3))
        fig.subplots_adjust(wspace=0.5, hspace=1)

        axs = axs.flatten()

        # plot the species results
        figure_index = 0
        # color_index = 0
        for species in self.species_names:
            ax = axs[figure_index]
            figure_index = figure_index + 1
            # selected_color = plt.rcParams['axes.prop_cycle'].by_key()['color'][color_index]
            # color_index = color_index + 1
            selected_color = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
            ax.step(time_out, mean_out[species], label=species + ' (estimates)', where='post', color=selected_color)
            ax.step(time_out, mean_out[species], label=species + ' (estimates)', where='post')
            mean_out[species] = np.array(mean_out[species]).reshape(-1)
            std_out[species] = np.array(std_out[species]).reshape(-1)
            ax.fill_between(time_out, mean_out[species] - std_out[species], mean_out[species] + std_out[species],
                            alpha=0.3)
            ax.set_ylabel(species)
            ax.set_xlabel('Time')
        # plt.legend()

        # plot the parameter results
        # color_index = 0
        for parameter in self.parameters_names:
            ax = axs[figure_index]
            figure_index = figure_index + 1
            # selected_color = plt.rcParams['axes.prop_cycle'].by_key()['color'][color_index]
            # color_index = color_index + 1
            selected_color = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
            if real_parameters is not None:
                ax.plot(time_out, [real_parameters[parameter]] * len(time_out), label=parameter + ' (real)',
                        color='red', linestyle='--')
            ax.step(time_out, mean_out[parameter], label=parameter + ' (estimates)', where='post', color=selected_color)
            mean_out[parameter] = np.array(mean_out[parameter]).reshape(-1)
            std_out[parameter] = np.array(std_out[parameter]).reshape(-1)
            ax.fill_between(time_out, mean_out[parameter] - std_out[parameter],
                            mean_out[parameter] + std_out[parameter], alpha=0.3, color=selected_color)
            ax.set_ylabel(parameter)
            ax.set_xlabel('Time')
        # plt.legend()


    def plot_mean_std_estimates_over_time_from_particles(self, particles_list, real_parameters=None):
        time_out, mean_out, std_out = self.extract_mean_std_estimates_over_time(particles_list)

        self.plot_mean_std_estimates_over_time(time_out, mean_out, std_out, real_parameters)