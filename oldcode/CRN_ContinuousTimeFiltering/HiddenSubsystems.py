import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from CRN_Simulation_Inference.CRN_ContinuousTimeFiltering.SparseMatrixStructure import SparseMatrixStructure

class HiddenSubsystems:

    def __init__(self, parameters, species, unobservable_reactions, xi_list, observable_reactions):
        self.parameters = parameters # list of strings
        self.species = species # list of strings
        self.unobservable_reactions = unobservable_reactions # list of strings
        self.xi_list = xi_list # list of indices of xi
        self.observable_reactions = observable_reactions # list of reaction names


    def set_states(self, parameter_species_ordering, states):
        self.states = states # np array each row corresponds to a particular states
        self.parameter_species_ordering = parameter_species_ordering # {'k1': 0, 'k3': 1, 'k2': 2, 'mRNA': 3}
        self.map_from_state_to_index = self.construct_the_map_from_state_to_index()



    # method to construct a map from a state to its index in the array self.states
    def construct_the_map_from_state_to_index(self):
        y = np.arange(0, self.states.shape[0])
        map_from_state_to_index = LinearRegression().fit(self.states, y)
        return map_from_state_to_index

    def get_index_of_states(self, state_list):
        """

        :param state_list: a list of states
        :return: a list of indices for these states in array self.states
        """
        # print(np.round(self.map_from_state_to_index.predict(state_list)).astype(int))
        return np.round(self.map_from_state_to_index.predict(state_list)).astype(int)

    # structure of matrix
    def set_A_evolution_structure(self, A):
        self.A_evolution_structure = A

    def set_A_jump_structure(self, A, xi_ordering):
        self.A_jump_structure = A # a list of SparseMatrixStructure, the ordering is given in the 2nd line
        self.A_jump_xi_ordering = xi_ordering
