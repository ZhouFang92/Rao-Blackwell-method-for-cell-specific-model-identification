import numpy as np

class Particle:

    def __init__(self, weight, states_dic, parameter_dic, follower_distributions):
        self.weight = weight
        self.states_dic = states_dic
        self.parameter_dic = parameter_dic
        self.follower_distributions = follower_distributions # DistributionOfSubsystems

    ##############################
    # get methods
    ##############################

    def get_follower_parameter_species_names(self):
        return [key for distribution in self.follower_distributions[1:] for key in distribution.parameter_species_ordering]

    def get_leader_species_names(self):
        follower_components = self.get_follower_parameter_species_names()
        return [key for key in self.states_dic.keys() if key not in follower_components]

    def get_membership_of_components(self, element):
        if element in self.get_leader_species_names():
            return 0
        for i in range(1, len(self.follower_distributions)):
            if element in self.follower_distributions[i].parameter_species_ordering.keys():
                return i


    ##############################
    # update methods
    ##############################

    # reconstruct the particle from the distribution
    def reconstruct_particle(self):
        states_dic = {key: self.states_dic[key] for key in self.get_leader_species_names()}
        parameter_dic = {}
        for i in range(1, len(self.follower_distributions)):
            # select states randomly from the distribution
            this_subsystem_states = self.follower_distributions[i].states
            this_subsystem_distribution = self.follower_distributions[i].distribution_list[-1]
            this_parameter_species_ordering = self.follower_distributions[i].parameter_species_ordering
            state_index = np.random.choice(range(len(this_subsystem_states)), p=this_subsystem_distribution.flatten())
            parameter_species_dic = {key: this_subsystem_states[state_index][this_parameter_species_ordering[key]]
                                     for key in this_parameter_species_ordering}
            states_dic.update({key: parameter_species_dic[key] for key in parameter_species_dic.keys() if key in self.states_dic.keys()})
            parameter_dic.update({key: parameter_species_dic[key] for key in parameter_species_dic.keys() if key in self.parameter_dic.keys()})
        self.states_dic = states_dic
        self.parameter_dic = parameter_dic


    # update the weight of the particle
    def update_weight(self, weight):
        self.weight = weight


    # update the states of the particle
    def update_states(self, state_dic):
        self.states_dic = state_dic


    # pdate the follower distributions
    def update_follower_distributions(self, follower_distributions):
        self.follower_distributions = follower_distributions
