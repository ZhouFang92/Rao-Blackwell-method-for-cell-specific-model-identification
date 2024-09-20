
class DistributionOfSubsystems():

    def __init__(self, states, parameter_species_ordering):
        self.states = states # a list of states
        self.parameter_species_ordering = parameter_species_ordering
        self.time_list = []
        self.distribution_list = []

    # include distributions at different time points.
    def extend_distributions(self, time_list, distribution_list):
        self.time_list.extend(time_list)
        self.distribution_list.extend(distribution_list)

    def replace_distributions(self, time_list, distribution_list):
        self.time_list = time_list
        self.distribution_list = distribution_list

    ##############################
    # get methods
    ##############################

    def get_states_of_the_element(self, element):
        if element in self.parameter_species_ordering.keys():
            return self.states[:,self.parameter_species_ordering[element]]
        else:
            raise ValueError('The element is not in the parameter_species_ordering.')