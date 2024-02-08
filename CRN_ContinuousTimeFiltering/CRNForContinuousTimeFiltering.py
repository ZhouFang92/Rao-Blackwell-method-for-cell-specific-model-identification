import numpy as np
import pandas as pd
import igraph as ig
import matplotlib.pyplot as plt
import math
import scipy
from tqdm import tqdm

from CRN_Simulation.CRN import CRN
from CRN_ContinuousTimeFiltering.HiddenSubsystems import HiddenSubsystems
from CRN_ContinuousTimeFiltering.SparseMatrixStructure import SparseMatrixStructure
from CRN_ContinuousTimeFiltering.DistributionOfSubsystems import DistributionOfSubsystems
from CRN_ContinuousTimeFiltering.MatrixExponentialKrylov import MatrixExponentialKrylov

# for debuging
import time

# This file defines the class for continuous time filtering and give some basic parts.

class CRNForContinuousTimeFiltering(CRN):
    
    def __init__(self, stoichiometric_matrix, species_names, parameters_names, reaction_names, propensities, \
                 observable_species, range_of_species, range_of_parameters, discretization_size_parameters, \
                 known_parameters = [], group_hidden_species_and_parameters=[] ):
        """
        To set up the basis of filtering problem

        :param observable_species: list of strings
        :param known_parameters: list of strings
        :param range_of_species: Data Frame of nx2 where n is the number of species; the truncated size of
                                    species space
        :param range_of_parameters: Data Frame of mx2 where m is the number of parameters
        :param discretization_size_parameters: Data Frame of m1x1 where m1 is the number of unknown parameters; the
                                                values shows the sizes of the discretized parameter spaces.
        :return:
        """
        super().__init__(stoichiometric_matrix, species_names, parameters_names, reaction_names, propensities)
        # DataFrame version of the stoichiometric matrix
        self.stoichiometric_matrix_df = pd.DataFrame(self.stoichiometric_matrix, index=self.species_names, columns=reaction_names)

        # set up basic information
        self.observable_species = observable_species # list of strings
        self.hidden_species = list( set(self.species_names).difference(self.observable_species) )
        self.known_parameters = known_parameters # list of strings
        self.unkown_parameters = list( set(self.parameters_names).difference(self.known_parameters) )

        # Define xi, observable reactions, observable reactions associated with xi
        # xi is the stoichiometric matrix of the observable speceis, and the reactions are grouped so they are different and nonzero
        self.xi = self.create_xi()
        self.xi.columns = ['xi_' + str(k) for k in np.arange(1, self.get_number_of_xi() + 0.1, 1, dtype=int)]

        # observable reactions compatible with xi: a list of lists
        self.O_xi = self.create_O_xi()
        self.xi_ordering = { self.xi.columns[i]:i for i in range(len(self.xi.columns)) }
        self.observable_reaction = list(set().union(*self.O_xi)) # list
        self.unobservable_reaction = \
            list( set(self.reaction_names).difference(self.observable_reaction) )

        # Decomposition of the hidden system, observable/unobservable reactions associated with each reaction
        self.group_hidden_species_and_parameters = group_hidden_species_and_parameters
        self.graph_for_decomposing_hidden_system = self.create_the_graph_for_decomposition()
        # compute the connected components of the graph
        self.connected_components_of_the_graph = \
            self.graph_for_decomposing_hidden_system.components(mode='weak')
        # create a dataframe to show which subsystem each parameter, species, reaction, xi belongs to
        self.membership = self.create_membership()
        self.subsystems = self.identify_subsystems()

        # Related to the finite state projection method
        if sorted(range_of_species.index.tolist()) != sorted(self.hidden_species):
            print("Error in definition the filtering class: range_of_species is not compatible with the unknown species")
        else:
            self.range_of_species = range_of_species # pandas objects
            # row --> species, first column --> the minimum value of the range, second column --> the maximum value of the range
        if sorted(range_of_parameters.index.tolist()) != sorted(self.unkown_parameters):
            print("Error in definition the filtering class: range_of_parameters is not compatible with the unknown species")
        else:
            self.range_of_parameters = range_of_parameters # pandas objects
            # row --> parameters, first column --> the minimum value of the range, second column --> the maximum value of the range
        if sorted(discretization_size_parameters.index.tolist()) != sorted(self.unkown_parameters):
            print("Error in definition the filtering class: discretization_size_parameters is not compatible with the unknown species")
        else:
            self.discretization_size_parameters = discretization_size_parameters
        self.prepared_for_filtering = False # to indicate whether the class is prepared for filtering

    #######################################
    #                                   ###
    # Methods for defining the class    ###
    #                                   ###
    #######################################

    def create_xi(self):
        stoichiometric_submatrix_observable_species = self.stoichiometric_matrix_df.loc[self.observable_species, :]
        xi = np.unique(stoichiometric_submatrix_observable_species, axis=1)
        xi = xi[:, ~np.all(xi == 0, axis=0)]  # remove zero vectors
        return pd.DataFrame(xi, index=self.observable_species)


    def create_O_xi(self):
        stoichiometric_submatrix_observable_species = self.stoichiometric_matrix_df.loc[self.observable_species, :]
        O_xi=[]
        for col_xi in self.xi.columns:
            O_xi.append([col for col in stoichiometric_submatrix_observable_species.columns \
                              if (self.xi[col_xi] == stoichiometric_submatrix_observable_species[col]).all()])
        return O_xi

    def create_the_graph_for_decomposition(self):
        """
        :return: self.graph_for_decomposing_hidden_system
        """
        # Nodes
        vertices_names = self.unkown_parameters + self.hidden_species + self.reaction_names + self.xi.columns.tolist()
        # self.xi.columns.tolist is the name of the reactions in xi

        # Edges
        edges = []

        # connect the species and parameters that we want to group together
        for group in self.group_hidden_species_and_parameters:
            # connect the species in the group
            for i in range(len(group)):
                for j in range(i+1, len(group)):
                    if group[i] in vertices_names and group[j] in vertices_names:
                        edges.append((vertices_names.index(group[i]), vertices_names.index(group[j])))

        # Edges from reactions to parameters or species
        for reaction_j in self.reaction_names:
            node_index_for_reaction_j = vertices_names.index(reaction_j)
            # compute the unknown parameters and hidden speceis influencing the propensity of reaction j
            parameters_species_affects_propensity_j = \
                set(self.propensities_dependencies[self.reaction_ordering[reaction_j]]).intersection(self.unkown_parameters + self.hidden_species)
            # compute the hidden speceis whose state is changed by reaction j
            hidden_species_affected_by_reaction_j = \
                [species for species in self.hidden_species if self.stoichiometric_matrix_df.loc[species, reaction_j] != 0]
            # combine the two sets above
            parameters_species_related_to_reaction_j = \
                list(parameters_species_affects_propensity_j.union(hidden_species_affected_by_reaction_j))
            # get the node indexes of all the species and parameters involved in reaction j
            node_index_for_relevant_parameters_species = \
                [vertices_names.index(PS) for PS in parameters_species_related_to_reaction_j]
            # add edges between reaction j and the selected species and parameters.
            edges.extend([node_index_for_reaction_j, PS] for PS in node_index_for_relevant_parameters_species)

        # Edges from the orignal reactions to the ones of xi
        for xi_reaction_k in self.xi.columns:
            node_index_for_O_xi_k = vertices_names.index(xi_reaction_k)
            Reactions_in_O_xi_k = self.O_xi[self.xi_ordering[xi_reaction_k]]
            node_index_for_relevant_reactions = [vertices_names.index(j) for j in Reactions_in_O_xi_k]
            edges.extend([node_index_for_O_xi_k, j] for j in node_index_for_relevant_reactions)

        graph_for_decomposing_hidden_system = ig.Graph(len(vertices_names),edges)
        graph_for_decomposing_hidden_system.vs['names'] = vertices_names

        # delete the conncected components with only no species and no parameters
        clustering = graph_for_decomposing_hidden_system.clusters()
        component_to_delete = []
        for i in range(len(clustering)):
            nodes_in_ith_clustering = [ graph_for_decomposing_hidden_system.vs['names'][index] for index in clustering[i]]
            if len(set(nodes_in_ith_clustering) & set(self.unkown_parameters + self.hidden_species)) == 0:
                component_to_delete.extend(clustering[i])

        graph_for_decomposing_hidden_system.delete_vertices(component_to_delete)

        return graph_for_decomposing_hidden_system


    def create_membership(self):
        membership = [i + 1 for i in self.connected_components_of_the_graph.membership]
        membership_df = pd.DataFrame(membership).transpose()
        membership_df.columns = self.graph_for_decomposing_hidden_system.vs['names']
        return membership_df


    def identify_subsystems(self):
        # # Each subsystem
        subsystems=[[]] # list of objects, also avoid index 0
        membership_transpose = self.membership.transpose()
        for i in np.arange(1, self.get_number_of_subsystem()+1, 1):
            members_in_subsystem_i = membership_transpose.index[membership_transpose[0] == i].tolist()
            parameters = list( set(members_in_subsystem_i).intersection(self.unkown_parameters))
            species = list( set(members_in_subsystem_i).intersection(self.hidden_species))
            xi_list = list( set(members_in_subsystem_i).intersection(self.xi.columns.tolist()))
            unobservable_reactions = list( set(members_in_subsystem_i).intersection(self.unobservable_reaction))
            observable_reactions = []
            for xi in xi_list:
                observable_reactions.extend(self.O_xi[self.xi_ordering[xi]])
            # print(observable_reactions)
            subsystems.append( HiddenSubsystems(parameters,species,unobservable_reactions,xi_list, observable_reactions) )

        return subsystems


    def plot_decomposition(self, figsize = (10,10)):
        # copy the graph
        fig, ax = plt.subplots(figsize=figsize, dpi=80)
        g = self.graph_for_decomposing_hidden_system

        # decide the colors: parameter: green; species: blue; reactions: red; xi: yellow
        ColorDictionary = {parameter: 'green' for parameter in self.parameters_names}
        ColorDictionary.update({species: 'cyan' for species in self.species_names})
        ColorDictionary.update({reaction: 'red' for reaction in self.reaction_names})
        ColorDictionary.update({xi: 'yellow' for xi in self.get_xi_names()})
        NodeColors= [ColorDictionary[node] for node in self.graph_for_decomposing_hidden_system.vs['names']]

        # decide the shapes of the nodes: species and parameters: circle; reactions square; xi: hexagon
        ShapeDictionary = {parameter: 'circle' for parameter in self.parameters_names}
        ShapeDictionary.update({species: 'circle' for species in self.species_names})
        ShapeDictionary.update({reaction: 'rectangle' for reaction in self.reaction_names})
        ShapeDictionary.update({xi: 'triangle-down' for xi in self.get_xi_names()})
        NodeShapes = [ShapeDictionary[node] for node in self.graph_for_decomposing_hidden_system.vs['names']]

        # set the position of the nodes: 1st layer: parameters and species; 2nd layer: reactions; 3nd layer: xi
        YPositionSpecies = 3
        YPositionReactions =2
        YPositionXi = 1
        # set the position
        X_position = 0
        layout = g.layout().coords
        dx = 0.7 # the horizontal distance between two nodes
        for i in range(1, self.get_number_of_subsystem()+1): # traverse all the connected component
            NodesInThisComponent = [self.graph_for_decomposing_hidden_system.vs['names'][count] \
                                     for count in self.graph_for_decomposing_hidden_system.components()[i-1]]
            # NodeParameters = set(NodesInThisComponent).intersection(self.parameters_names)
            # NodeSpecies = set(NodesInThisComponent).intersection(self.species_names)
            # NodeReactions = set(NodesInThisComponent).intersection(self.reaction_names)
            # NodeXi = set(NodesInThisComponent).intersection(self.get_xi_names())
            # NodeSpeciesParameters = NodeParameters.union(NodeSpecies)
            NodeSpeciesParameters = self.subsystems[i].parameters + self.subsystems[i].species
            NodeReactions = self.subsystems[i].unobservable_reactions + self.subsystems[i].observable_reactions
            NodeXi = self.subsystems[i].xi_list
            MaximumNumberOfNodesInEachLayer = max(len(NodeSpeciesParameters), len(NodeReactions), len(NodeXi))
            HorizontalLengthThisComponent = dx * (MaximumNumberOfNodesInEachLayer+1)
            # the position for parameters and species
            dX_spcies = HorizontalLengthThisComponent/(len(NodeSpeciesParameters)+1)
            X_species = X_position + dX_spcies
            for node in NodeSpeciesParameters:
                index_node = self.graph_for_decomposing_hidden_system.vs['names'].index(node)
                layout[index_node] = [X_species,YPositionSpecies]
                X_species = X_species+dX_spcies # renew the x poistion for the next node
            # the position for reactions
            dX_reactions = HorizontalLengthThisComponent/(len(NodeReactions)+1)
            X_reactions = X_position + dX_reactions
            for node in NodeReactions:
                index_node = self.graph_for_decomposing_hidden_system.vs['names'].index(node)
                layout[index_node] = [X_reactions,YPositionReactions]
                X_reactions = X_reactions + dX_reactions # renew the x poistion for the next node
            # the position for Xi
            dX_xi = HorizontalLengthThisComponent/(len(NodeXi)+1)
            X_xi = X_position + dX_xi
            for node in NodeXi:
                index_node = self.graph_for_decomposing_hidden_system.vs['names'].index(node)
                layout[index_node] = [X_xi,YPositionXi]
                X_xi = X_xi + dX_xi # renew the x poistion for the next node
            # update the X position for the next component
            X_position = X_position + HorizontalLengthThisComponent - dx

        # plot
        ig.plot(
            g,
            target = ax,
            layout=layout,
            vertex_size = 0.5,
            vertex_color = NodeColors,
            vertex_shape = NodeShapes,
            # vertex_frame_width=4.0,
            # vertex_frame_color="white",
            vertex_label = self.graph_for_decomposing_hidden_system.vs['names'],
            fontweight='bold',
            vertex_label_size = 10.0
            # edge_width=[2 if married else 1 for married in g.es["married"]],
            # edge_color=["#7142cf" if married else "#AAA" for married in g.es["married"]]
        )
        plt.show()
        print("Number of subsystems:", self.get_number_of_subsystem())
        print('-----------------------------------------------------')
        print(self.xi)
        print('-----------------------------------------------------')

        # print subsystem
        for i in range(1, self.get_number_of_subsystem()+1):
            print('Subsystem: ', i)
            print('Involved parameters and species:', self.subsystems[i].parameters + self.subsystems[i].species)
            print('Unobservable reactions involved:', self.subsystems[i].unobservable_reactions)
            print('Observable reactions involved:', self.subsystems[i].observable_reactions)
            print('Xi involved:', self.subsystems[i].xi_list)
            print('Size of state space of this subsystem:', self.get_size_of_subsystems()[i])
            print('-----------------------------------------------------')



    #######################################
    #                                   ###
    #           get methods             ###
    #                                   ###
    #######################################

    def get_number_of_unknown_parameters(self):
        return len(self.unkown_parameters)

    def get_number_of_subsystem(self):
        if self.connected_components_of_the_graph.membership == []:
            return 0
        else:
            return max(self.connected_components_of_the_graph.membership) + 1

    def get_number_of_xi(self):
        return self.xi.columns.size

    def get_xi_names(self):
        return self.xi.columns.tolist()

    def get_number_of_observable_species(self):
        return len(self.observable_species)

    def get_number_of_hidden_species(self):
        return len(self.hidden_species)

    def get_size_of_space_for_each_species(self):
        df_size_species_space = self.range_of_species['max'].sub(self.range_of_species['min'])
        df_size_species_space = df_size_species_space.add(1)
        return df_size_species_space.to_frame()

    def get_observable_reactions_compatible_with_xi(self, xi):
        return self.O_xi[self.xi_ordering[xi]]


    def get_size_of_subsystems(self):
        size = [0] # list of objects, also avoid index 0
        # compute the size of the species space for each species
        df_size_species_space = self.get_size_of_space_for_each_species()

        # compute the size for each subsystem
        for i in range(1,self.get_number_of_subsystem()+1):
            # the size of parameter space
            df = self.discretization_size_parameters.loc[self.subsystems[i].parameters] # the parameters relivant to this subystem
            if df.empty:
                size_parameter_space = 1
            else:
                size_parameter_space = df[0].prod()
            # the size of species space
            df = df_size_species_space.loc[self.subsystems[i].species]  # the species relivant to this subystem
            if df.empty:
                size_species_space = 1
            else:
                size_species_space = df[0].prod()
            # record the size
            size.append(size_species_space*size_parameter_space)
        return size


    def get_size_of_the_largest_subsystem(self):
        return max(self.get_size_of_subsystems())

    def get_reaction_vector(self,reaction_name):
        return self.stoichiometric_matrix_df[reaction_name]

    def get_reaction_vector_for_specific_species(self,reaction_name,species_names):
        return self.get_reaction_vector(reaction_name).loc[species_names]

    def check_state_in_the_range(self, state, ordering):
        for name, count in ordering.items():
            if name in self.parameters_names:
                if state[count] > self.range_of_parameters.loc[name,'max'] or state[count] < self.range_of_parameters.loc[name,'min']:
                    return False
            else:
                if state[count] > self.range_of_species.loc[name,'max'] or state[count] < self.range_of_species.loc[name,'min']:
                    return False
        return True




    #######################################
    #                                   ###
    #  methods for preparing filtering  ###
    #                                   ###
    #######################################

    def set_up_preparations_for_filtering(self):
        for i in range(1, self.get_number_of_subsystem()+1):
            # set up states
            parameter_species_ordering, states = self.set_the_states_for_each_subsystems(i)
            self.subsystems[i].set_states(parameter_species_ordering=parameter_species_ordering, states=states)

            # set up the structure of matrix A_evolution
            self.subsystems[i].set_A_evolution_structure(self.set_the_structure_of_matirx_A_evolution(i))

            # set up the structure of matrix A_jump
            A_jump_structure = []
            for xi in self.subsystems[i].xi_list:
                A_jump_structure.append( self.set_the_strcutre_of_matrix_A_jump(i,xi))
            A_jump_xi_ordering = { self.subsystems[i].xi_list[j]:j for j in range(len(self.subsystems[i].xi_list)) }
            self.subsystems[i].set_A_jump_structure(A_jump_structure, A_jump_xi_ordering)


        # record the preparation
        self.prepared_for_filtering = True

        return


    def set_the_states_for_each_subsystems(self, index_subsystem):
        if self.check_whether_index_of_subsystem_is_legitimate(index_subsystem) == False:
            return

        # construct the coordinates
        coords_parameters \
            = [np.linspace(self.range_of_parameters.loc[parameters, 'min'], self.range_of_parameters.loc[parameters, 'max'],
                           self.discretization_size_parameters.loc[parameters, 0]) \
               for parameters in self.subsystems[index_subsystem].parameters]
        coords_species = [
            np.arange(self.range_of_species.loc[species, 'min'], self.range_of_species.loc[species, 'max'] + 1) \
            for species in self.subsystems[index_subsystem].species]
        coords = coords_parameters + coords_species

        # construct the state
        meshes = np.meshgrid(*coords, indexing='ij')
        matrix = np.stack(meshes, axis=-1)
        states = matrix.reshape(-1, matrix.shape[-1])  # each row is a state of hidden species and species

        # the ordering
        names = self.subsystems[index_subsystem].parameters + self.subsystems[index_subsystem].species
        parameter_species_ordering = { names[i] :i for i in range(len(names)) }

        return parameter_species_ordering, states


    def set_the_structure_of_matirx_A_evolution(self, index_subsystem):
        # return row_index, column_index, reaction_list, state_column, sign
        state_column = [] # the state correpsonds to the column index
        state_row = [] # the state corresponds to the row index
        reaction_list = [] # the reactions correspond to the element in the matrix
        sign = [] # the sign of the element
        matrix_dimension = len(self.subsystems[index_subsystem].states)

        # unobservable reactions
        for reaction in self.subsystems[index_subsystem].unobservable_reactions:
            # construct the state change vector
            species_state_change = \
                self.get_reaction_vector_for_specific_species(reaction, self.subsystems[index_subsystem].species)
            state_change = [0] * len(self.subsystems[index_subsystem].parameter_species_ordering)
            for name, count in self.subsystems[index_subsystem].parameter_species_ordering.items():
                try:
                    state_change[count] = species_state_change.loc[name]
                except KeyError:
                    state_change[count] = 0.

            for state_c in self.subsystems[index_subsystem].states:
                state_r = state_c + np.array(state_change)
                # egligible state or note
                if self.check_state_in_the_range(state_r, self.subsystems[index_subsystem].parameter_species_ordering):
                    # off-diagonal element
                    state_column.append(state_c)
                    state_row.append(state_r)
                    reaction_list.append(reaction)
                    sign.append(1)

                    # diagonal element
                    state_column.append(state_c)
                    state_row.append(state_c)
                    reaction_list.append(reaction)
                    sign.append(-1)

        # observable reactions
        for reaction in self.subsystems[index_subsystem].observable_reactions:
            for state_c in self.subsystems[index_subsystem].states:
                state_column.append(state_c)
                state_row.append(state_c)
                reaction_list.append(reaction)
                sign.append(-1)

        # index of columns and rows
        if len(state_row) > 0: # if the matrix is not empty
            state_row = np.array(state_row)
            state_column = np.array(state_column)
            row_index = self.subsystems[index_subsystem].get_index_of_states(state_row)
            column_index = self.subsystems[index_subsystem].get_index_of_states(state_column)

            A = SparseMatrixStructure(row_index, column_index, reaction_list, state_column, sign, \
                                  self.subsystems[index_subsystem].parameter_species_ordering, \
                                  self.propensities, self.reaction_ordering, matrix_dimension)
        else:
            size = self.get_size_of_subsystems()[index_subsystem]
            A = scipy.sparse.coo_matrix(([0], ([0], [0])), shape=(size, size)).tocsr()

        return A #row_index, column_index, reaction_list, state_column, sign


    def set_the_strcutre_of_matrix_A_jump(self, index_subsystem, xi):
        # return row_index, column_index, reaction_list, state_column, sign
        state_column = [] # the state correpsonds to the column index
        state_row = [] # the state corresponds to the row index
        reaction_list = [] # the reactions correspond to the element in the matrix
        sign = [] # the sign of the element
        matrix_dimension = len(self.subsystems[index_subsystem].states)

        # transverse all the observable reactions compatible with xi
        for reaction in self.get_observable_reactions_compatible_with_xi(xi):
            # construct the state change vector
            species_state_change = \
                self.get_reaction_vector_for_specific_species(reaction, self.subsystems[index_subsystem].species)
            state_change = [0] * len(self.subsystems[index_subsystem].parameter_species_ordering)
            for name, count in self.subsystems[index_subsystem].parameter_species_ordering.items():
                try:
                    state_change[count] = species_state_change.loc[name]
                except KeyError:
                    state_change[count] = 0.
                    
            for state_c in self.subsystems[index_subsystem].states:
                state_r = np.array(state_c + state_change)
                # egligible state or note
                if self.check_state_in_the_range(state_r, self.subsystems[index_subsystem].parameter_species_ordering):
                    state_column.append(state_c)
                    state_row.append(state_r)
                    reaction_list.append(reaction)
                    sign.append(1)

        # index of columns and rows
        state_row = np.array(state_row)
        state_column = np.array(state_column)
        row_index = self.subsystems[index_subsystem].get_index_of_states(state_row)
        column_index = self.subsystems[index_subsystem].get_index_of_states(state_column)

        A = SparseMatrixStructure(row_index, column_index, reaction_list, state_column, sign, \
                                  self.subsystems[index_subsystem].parameter_species_ordering, \
                                  self.propensities, self.reaction_ordering, matrix_dimension)

        return A #row_index, column_index, reaction_list, state_column, sign

    def check_whether_index_of_subsystem_is_legitimate(self, index_subsystem):
        if index_subsystem < 1 or index_subsystem > self.get_number_of_subsystem():
            print('Error: the index of subsystem should be positive and less than the largest index of the subsystem')
            return False
        if isinstance(index_subsystem, int):
            return True
        else:
            print('Error: the index of subsystem should be an integer')
            return False

    #######################################
    #                                   ###
    #       Methods for filtering       ###
    #                                   ###
    #######################################

    # TODO: filtering algorithm returning the final distribution only

    # Y_trajectory is a list of list (a matrix), Time_Y is a list
    def filteringFFSP(self, Y_trajectory, Y_ordering, Time_Y, Initial_Distributions, tqdm_disable = False):
        """

        :param Y_trajectory: a list of lists (a matrix)
        :param Time_Y: a list to show the jumping time of observable species
        :param Initial_Distributions: A list of initial distributions for subsystems.
        :return: Return the result of continuous-time filtering
                a list of distributions
        """
        # Check if the list of states is generated or not

        # Check the legality
        if self.prepared_for_filtering == False:
            print("Error in filtering: please first run the method set_up_preparations_for_filtering")
            return "Error in filtering"
        if len(Y_trajectory[0]) != self.get_number_of_observable_species():
            print("Error in filtering: the dimension of Y_trajectory does not compatible with the number of observable species.")
            return "Error in filtering"
        if len(Y_trajectory) != len(Time_Y):
            print("Error in filtering: the length of Y_trajectory is not compatible with the length of Time_Y")
            return "Error in filtering"
        if len(Initial_Distributions) != self.get_number_of_subsystem() +1:
            print("Error in filtering: the length of Initial_Distribution does not fit the number of subsystems")
            return "Error in filtering"
        for i in range(1, self.get_number_of_subsystem()+1):
            if len(Initial_Distributions[i]) != len(self.subsystems[i].states):
                print("Error in filtering: the length of Initial_Distribution[" + str(i) + "] does not fit the size of subsystems")
                return "Error in filtering"
        if set(Y_ordering.keys()) != set(self.observable_species):
            print("Error in filtering: Y_ordering does not compatible with the observable species")
            return "Error in filtering"

        # initialize the output
        output = [[]]
        for i in range(1, self.get_number_of_subsystem()+1):
            output.append(DistributionOfSubsystems(self.subsystems[i].states, self.subsystems[i].parameter_species_ordering))
            output[i].extend_distributions([Time_Y[0]], [Initial_Distributions[i]])

        # Solve the filtering equation iteratively
        for t_index in tqdm(range(len(Time_Y)-1), desc="Progress", unit=" Observable_Jumps", position=0, disable=tqdm_disable): # range(len(Time_Y)-1):
            for i in range(1, self.get_number_of_subsystem()+1):
                # Solve the filtering equation between jump times
                A = self.subsystems[i].A_evolution_structure.build_final_matrix(Y_trajectory[t_index], Y_ordering)
                if A.shape[0] >= 45:
                    time_list, unnormalized_distribution_list = \
                        MatrixExponentialKrylov.exp_AT_x(A, Time_Y[t_index], Time_Y[t_index+1], output[i].distribution_list[-1])
                else:
                    time_list = [Time_Y[t_index+1]]
                    A = A.toarray()
                    unnormalized_distribution = scipy.linalg.expm( A*(Time_Y[t_index+1] - Time_Y[t_index]) ).dot(output[i].distribution_list[-1])
                    unnormalized_distribution_list = [unnormalized_distribution]
                time_list[-1] = time_list[-1] - 10**(-8) # time immediately before the jump
                normalized_distribution_list = self.normalize_distribution_list(unnormalized_distribution_list)
                output[i].extend_distributions(time_list, normalized_distribution_list)

                # Solve the conditional distribution at the jump time
                dY = Y_trajectory[t_index+1] - Y_trajectory[t_index]
                corresponding_xi = self.check_compatible_xi(dY, Y_ordering) # check which xi is compatible
                if corresponding_xi in self.subsystems[i].xi_list:
                    index_xi = self.subsystems[i].A_jump_xi_ordering[corresponding_xi]
                    A = self.subsystems[i].A_jump_structure[index_xi].build_final_matrix(Y_trajectory[t_index], Y_ordering)
                    unnormalized_distribution = A.dot(normalized_distribution_list[-1])
                    normalized_distribution = self.normalize_distribution(unnormalized_distribution)
                    output[i].extend_distributions([Time_Y[t_index + 1]], [normalized_distribution])
                else:
                    output[i].extend_distributions([Time_Y[t_index+1]], [normalized_distribution_list[-1]])


        return output

    def filteringFFSP_return_final_distribution(self, Y_trajectory, Y_ordering, Time_Y, Initial_Distributions, tqdm_disable=False):
        output = self.filteringFFSP(Y_trajectory, Y_ordering, Time_Y, Initial_Distributions, tqdm_disable)

        for i in range(1, self.get_number_of_subsystem() + 1):
            output[i].distribution_list = [output[i].distribution_list[-1]]
            output[i].time_list = [output[i].time_list[-1]]

        return output

    def normalize_distribution_list(self, distribution_list):
        for i in range(len(distribution_list)):
            non_negative_array = np.clip(distribution_list[i], a_min=0, a_max=None)
            distribution_list[i] = non_negative_array / np.sum(non_negative_array)

        return distribution_list

    def normalize_distribution(self, distribution):
        non_negative_array = np.clip(distribution, a_min=0, a_max=None)
        return non_negative_array / np.sum(non_negative_array)

    def check_compatible_xi(self, dY, Y_ordering):
        Y_ordering = sorted(Y_ordering.items(), key=lambda x: x[1])
        Y_ordering = dict(Y_ordering)
        for xi in self.get_xi_names():
            xi_vector = np.array([self.xi.at[species, xi] for species in Y_ordering])
            if (xi_vector == dY).all():
                return xi

        return None

    # plot the distribution of parameters
    def plot_unknown_parameter_distribution_final_time(self, distribution_list_from_filtering, parameters_real_value = None):
        """

        :param distribution_list_from_filtering: the result from the method filtering
        :param parameters_real_value: the real value of parameters, a dictionary
        :return:
        """

        rows = math.ceil(self.get_number_of_unknown_parameters() / 2)
        columns = 2

        fig, axs = plt.subplots(math.ceil(self.get_number_of_unknown_parameters() / 2), 2, figsize=(columns*3, rows*3))
        fig.subplots_adjust(wspace=0.5, hspace=1)

        for i, ax in enumerate(axs.flatten()):
            if i >= self.get_number_of_unknown_parameters():
                break
            parameter = self.unkown_parameters[i]
            # grab the information from the result
            subsystem_index = self.membership.at[0, parameter]
            distribution = distribution_list_from_filtering[subsystem_index].distribution_list[-1]
            parameter_species_ordering = distribution_list_from_filtering[subsystem_index].parameter_species_ordering
            states = distribution_list_from_filtering[subsystem_index].states[:, parameter_species_ordering[parameter]]

            # extract the information
            x_value = np.unique(states)
            # update y_value
            y_value = []
            for j in range(len(x_value)):
                indices = np.where(states == x_value[j])
                y_value.append( np.sum(distribution[indices]) )

            #plot
            row_index =  i // 2
            column_inex = i % 2
            ax.bar(x_value, y_value, width=x_value[1]-x_value[0])
            ax.set_xlabel(parameter)
            ax.set_ylabel('Probability')
            if parameters_real_value != None:
                real_value = parameters_real_value[parameter]
                ax.axvline(x=real_value, color='r', linestyle='--', label='Real Value')
                # axs[i].legend()

        plt.show()



    def plot_hidden_species_distribution_final_time(self, distribution_list_from_filtering, species_real_value = None):
        """

        :param distribution_list_from_filtering: the result from the method filtering
        :param species_real_value: the real value of species, a dictionary
        :return:
        """
        fig, axs = plt.subplots(math.ceil(self.get_number_of_hidden_species() / 2), 2, figsize=(5, 3))
        fig.subplots_adjust(wspace=0.5, hspace=1)

        for i, ax in enumerate(axs.flatten()):
            if i >= self.get_number_of_hidden_species():
                break
            species = self.hidden_species[i]
            # grab the information from the result
            subsystem_index = self.membership.at[0, species]
            distribution = distribution_list_from_filtering[subsystem_index].distribution_list[-1]
            parameter_species_ordering = distribution_list_from_filtering[subsystem_index].parameter_species_ordering
            states = distribution_list_from_filtering[subsystem_index].states[:, parameter_species_ordering[species]]

            # extract the information
            x_value = np.unique(states)
            # update y_value
            y_value = []
            for j in range(len(x_value)):
                indices = np.where(states == x_value[j])
                y_value.append( np.sum(distribution[indices]) )

            #plot
            row_index =  i // 2
            column_inex = i % 2
            ax.bar(x_value, y_value, width=x_value[1]-x_value[0])
            ax.set_xlabel(species)
            ax.set_ylabel('Probability')
            if species_real_value != None:
                real_value = species_real_value[species]
                ax.axvline(x=real_value, color='r', linestyle='--', label='Real Value')
                # axs[i].legend()

        plt.show()






