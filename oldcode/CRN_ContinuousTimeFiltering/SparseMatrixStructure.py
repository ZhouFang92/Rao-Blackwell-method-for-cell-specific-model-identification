import numpy as np
import inspect
from scipy import sparse

# for debug only
import time


class SparseMatrixStructure:

    def __init__(self, row_index, column_index, reaction_list, state_column, sign, parameter_species_ordering, \
                 propensities, reaction_ordering, matrix_dimension):
        """

        :param row_index: a list of row indices
        :param column_index: a list column indices
        :param reaction_list: a list of reactions for elements
        :param state_column: a list of states from which the probability outflows
        :param sign: the sign of elements
        :parameter_species_ordering: the order of the species and parameter in the states
        :return:
        """
        self.row_index = row_index
        self.column_index = column_index
        self.reaction_list = reaction_list
        self.state_column = state_column
        self.sign = sign

        self.parameter_species_ordering = parameter_species_ordering # {'k1': 0, 'k3': 1, 'k2': 2, 'mRNA': 3}
        self.propensities = propensities
        self.reaction_ordering = reaction_ordering

        self.matrix_dimension = matrix_dimension

        # # for method 1
        # self.vectorized_function_to_compute_values = np.vectorize(self.value_in_matrix)
        # self.zero_to_maximum_column_index = np.arange(len(self.column_index))

        # for method 2 & 3
        reactions_involved = list(set(self.reaction_list))
        self.reaction_involved = { a : count for a, count in zip(reactions_involved, range(len(reactions_involved))) }# unique elements n the reaction list
        self.check_reaction = []
        for a in self.reaction_involved:
            check_vector = [ name == a for name in self.reaction_list ]
            self.check_reaction.append(np.array(check_vector))

        # for method 3
        self.propensities_dependencies = [inspect.getfullargspec(p).args for p in self.propensities]
        # observation irrelevant part
        self.row_index_observ_irrel, self.column_index_observ_irrel, self.reaction_list_observ_irrel, \
            self.state_column_observ_irrel, self.sign_observ_irrel = self.get_observation_irrelevant_elements()
        self.check_reaction_observ_irrel = []
        for a in self.reaction_involved:
            check_vector_observ_irrel = [ name == a for name in self.reaction_list_observ_irrel ]
            self.check_reaction_observ_irrel.append(np.array(check_vector_observ_irrel))
        # observation relevant part
        self.row_index_observ_rel, self.column_index_observ_rel, self.reaction_list_observ_rel, \
            self.state_column_observ_rel, self.sign_observ_rel = self.get_observation_relevant_elements()
        self.check_reaction_observ_rel = []
        for a in self.reaction_involved:
            check_vector_observ_rel = [ name == a for name in self.reaction_list_observ_rel ]
            self.check_reaction_observ_rel.append(np.array(check_vector_observ_rel))
        # construct the part of matrix irrelevant to observations
        self.matrix_observation_irrel = self.build_sparse_matrix_general(
            self.row_index_observ_irrel, self.column_index_observ_irrel, self.reaction_list_observ_irrel,
            self.state_column_observ_irrel, self.sign_observ_irrel, self.check_reaction_observ_irrel, [0], {'whatever': 0})


    def get_dimension(self):
        return  self.matrix_dimension #max(self.column_index) + 1

    # method 3 : split the matrix into an observation dependent part and an observation irrelevant part

    def get_observation_irrelevant_elements(self):
        row_index_observ_irrel = []
        column_index_observ_irrel = []
        reaction_list_observ_irrel = []
        state_column_observ_irrel = []
        sign_observ_irrel = []

        for i in np.arange(len(self.row_index)):
            index_of_the_considered_reaction = self.reaction_ordering[self.reaction_list[i]]
            # check if the reaction propensity depends on observable species
            if set(self.propensities_dependencies[index_of_the_considered_reaction]).issubset(self.parameter_species_ordering):
                row_index_observ_irrel.append(self.row_index[i])
                column_index_observ_irrel.append(self.column_index[i])
                reaction_list_observ_irrel.append(self.reaction_list[i])
                state_column_observ_irrel.append(self.state_column[i])
                sign_observ_irrel.append(self.sign[i])


        return np.array(row_index_observ_irrel), np.array(column_index_observ_irrel), reaction_list_observ_irrel, \
            np.array(state_column_observ_irrel), np.array(sign_observ_irrel)


    def get_observation_relevant_elements(self):
        row_index_observ_rel = []
        column_index_observ_rel = []
        reaction_list_observ_rel = []
        state_column_observ_rel = []
        sign_observ_rel = []

        for i in np.arange(len(self.row_index)):
            index_of_the_considered_reaction = self.reaction_ordering[self.reaction_list[i]]
            # check if the reaction propensity depends on observable species
            if set(self.propensities_dependencies[index_of_the_considered_reaction]).issubset(self.parameter_species_ordering):
                pass
            else:
                row_index_observ_rel.append(self.row_index[i])
                column_index_observ_rel.append(self.column_index[i])
                reaction_list_observ_rel.append(self.reaction_list[i])
                state_column_observ_rel.append(self.state_column[i])
                sign_observ_rel.append(self.sign[i])

        return np.array(row_index_observ_rel), np.array(column_index_observ_rel), np.array(reaction_list_observ_rel), \
            np.array(state_column_observ_rel), np.array(sign_observ_rel)


    def build_sparse_matrix_general(self, row_index, column_index, reaction_list, state_column, sign, check_reaction_list, Y, Y_ordering):
        if len(row_index) == 0: # zero matrix
            return sparse.coo_matrix(([0], ([0], [0])), shape=(self.get_dimension(), self.get_dimension())).tocsr()

        value = np.zeros( len(column_index) )
        Y_matrix = np.tile( np.array([Y]), (len(column_index),1) )
        reaction_involved_here = list(set(reaction_list))

        for reaction in reaction_involved_here:
            prop = self.propensities[ self.reaction_ordering[reaction] ]
            args = inspect.getfullargspec(prop).args
            input_args = {} # input args of this specific propensity
            for a in args:
                if a in self.parameter_species_ordering:
                    input_args[a] = state_column[:, self.parameter_species_ordering[a]]
                else:
                    input_args[a] = Y_matrix[:, Y_ordering[a]]

            check_reaction = check_reaction_list[self.reaction_involved[reaction]]
            # compute the value
            add = prop(**input_args)
            add = add * check_reaction * sign
            value = value + add

        A = sparse.coo_matrix( (value, (row_index, column_index)), \
                                shape=(self.get_dimension(),self.get_dimension())).tocsr()

        return A


    def build_final_matrix(self, Y, Y_ordering):

        # construct the part of the matrix relevant to observations
        if len(self.row_index_observ_rel) == 0:
            return self.matrix_observation_irrel

        A = self.build_sparse_matrix_general(
            row_index=self.row_index_observ_rel,
            column_index=self.column_index_observ_rel,
            reaction_list=self.reaction_list_observ_rel,
            state_column=self.state_column_observ_rel,
            sign=self.sign_observ_rel,
            check_reaction_list=self.check_reaction_observ_rel,
            Y=Y,
            Y_ordering=Y_ordering
        )

        return A + self.matrix_observation_irrel

    # old methods

    def set_Y_and_its_ordering(self, Y, Y_ordering):
        self.Y = Y
        self.Y_ordering = Y_ordering


    def build_sparse_matrix(self, Y, Y_ordering):
        """

        :param Y: the value of observation
        :param Y_ordering: a dictionary shows the order of Y
        :param propensity: a list of lambda function
        :param reaction_ordering: the ordering of the reactions
        :return:
        """

        # method 1
        # set the observation
        # self.set_Y_and_its_ordering(Y, Y_ordering)
        # value = self.vectorized_function_to_compute_values(self.zero_to_maximum_column_index, self.reaction_list, self.sign)

        # method 2 to compute the values for each reations separately.
        value = np.zeros( len(self.column_index) )
        Y_matrix = np.tile( np.array([Y]), (len(self.column_index),1) )


        for reaction, count in zip(self.reaction_involved, range(len(self.reaction_involved))):
            prop = self.propensities[ self.reaction_ordering[reaction] ]
            args = inspect.getfullargspec(prop).args
            input_args = {} # input args of this specific propensity
            for a in args:
                if a in self.parameter_species_ordering:
                    input_args[a] = self.state_column[:, self.parameter_species_ordering[a]]
                else:
                    input_args[a] = Y_matrix[:, Y_ordering[a]]

            check_reaction = self.check_reaction[count]
            # compute the value
            add = prop(**input_args)
            add = add * check_reaction * self.sign
            value = value + add

        A = sparse.coo_matrix( (value, (self.row_index, self.column_index)), \
                                shape=(self.get_dimension(),self.get_dimension())).tocsr()

        return A

    # give the value of each element
    def value_in_matrix(self, index_state_column, reaction, sign):
        state = self.state_column[index_state_column]
        prop = self.propensities[ self.reaction_ordering[reaction] ]
        args = inspect.getfullargspec(prop).args
        input_args = {}  # input args of this specific propensity
        for a in args:
            try: # arguments is a hidden variable
                input_args[a] = state[self.parameter_species_ordering[a]]
            except KeyError:
                try: # arguments is an observable vriable
                    input_args[a] = self.Y[self.Y_ordering[a]]
                except KeyError:
                    raise Exception(f'argument {a} is nor a parameter or a species')

        return sign * prop(**input_args)

