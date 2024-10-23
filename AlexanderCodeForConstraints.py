import torch


# def generate_mass_conservation_constraints_strings(species_names, species_order, among_which_species, value_of_the_sum):

#     indexes = [species_order[name] for name in among_which_species]
#     indexes.sort()

#     constraints = []
#     for i in range(len(indexes)):
#         if i < len(indexes) - 1:
#             constraints.append(("lambda actions, options,i=i, indexes=indexes, value_of_the_sum=value_of_the_sum: torch.vmap(lambda actions, options: " + "+".join([f"actions[{i}]" for i in indexes[:i]]) + (" + " if i>0 else "") + "options " + f"<= {value_of_the_sum})(actions, options.repeat(actions.shape[0], 1))"))
#         else:
#             constraints.append(("lambda actions, options,i=i, indexes=indexes, value_of_the_sum=value_of_the_sum: torch.vmap(lambda actions, options: " + "+".join([f"actions[{i}]" for i in indexes[:i]]) + (" + " if i>0 else "") + "options " + f"== {value_of_the_sum})(actions, options.repeat(actions.shape[0], 1))"))


#     # merge constraints for all species

#     print(constraints)  

    
generate_mass_conservation_constraints_strings(scpecies_names, species_order, ["A", "C"], 1)


def alltrue_constraint():
    # shortcut to generate an always true constraint
    return lambda action, options: torch.vmap(lambda a, o: torch.ones(o.shape).to(o))(action, options.repeat(action.shape[0], 1)).bool()

def generate_mass_conservation_constraints(species_names, species_order, among_which_species, value_of_the_sum):
    """ 
    Generate constraints for mass conservation. 
    Note that we ensure that it is always possible to sample a valid option for the last species in the list.
    Combining constraints might invalidate that choice (thus becoming unfit for the autoregressive setting).
    You can always discard invalid sequences, but this will be inefficient. 
    No problem if the constraints involve separate species sets.
    To fix that it would be necessary to merge the constraints with a dependency graph.

    Args:
    species_names: list of species names
    species_order: dictionary of species orders (used for conversions)
    among_which_species: list of species names that need to be conserved
    value_of_the_sum: value of the sum of the species

    Returns:
    a list of callable functions with signature (actions, options) -> bool (tensor)
    """

    indexes = [species_order[name] for name in among_which_species]
    indexes.sort()

    # generate constraints for each species

    constraints = []
    for i in range(len(indexes)):
        if i < len(indexes) - 1:
            # we need to use eval here to generate the lambda functions
            # this allows to have efficient indexing while using the functions, as all the indexing will have been precomputed beforehand
            # vmap allows to efficiently vectorize the constraints to apply it to a tensor of actions
            # constraint of the type (a1 + a2 + ... + vector_of_options) <= value_of_the_sum
            constraints.append(eval("lambda actions, options, i=i, indexes=indexes, value_of_the_sum=value_of_the_sum: torch.vmap(lambda a, o: " + "+".join([f"a[{i}]" for i in indexes[:i]]) + (" + " if i>0 else "") + "o " + f"<= {value_of_the_sum})(actions, options.repeat(actions.shape[0], 1))"))
        else:
            # same as above, but with an equality for the last constrained species 
            # note that this is the case in which you would technically not need to sample the probability distribution.
            # Of course this might change if you combine multiple constraints together.
            # constraint of the type (a1 + a2 + ... + vector_of_options) == value_of_the_sum
            constraints.append(eval("lambda actions, options, i=i, indexes=indexes, value_of_the_sum=value_of_the_sum: torch.vmap(lambda a, o: " + "+".join([f"a[{i}]" for i in indexes[:i]]) + (" + " if i>0 else "") + "o " + f"== {value_of_the_sum})(actions, options.repeat(actions.shape[0], 1))"))

    # Create alltrue constraints for the species that are not constrained and fix ordering
    full_constraints = []
    index_on_indexes = 0
    for i in range(len(species_names)):
        if i in indexes:
            full_constraints.append(constraints[index_on_indexes])
            index_on_indexes += 1
        else:
            full_constraints.append(alltrue_constraint())

    return full_constraints

class Constraint(torch.nn.Module):

    # a constraint is defined by a pair of actions (the samples you have chosen so far) and options (possible next actions), and returns a boolean tensor 
    # of shape [len(actions), len(options)] that tells you if the constraint is satisfied for each pair of actions and options
    # By default, we assume that all constraints are satisfied

    def __init__(self, species_names, species_order):
        """ 
        Args:
        species_names: list of species names
        species_order: dictionary of species orders (used for conversions)
        """
        super(Constraint, self).__init__()
        self.species_names = species_names
        self.species_order = species_order
        # alltrue constraint
        self.constraints = [alltrue_constraint() for _ in species_names]

    def merge(self, other):
        """ 
        Out-of-place merging of two constraints

        Args:
        other: another constraint

        Returns:
        a new constraint that is the merge of the two constraints (aka logical and)

        # TODO we can easily implement other logical operations with constraints if needed
        """
        new_constraint = Constraint(self.species_names, self.species_order)
        for i in range(len(self.constraints)):
            new_constraint.constraints[i] = lambda actions, options, i=i, self=self, other=other: self.constraints[i](actions, options) & other.constraints[i](actions, options)

        return new_constraint
    
    def forward(self, species_index, actions, options):
        """
        Args:
            species_index: index of the species for which the constraint is applied
            actions: tensor of shape [batch_size, actions_taken_so_far] of actions
            options: tensor of shape [batch_size, available_options] of options

        Returns:    
            a tensor of shape [batch_size, available_options] of boolean values, indicating if the constraint is satisfied
        """ 
        return self.constraints[species_index](actions, options)
    
class MassConservationConstraint(Constraint):
    # TODO you can, if needed, implement constraints representing different operations for mass conservation,
    # like inequalities, or ratios and so on. 
    # just keep this in mind if needed.

    def __init__(self, species_names, species_order, among_which_species, value_of_the_sum):
        """ 
        Args:
        species_names: list of species names
        species_order: dictionary of species orders (used for conversions)
        among_which_species: list of species names that need to be conserved (as string list)
        value_of_the_sum: value of the sum for the species
        """
        super(MassConservationConstraint, self).__init__(species_names, species_order)
        self.among_which_species = among_which_species
        self.value_of_the_sum = value_of_the_sum
        self.constraints = generate_mass_conservation_constraints(species_names, species_order, among_which_species, value_of_the_sum)


# Example:
scpecies_names = ["A", "B", "C", "D"]
species_order = {name: i for i, name in enumerate(scpecies_names)}
# A + C = 2a
mccAC2 = MassConservationConstraint(scpecies_names, species_order, ["A", "C"], 2)

# now assume we have some choices for A and B and we want to find the possible choices for C
actions = torch.tensor(
    [[0,0], [1,0], [2,0], [3,0]] # A = 0 B = 0, A = 1 B = 0, A = 2 B = 0
)
# and the possible choices for C are 
options = torch.arange(0, 4)

# let's check the constraints for C
print("without B-C constraint:")
print("we expect A to have a single choice of C for each A<3")
print(mccAC2(mccAC2.species_order["C"], actions, options))

# now we can also bound the choices for C relative to B
# B + C = 1
mccBC1 = MassConservationConstraint(scpecies_names, species_order, ["B", "C"], 1)
# now we can merge the constraints
combined = mccAC2.merge(mccBC1)
# and check the constraints
print("with B-C constraint:")
print("we expect only a single valid choice for C = 1 when A = 1 and B = 1")
print(combined(combined.species_order["C"], actions, options))
