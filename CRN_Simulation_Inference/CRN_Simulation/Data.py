import numpy as np
import math

class Trajectory():
    """
    Class to store the trajectory of the Y process

    Fields:
        state_sequence (list): list of the states of the Y process
        time_sequence (list): list of the times at which Y changes
        firing_reactions (list): list of the reactions that fire at each jumpt time
        propensities (list): list of the propensities of the reactions that fire at each jump time
        propensity_mask (list): list of the propensities that are > 0 at each jump time
    """

    def __init__(self) -> None:
        """
        Initialize the trajectory
        """
        self.state_sequence = []   # len T
        self.time_sequence = []    # len T
        self.firing_reactions = [] # len T - 2
        self.propensities = []     # len T - 2
        self.propensity_mask = []  # len T - 2

    def project(self, indexes, reaction_indexes):
        """
        Project the trajectory onto a subset of the species and reactions

        Args:
            indexes (list): list of the indexes of the species to keep
            reaction_indexes (list): list of the indexes of the reactions to keep

        Returns:
            projected_trajectory (Trajectory): the projected trajectory

        Note: TODO
            We might want to make this more flexible to retain part of the lost information... (discussion about V and w jumps) 
        """
        projected_trajectory = Trajectory()
        #first
        projected_trajectory.add_time(self.time_sequence[0])
        projected_trajectory.add_state(self.state_sequence[0][indexes])
        #middle
        for i in range(len(self.state_sequence)):
            if np.any(projected_trajectory.state_sequence[-1] != self.state_sequence[i][indexes]):
                projected_trajectory.add_state(self.state_sequence[i][indexes])
                projected_trajectory.add_time(self.time_sequence[i])
        for i in range(len(self.firing_reactions)):
            if self.firing_reactions[i] in reaction_indexes:
                new_index = len([k for k in sorted(reaction_indexes) if k < self.firing_reactions[i]])
                projected_trajectory.add_firing_reaction(new_index)
                projected_trajectory.add_propensity(self.propensities[i][reaction_indexes])
        #last
        projected_trajectory.add_state(self.state_sequence[-1][indexes])
        projected_trajectory.add_time(self.time_sequence[-1])
        return projected_trajectory

    def add_state(self, state):
        self.state_sequence.append(state)

    def add_time(self, time):
        self.time_sequence.append(time)

    def add_firing_reaction(self, reaction):
        self.firing_reactions.append(reaction)

    def add_propensity(self, propensity):
        self.propensities.append(propensity)
        self.propensity_mask.append(propensity > 0) # the mask is used to simplify the computation of the loss
        

    def __iadd__(self, other):
        """
        Inplace addition of two trajectories (concatenation)
        """
        #self.state_sequence += other.state_sequence
        self.state_sequence.extend(other.state_sequence)
        self.time_sequence += other.time_sequence
        self.firing_reactions += other.firing_reactions
        self.propensities += other.propensities
        self.propensity_mask += other.propensity_mask
        return self

    def __add__(self, other):
        """ 
        Addition of two trajectories (concatenation)
        """
        #self.state_sequence += other.state_sequence
        self.state_sequence.extend(other.state_sequence)
        self.time_sequence += other.time_sequence
        self.firing_reactions += other.firing_reactions
        self.propensities += other.propensities
        self.propensity_mask += other.propensity_mask
        return self

    def __len__(self):
        return len(self.time_sequence)

    def access(self): # for backwards compatibility this should be modified when we refactor the code TODO
        return self.time_sequence, self.state_sequence, self.firing_reactions, self.propensities

    def to_tensor_dict(self):
        """
        transform the trajectory into a dictionary of tensors to be used by the neural network after padding
        """
        return {
            "Y_times" : torch.tensor(self.time_sequence).float(),
            "Y_states" : torch.tensor(np.array(self.state_sequence)).float(),
            "Y_firing_reactions" : torch.tensor(self.firing_reactions).int(),
            "Y_propensities" : torch.tensor(self.propensities).float(),
            "Y_propensity_masks" : torch.tensor(self.propensity_mask).bool()
        }

    def __str__(self):
        # print the size of everything
        s = ""
        s += "time_sequence: " + str(len(self.time_sequence)) + "\n"
        s += "state_sequence: " + str(len(self.state_sequence)) + "\n"
        s += "firing_reactions: " + str(len(self.firing_reactions)) + "\n"
        s += "propensities: " + str(len(self.propensities)) + "\n"
        s += "propensity_mask: " + str(len(self.propensity_mask)) + "\n"
        return s
    

class Vw_trajectory(Trajectory):
    """
    This class extend a Y trajectory to include the V and w jumps 

    Fields:
        w_minus (list): list of the w- jumps
        w_plus (list): list of the w+ jumps
        Y_V_converter (list): convert jump indexes of Y to the corresponding V jump indexes (speed up the computation of the loss)
        associated_Y_process (Y_process): the Y process associated to this Vw trajectory (just a pointer)
        aO (list): list of the indexes of the reactions that are not in the Vw process (speed up the computation of the loss)
    """

    def __init__(self, Y) -> None:
        super().__init__()
        self.w_minus = [] # len T 
        self.w_plus  = [] # len T 
        self.Y_V_converter = [] # len Y
        self.associated_Y_process = Y # pointer
        self.aO = [] # len T TODO (this is just a quick patch that needs to be replaced ...)

    def set_associated_Y_process(self, Y_process):
        self.associated_Y_process = Y_process
    
    def add_w_minus(self, w_minus):
        self.w_minus.append(w_minus)

    def add_w_plus(self, w_plus):
        self.w_plus.append(w_plus)

    def add_Ycheckpoint(self, Y_V_converter):
        self.Y_V_converter.append(Y_V_converter)

    def Y2V(self, index_on_y_times):
        return self.Y_V_converter[index_on_y_times]
    
    def add_aO(self, aO): # TODO this is just a quick patch that needs to be replaced ...
        self.aO.append(aO)

    def plot(self):
        """
        Make a plot of the trajectories of V, Y, and w 

        see the quickRun notebook form an example
        """
        fig, axs = plt.subplots(3)
        # plot the dynamics of the processes
        fig.suptitle('Trajectories of the stochastic processes')
        times = [self.associated_Y_process["time_list"][0]]
        values = [self.associated_Y_process["state_list"][0]]
        for  i in range(1, len(self.associated_Y_process["time_list"])):
            times.extend([self.associated_Y_process["time_list"][i], self.associated_Y_process["time_list"][i]])
            values.extend([self.associated_Y_process["state_list"][i-1], self.associated_Y_process["state_list"][i]])
        axs[0].plot(times, values, label='Y', color='red')
        
        times = [self.time_sequence[0]]
        values = [self.state_sequence[0]]
        for  i in range(1, len(self.time_sequence)):
            times.extend([self.time_sequence[i], self.time_sequence[i]])
            values.extend([self.state_sequence[i-1], self.state_sequence[i]])
        axs[1].plot(times, values, label='V', color='green')

        values = []
        times = []
        for i in range(len(self.time_sequence)):
            times.extend([self.time_sequence[i], self.time_sequence[i]])
            values.extend([math.log(self.w_minus[i]), math.log(self.w_plus[i])])
        axs[2].plot(times, values, label='w', color='blue') 
        fig.legend()
        fig.show()

    def to_tensor_dict(self):
        """
        convert to dictionary of tensors to be used by the neural network after padding 
        """
        return {
            "V_times" : torch.tensor(self.time_sequence).float(),
            "V_states" : torch.tensor(np.array(self.state_sequence)).float(), # float even if it is int
            "V_firing_reactions" : torch.tensor(self.firing_reactions).int(),
            "V_propensities" : torch.tensor(self.propensities).float().float(),
            "V_propensity_masks" : torch.tensor(self.propensity_mask).bool(),
            "w_minus" : torch.tensor(self.w_minus).float(),
            "w_plus" : torch.tensor(self.w_plus).float(),
            "Y_V_converter" : torch.tensor(self.Y_V_converter).int(),
            "aO" : torch.tensor(self.aO).float()
        }

    
import torch

def pad(tensor_list, key, padding_value): # it modifies the tensor_list
    """
    pad a list of tensors to the same shape

    Args:
        tensor_list (list): list of tensors
        key (str): key of the field to pad
        padding_value (float): value to use for padding

    Returns:
        list: list of the padded tensors

    note that this procedure also manages empty tensors
    """

    shapes = [tensor_list[i][key].shape for i in range(len(tensor_list))]
    # max shape length of the tensor
    max_shape_length = max([len(shape) for shape in shapes])
    # adapt each shape to the max shape length
    for i, s in enumerate(shapes):
        shapes[i] = shapes[i] + (0,) * (max_shape_length - len(shapes[i]))
    # compute largest shape for each dimension
    max_shape = [max([s[i] for s in shapes]) for i in range(max_shape_length)]
    # pad each tensor

    shape_changes = []
    for i in range(len(tensor_list)):
        shape_changes.append([])
        for j in range(len(max_shape)-1,-1,-1):
            shape_changes[-1].append(0)
            shape_changes[-1].append(max_shape[j] - shapes[i][j])

    for i in range(len(tensor_list)):
        if 0 in shapes[i]:
            tensor_list[i][key] = torch.ones(max_shape) * padding_value
        else:
            tensor_list[i][key] = torch.nn.functional.pad(tensor_list[i][key], shape_changes[i], value=padding_value)


def pack(tensor_dict_list):
    """
    pack a list of tensor dictionaries into a single dictionary of tensors (and pad them to the same shape)
    This step is necessary to use the batches with the neural network.

    Args:
        tensor_dict_list (list): list of tensor dictionaries

    Returns:
        dict: dictionary of tensors
    """
    keys = tensor_dict_list[0].keys()

    for key in keys:
        pad(tensor_dict_list, key, -1)

    out = {}
    for key in keys:
        out[key] = torch.stack([tensor_dict_list[i][key] for i in range(len(tensor_dict_list))])

    return out

def turn_dict_insideout(tensor_dict):
    """
    turn a dictionary of tensors into a list of dictionaries of tensors (using the first dimension)

    Args:
        tensor_dict (dict): dictionary of tensors

    Returns:
        list: list of dictionaries of tensors
    """
    out = [{} for i in range(len(tensor_dict[list(tensor_dict.keys())[0]]))]
    keys = tensor_dict.keys()
    for k,v in tensor_dict.items():
        for i in range(len(out)):
            out[i][k] = v[i]
    return out

# TODO this function is not used as currently there is no support for masked tensors for the operation that we need 

# from torch import masked # not written in the docs, but it exists (we use then masked.MaskedTensor)

# def mask(tensor_dict):
#     keys = tensor_dict.keys()
#     for key in keys:
#         mask = tensor_dict[key] > -1 + 0.0001
#         tensor_dict[key] = masked.MaskedTensor(tensor_dict[key], mask)
#     return tensor_dict

