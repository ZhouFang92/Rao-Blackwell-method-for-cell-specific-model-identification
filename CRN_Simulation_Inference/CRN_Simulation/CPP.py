import numpy as np
import matplotlib.pyplot as plt

class CPP():
    """
    Class to represent a centered Poisson Process (cPP) for a CRN

    Fields:
        time_list (list): list of times at which the cPP is updated
        centered_PP (list): list of the cPP values
        centered_PP_actual (list): list of the actual cPP values
        centered_PP_mean (list): list of the mean cPP values
        current_a (list): list of the current propensities (the one used before the jump event!)
    """

    def __init__(self, a, number_of_reactions):
        self.time_list = [0.]
        # the cPP will be represented as 
        # cPP_k = cPP_k_actual - cPP_k_mean
        # all to zeros
        self.centered_PP = [np.zeros(number_of_reactions)]
        self.centered_PP_actual = [np.zeros(number_of_reactions)]
        self.centered_PP_mean = [np.zeros(number_of_reactions)]
        self.current_a = [a]

    def append(self, a, index, tau):
        """
        add an additional event to the cPP

        Args:
            a (np.array): list of the current propensities 
            index (None | int): firing reaction at the event, None if this is just a middle point (no firing)
            tau (float): elapsed time from last fired reaction
        """
        self.time_list.append(self.time_list[-1] + tau)
        # update all the means by a*tau
        self.centered_PP_mean.append(self.centered_PP_mean[-1] + a*tau)
        # update the actual cPP by adding the reaction (if any)
        self.centered_PP_actual.append(self.centered_PP_actual[-1].copy())
        if not index is None:
            self.centered_PP_actual[-1][index] += 1 
        # update the cPP by subtracting the mean
        self.centered_PP.append(self.centered_PP_actual[-1] - self.centered_PP_mean[-1])
        self.current_a.append(a)

    def at(self, time):
        """
        return the cPP at a specific time

        Args:
            time (float): time at which to return the cPP

        O(log(n)) complexity
        """
        left_index = np.searchsorted(np.array(self.time_list), time, side='right') - 1
        actual = self.centered_PP_actual[left_index] 
        mean = self.centered_PP_mean[left_index] + self.current_a[left_index] * (time - self.time_list[left_index])
        return actual - mean

    def sample_at_times(self, times):
        """
        return the cPP at a list of times

        Args:
            times (np.array): list of times at which to return the cPP (assumed sorted!)

        It is more efficient than calling 'at' multiple times as the complexity is O(n) instead of O(nlog(n))
        """
        times_index = 0
        cPP_index = 0
        cPP = []

        # note: the stored a tells us the propensity at the time BEFORE the event
        # we need to use the one of the following interval

        while times_index < len(times):
            # we reached the end of the cPP
            if cPP_index == len(self.time_list): 
                left_index = len(self.time_list) - 1
                actual = self.centered_PP_actual[left_index] 
                mean = self.centered_PP_mean[left_index] + self.current_a[min(len(self.time_list) - 1, left_index + 1)] * (times[times_index] - self.time_list[left_index])
                cPP.append(actual - mean)
                times_index += 1
            # we need to advance in the cPP indexes
            elif self.time_list[cPP_index] <= times[times_index]: 
                cPP_index += 1
            # we need to reconstruct the exact cPP value
            else: 
                left_index = max(0, cPP_index - 1)
                actual = self.centered_PP_actual[left_index] 
                mean = self.centered_PP_mean[left_index] + self.current_a[min(len(self.time_list) - 1, left_index + 1)] * (times[times_index] - self.time_list[left_index])
                cPP.append(actual - mean)
                times_index += 1
        return np.array(cPP)

        
    def _add_pre_times(self, times, eps=1e-10):
        """
        add a small epsilon to the times to locate the discontinuities for plotting

        Args:
            times (np.array): list of times
            eps (float): epsilon to add

        Returns:
            np.array: list of times with eps added
        """
        new_times = []
        for t in times:
            new_times.append(max(0, t - eps))
            new_times.append(max(0, t + eps))
        return new_times

    # plot the cPP
    def plot(self, times=None):
        """
        plot the cPP

        Args:
            times (None | np.array): list of times at which to plot the cPP, if None, the times of the cPP will be used

        NOTE: 
            to better visualize the cPP, we add a small epsilon to the times to locate the discontinuities.
            This is done only for the plot with times=None
        """
        if times is None:
            time_list = self._add_pre_times(self.time_list)
        else:
            time_list = times
        cPP = self.sample_at_times(time_list)
        for i in range(cPP.shape[1]):
            plt.plot(time_list, cPP[:,i])
            plt.xlabel('Time')
            plt.ylabel('cPP value')
            plt.title('Centered Poisson Process')


# import torch

# def CRN_simulations_to_dataloaders(ts,Xs,Ys,cPPs, batch_size, test_split=0.2, shuffle_dataset=True):

#     # torchify
#     ts = [torch.tensor(t) for t in ts]
#     Xs = [torch.tensor(X) for X in Xs]
#     Ys = [torch.tensor(Y) for Y in Ys]
#     cPPs = [torch.tensor(cPP) for cPP in cPPs]

#     tensor_dict = {}
#     tensor_dict["t"] = torch.stack(ts)
#     tensor_dict["X"] = torch.stack(Xs)
#     tensor_dict["Y"] = torch.stack(Ys)
#     tensor_dict["cPPs"] = torch.stack(cPPs)

#     dataset = torch.utils.data.TensorDataset(tensor_dict["t"], tensor_dict["X"], tensor_dict["Y"], tensor_dict["cPPs"])
#     train_set, test_set = torch.utils.data.random_split(dataset, [int(len(dataset)*(1-test_split)), int(len(dataset)*test_split)])

#     train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=shuffle_dataset)
#     test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)

#     return train_loader, test_loader






