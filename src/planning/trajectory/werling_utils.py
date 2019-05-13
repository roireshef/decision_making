import numpy as np

from decision_making.src.planning.types import FrenetState1D, \
    FrenetTrajectory1D


class WerlingUtils:
    @staticmethod
    def repeat_1d_state(fstate: FrenetState1D, repeats: int,
                        override_values: FrenetState1D, override_mask: FrenetState1D):
        """please see documentation in used method"""
        return WerlingUtils.repeat_1d_states(fstate[np.newaxis, :], repeats, override_values, override_mask)[0]

    @staticmethod
    def repeat_1d_states(fstates: FrenetTrajectory1D, repeats: int,
                         override_values: FrenetState1D, override_mask: FrenetState1D):
        """
        Given an array of 1D-frenet-states [x, x_dot, x_dotdot], this function builds a block of replicates of each one
        of them <repeats> times while giving the option for overriding values (part or all) their values.
        :param fstates: the set of 1D-frenet-states to repeat
        :param repeats: length of repeated block (number of replicates for each state)
        :param override_values: 1D-frenet-state vector of values to override while repeating every state in <fstates>
        :param override_mask: mask vector for <override_values>. Where mask values == 1, override will apply, whereas
        mask value of 0 will incur no value override
        :return:
        """
        repeating_slice = np.logical_not(override_mask) * fstates + \
                          override_mask * np.repeat(override_values[np.newaxis, :], repeats=fstates.shape[0], axis=0)
        repeated_block = np.repeat(repeating_slice[:, np.newaxis, :], repeats=repeats, axis=1)
        return repeated_block
