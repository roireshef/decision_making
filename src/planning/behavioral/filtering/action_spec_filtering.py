import rte.python.profiler as prof
from decision_making.src.global_constants import FILTER_V_0_GRID, FILTER_A_0_GRID, \
    FILTER_V_T_GRID, BP_JERK_S_JERK_D_TIME_WEIGHTS_FOLLOW_LANE
from decision_making.src.planning.utils.generalized_frenet_serret_frame import GeneralizedFrenetSerretFrame
from decision_making.src.state.state import State
from logging import Logger
from typing import List, Optional
import numpy as np

import six
from abc import ABCMeta, abstractmethod
from decision_making.src.planning.behavioral.behavioral_state import BehavioralState
from decision_making.src.planning.behavioral.data_objects import ActionSpec, AggressivenessLevel, ActionType


@six.add_metaclass(ABCMeta)
class ActionSpecFilter:
    """
    Base class for filter implementations that act on ActionSpec and returns a boolean value that corresponds to
    whether the ActionSpec satisfies the constraint in the filter. All filters have to get as input ActionSpec
    (or one of its children) and  BehavioralState (or one of its children) even if they don't actually use them.
    """
    @abstractmethod
    def filter(self, action_specs: List[ActionSpec], behavioral_state: BehavioralState, state: State) -> List[bool]:
        pass

    def __str__(self):
        return self.__class__.__name__

    @staticmethod
    def check_ability_to_brake_beyond_spec(action_spec: ActionSpec, frenet: GeneralizedFrenetSerretFrame,
                                           frenet_points_idxs: np.array, vel_limit_in_points: np.array,
                                           action_distances: np.array):
        """
        Given action spec and velocity limits on a subset of Frenet points, check if it's possible to brake enough
        before arriving to these points. The ability to brake is verified using static actions distances.
        :param action_spec: action specification
        :param frenet: generalized Frenet Serret frame
        :param frenet_points_idxs: array of indices of the Frenet frame points, having limited velocity
        :param vel_limit_in_points: array of maximal velocities at frenet_points_idxs
        :param action_distances: dictionary of distances of static actions
        :return: True if the agent can brake before each given point to its limited velocity
        """
        # create constraints for static actions per point beyond the given spec
        dist_to_points = frenet.get_s_from_index_on_frame(frenet_points_idxs, delta_s=0) - action_spec.s

        # retrieve distances of static actions for the most aggressive level, since they have the shortest distances
        wJ, _, wT = BP_JERK_S_JERK_D_TIME_WEIGHTS_FOLLOW_LANE[AggressivenessLevel.STANDARD.value]
        brake_dist = action_distances[(ActionType.FOLLOW_LANE.name.lower(), wT, wJ)][
            FILTER_V_0_GRID.get_index(action_spec.v), FILTER_A_0_GRID.get_index(0), :]
        return (brake_dist[FILTER_V_T_GRID.get_indices(vel_limit_in_points)] < dist_to_points).all()


class ActionSpecFiltering:
    """
    The gateway to execute filtering on one (or more) ActionSpec(s). From efficiency point of view, the filters
    should be sorted from the strongest (the one filtering the largest number of recipes) to the weakest.
    """
    def __init__(self, filters: Optional[List[ActionSpecFilter]], logger: Logger):
        self._filters: List[ActionSpecFilter] = filters or []
        self.logger = logger

    def filter_action_specs(self, action_specs: List[ActionSpec], behavioral_state: BehavioralState, state: State) -> List[bool]:
        """
        Filters a list of 'ActionSpec's based on the state of ego and nearby vehicles (BehavioralState).
        :param action_specs: A list of objects representing the specified actions to be considered
        :param behavioral_state: semantic behavioral state, containing the semantic grid
        :return: A boolean List , True where the respective action_spec is valid and false where it is filtered
        """
        mask = [True]*len(action_specs)
        for action_spec_filter in self._filters:
            mask = action_spec_filter.filter(action_specs, behavioral_state, state)
            action_specs = [action_specs[i] if mask[i] else None for i in range(len(action_specs))]
        return mask

    @prof.ProfileFunction()
    def filter_action_spec(self, action_spec: ActionSpec, behavioral_state: BehavioralState, state: State) -> bool:
        """
        Filters an 'ActionSpec's based on the state of ego and nearby vehicles (BehavioralState).
        :param action_spec: An object representing the specified actions to be considered
        :param behavioral_state: semantic behavioral state, containing the semantic grid
        :return: A boolean , True where the action_spec is valid and false where it is filtered
        """
        return self.filter_action_specs([action_spec], behavioral_state, state)[0]

