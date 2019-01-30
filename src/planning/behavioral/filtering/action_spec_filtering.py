import traceback
from abc import ABCMeta, abstractmethod
from decision_making.src.global_constants import BP_JERK_S_JERK_D_TIME_WEIGHTS, FILTER_V_0_GRID, FILTER_A_0_GRID, \
    FILTER_V_T_GRID, FILTER_S_T_GRID
from decision_making.src.planning.behavioral.default_config import DEFAULT_STATIC_RECIPE_FILTERING
from decision_making.src.planning.utils.generalized_frenet_serret_frame import GeneralizedFrenetSerretFrame
from logging import Logger
from typing import List, Optional
import numpy as np

import six

import rte.python.profiler as prof
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
    def filter(self, recipe: ActionSpec, behavioral_state: BehavioralState) -> bool:
        pass

    def __str__(self):
        return self.__class__.__name__

    @staticmethod
    def check_velocity_limit_beyond_spec(action_spec: ActionSpec, frenet: GeneralizedFrenetSerretFrame,
                                         frenet_points_idxs: np.array, vel_limit_in_points: np.array,
                                         predicates: np.array):
        """
        Given action spec and velocity limits on a subset of Frenet points, check if it's possible to brake enough
        before arriving to these points. The ability to brake is verified using quintic 'limits' predicates.
        :param action_spec: action specification
        :param frenet: generalized Frenet Serret frame
        :param frenet_points_idxs: array of indices of the Frenet frame points, having limited velocity
        :param vel_limit_in_points: array of maximal velocities at frenet_points_idxs
        :param predicates: dictionary of quintic 'limits' predicates
        :return: True if the agent can brake before each given point to its limited velocity
        """
        if len(frenet_points_idxs) == 0:
            return True  # all points beyond the spec have velocity limit higher than spec.v, so no need to brake

        # test ability to brake after the the end of spec's trajectory, complying max_velocity at suspected points
        s_T = frenet.get_s_from_index_on_frame(frenet_points_idxs, delta_s=0)
        v_0 = np.full(frenet_points_idxs.shape, action_spec.v)
        a_0 = np.full(frenet_points_idxs.shape, 0.)
        s = s_T - action_spec.s
        v_T = vel_limit_in_points
        if 'FilterIfAggressive' in DEFAULT_STATIC_RECIPE_FILTERING._filters.__str__():
            most_aggressive_level = AggressivenessLevel.STANDARD
        else:
            most_aggressive_level = AggressivenessLevel.AGGRESSIVE
        wJ, _, wT = BP_JERK_S_JERK_D_TIME_WEIGHTS[most_aggressive_level.value]
        predicate = predicates[(ActionType.FOLLOW_VEHICLE.name.lower(), wT, wJ)]
        predicates_ret = (predicate[FILTER_V_0_GRID.get_indices(v_0), FILTER_A_0_GRID.get_indices(a_0),
                                    FILTER_S_T_GRID.get_indices(s), FILTER_V_T_GRID.get_indices(v_T)] <= 0).all()
        print('beyond_spec_vel_limit: spec.v=%.2f idx=%d s_T=%4.2f s=%4.2f v_T=%4.2f %d' %
              (action_spec.v, frenet_points_idxs[0], s_T[0], s[0], v_T[0], predicates_ret))
        return predicates_ret


class ActionSpecFiltering:
    """
    The gateway to execute filtering on one (or more) ActionSpec(s). From efficiency point of view, the filters
    should be sorted from the strongest (the one filtering the largest number of recipes) to the weakest.
    """
    def __init__(self, filters: Optional[List[ActionSpecFilter]], logger: Logger):
        self._filters: List[ActionSpecFilter] = filters or []
        self.logger = logger

    def filter_action_spec(self, action_spec: ActionSpec, behavioral_state: BehavioralState) -> bool:
        for action_spec_filter in self._filters:
            try:
                if not action_spec_filter.filter(action_spec, behavioral_state):
                    return False
            except Exception:
                self.logger.warning('Exception during filtering at %s: %s', self.__class__.__name__, traceback.format_exc())
                return False
        return True

    @prof.ProfileFunction()
    def filter_action_specs(self, action_specs: List[ActionSpec], behavioral_state: BehavioralState) -> List[bool]:
        return [self.filter_action_spec(action_spec, behavioral_state) for action_spec in action_specs]
