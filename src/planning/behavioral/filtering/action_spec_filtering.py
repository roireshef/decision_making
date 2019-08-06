from collections import defaultdict
from itertools import compress

import numpy as np
import six
from abc import ABCMeta, abstractmethod
from logging import Logger
from typing import List
from typing import Optional

from decision_making.src.planning.types import CRT_LEN, FS_2D_LEN, FS_SX, FS_SV, FS_SA
import rte.python.profiler as prof
from decision_making.src.global_constants import BP_ACTION_T_LIMITS, TRAJECTORY_TIME_RESOLUTION, \
    MINIMUM_REQUIRED_TRAJECTORY_TIME_HORIZON
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import ActionSpec, ActionType
from decision_making.src.planning.utils.kinematics_utils import KinematicUtils
from decision_making.src.planning.utils.optimal_control.poly1d import QuinticPoly1D


@six.add_metaclass(ABCMeta)
class ActionSpecFilter:
    """
    Base class for filter implementations that act on ActionSpec and returns a boolean value that corresponds to
    whether the ActionSpec satisfies the constraint in the filter. All filters have to get as input ActionSpec
    (or one of its children) and  BehavioralGridState (or one of its children) even if they don't actually use them.
    """
    @abstractmethod
    def filter(self, action_specs: List[ActionSpec], behavioral_state: BehavioralGridState) -> List[bool]:
        pass

    @staticmethod
    def _group_by_lane(action_specs: List[ActionSpec]):
        """
        takes a list of action specs and groups them (and their indices) into the 3 corresponding lanes
        :param action_specs: an ordered list of action specs
        :return: a tuple of two dictionaries: ({relative lane: action spec}, {relative lane: index})
        """
        # group all specs and their indices by the relative lanes
        specs_by_rel_lane = defaultdict(list)
        indices_by_rel_lane = defaultdict(list)
        for i, spec in enumerate(action_specs):
            if spec is not None:
                specs_by_rel_lane[spec.relative_lane].append(spec)
                indices_by_rel_lane[spec.relative_lane].append(i)

        return specs_by_rel_lane, indices_by_rel_lane

    @staticmethod
    def _build_trajectories(action_specs: List[ActionSpec], behavioral_state: BehavioralGridState) -> (np.ndarray, np.ndarray):
        """
        Builds a baseline trajectory out of the action specs (terminal states)
        :param action_specs: an ordered list of action specs
        :param behavioral_state:
        :return: A tuple of (cartesian_trajectories, lane_based_velocity_limits) the latter is all zero
        if build_lane_segment_velocities is False
        """
        # group all specs and their indices by the relative lanes
        specs_by_rel_lane, indices_by_rel_lane = ActionSpecFilter._group_by_lane(action_specs)

        time_samples = np.arange(0, BP_ACTION_T_LIMITS[1], TRAJECTORY_TIME_RESOLUTION)
        ctrajectories = np.empty((len(action_specs), len(time_samples), CRT_LEN), dtype=np.float)
        ftrajectories = np.empty((len(action_specs), len(time_samples), FS_2D_LEN), dtype=np.float)

        # loop on the target relative lanes and calculate lateral accelerations for all relevant specs
        for rel_lane, lane_specs in specs_by_rel_lane.items():
            specs_t = np.array([spec.t for spec in lane_specs])
            pad_mode = np.array([spec.only_padding_mode for spec in lane_specs])
            goal_fstates = np.array([spec.as_fstate() for spec in lane_specs])

            lane_frenet = behavioral_state.extended_lane_frames[rel_lane]  # the target GFF
            ego_fstate = behavioral_state.projected_ego_fstates[rel_lane]
            ego_fstates = np.tile(ego_fstate, len(lane_specs)).reshape((len(lane_specs), -1))

            # calculate polynomials
            poly_coefs_s, poly_coefs_d = KinematicUtils.calc_poly_coefs(specs_t, ego_fstates, goal_fstates, pad_mode)

            # create Frenet trajectories for s axis for all trajectories of rel_lane and for all time samples
            ftrajectories_s = QuinticPoly1D.polyval_with_derivatives(poly_coefs_s, time_samples)
            ftrajectories_d = QuinticPoly1D.polyval_with_derivatives(poly_coefs_d, time_samples)

            # Pad (extrapolate) short trajectories from spec.t until minimal action time.
            # Beyond the maximum between spec.t and minimal action time the Frenet trajectories are set to zero.
            ftrajectories[indices_by_rel_lane[rel_lane]] = ActionSpecFilter._pad_trajectories_beyond_spec(
                lane_specs, ftrajectories_s, ftrajectories_d, specs_t, pad_mode)

            # convert Frenet trajectories to cartesian trajectories
            ctrajectories[indices_by_rel_lane[rel_lane]] = lane_frenet.ftrajectories_to_ctrajectories(
                ftrajectories[indices_by_rel_lane[rel_lane]])

        return ftrajectories, ctrajectories

    @staticmethod
    def _pad_trajectories_beyond_spec(action_specs: List[ActionSpec], ftrajectories_s: np.array,
                                      ftrajectories_d: np.array, T: np.array, in_padding_mode: np.array) -> np.array:
        """
        Given action specs and their Frenet trajectories, pad (extrapolate) short trajectories from spec.t until
        minimal action time. Beyond the maximum between spec.t and minimal action time Frenet trajectories are set to
        zero.
        Important! Here we assume that zero Frenet states converted to Cartesian states pass all kinematic Cartesian
        filters.
        :param action_specs: list of actions spec
        :param ftrajectories_s: matrix Nx3 of N Frenet trajectories for s component
        :param ftrajectories_d: matrix Nx3 of N Frenet trajectories for d component
        :param T: array of size N: time horizons for each action
        :param in_padding_mode: boolean array of size N: True if an action is in padding mode
        :return: full Frenet trajectories (s & d)
        """
        # calculate trajectory time indices for all spec.t
        spec_t_idxs = (T / TRAJECTORY_TIME_RESOLUTION).astype(int) + 1
        spec_t_idxs[in_padding_mode] = 0

        # calculate trajectory time indices for t = max(spec.t, MINIMUM_REQUIRED_TRAJECTORY_TIME_HORIZON)
        last_pad_idxs = KinematicUtils.get_time_index_of_padded_actions(T) + 1

        # pad short ftrajectories beyond spec.t until MINIMUM_REQUIRED_TRAJECTORY_TIME_HORIZON
        for (spec_t_idx, last_pad_idx, trajectory_s, trajectory_d, spec) in \
                zip(spec_t_idxs, last_pad_idxs, ftrajectories_s, ftrajectories_d, action_specs):
            # if spec.t < MINIMUM_REQUIRED_TRAJECTORY_TIME_HORIZON, pad ftrajectories_s from spec.t to
            # MINIMUM_REQUIRED_TRAJECTORY_TIME_HORIZON
            if spec_t_idx < last_pad_idx:
                times_beyond_spec = np.arange(spec_t_idx, last_pad_idx) * TRAJECTORY_TIME_RESOLUTION - spec.t
                trajectory_s[spec_t_idx:last_pad_idx] = np.c_[spec.s + times_beyond_spec * spec.v,
                                                              np.full(times_beyond_spec.shape, spec.v),
                                                              np.zeros_like(times_beyond_spec)]
            # need to add padding, as we look at the whole trajectory to decide if we meet the speed limits
            # pad s beyond last_pad_idx: in case of short actions padding take the last s, otherwise take spec.s
            trajectory_s[last_pad_idx:] = 0
            trajectory_d[spec_t_idx:] = 0

        # return full Frenet trajectories
        return np.c_[ftrajectories_s, ftrajectories_d]

    def __str__(self):
        return self.__class__.__name__


class ActionSpecFiltering:
    """
    The gateway to execute filtering on one (or more) ActionSpec(s). From efficiency point of view, the filters
    should be sorted from the strongest (the one filtering the largest number of recipes) to the weakest.
    """
    def __init__(self, filters: Optional[List[ActionSpecFilter]], logger: Logger):
        self._filters: List[ActionSpecFilter] = filters or []
        self.logger = logger

    def filter_action_specs(self, action_specs: List[ActionSpec], behavioral_state: BehavioralGridState) -> List[bool]:
        """
        Filters a list of 'ActionSpec's based on the state of ego and nearby vehicles (BehavioralGridState).
        :param action_specs: A list of objects representing the specified actions to be considered
        :param behavioral_state: semantic behavioral state, containing the semantic grid
        :return: A boolean List , True where the respective action_spec is valid and false where it is filtered
        """
        mask = np.full(shape=len(action_specs), fill_value=True, dtype=np.bool)
        for action_spec_filter in self._filters:
            if ~np.any(mask):
                break

            # list of only valid action specs
            valid_action_specs = list(compress(action_specs, mask))

            # a mask only on the valid action specs
            current_mask = action_spec_filter.filter(valid_action_specs, behavioral_state)

            # use the reduced mask to update the original mask (that contains all initial actions specs given)
            mask[mask] = current_mask
        return mask.tolist()

    @prof.ProfileFunction()
    def filter_action_spec(self, action_spec: ActionSpec, behavioral_state: BehavioralGridState) -> bool:
        """
        Filters an 'ActionSpec's based on the state of ego and nearby vehicles (BehavioralGridState).
        :param action_spec: An object representing the specified actions to be considered
        :param behavioral_state: semantic behavioral state, containing the semantic grid
        :return: A boolean , True where the action_spec is valid and false where it is filtered
        """
        return self.filter_action_specs([action_spec], behavioral_state)[0]

