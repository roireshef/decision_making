from collections import defaultdict

from decision_making.src.global_constants import BP_ACTION_T_LIMITS, TRAJECTORY_TIME_RESOLUTION, \
    MINIMUM_REQUIRED_TRAJECTORY_TIME_HORIZON
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import ActionSpec
from decision_making.src.planning.utils.generalized_frenet_serret_frame import GeneralizedFrenetSerretFrame
from decision_making.src.planning.utils.kinematics_utils import KinematicUtils
from decision_making.src.planning.utils.optimal_control.poly1d import QuinticPoly1D
from decision_making.src.utils.map_utils import MapUtils
from typing import List
import numpy as np


class ActionSpecTrajectoryBuilder:

    @staticmethod
    def build_trajectories(action_specs: List[ActionSpec], behavioral_state: BehavioralGridState,
                           get_lane_segment_velocities=False) -> (np.ndarray, np.ndarray):
        """
        Builds a baseline trajectory out of the action specs (terminal states)

        :param action_specs: list of action specs
        :param behavioral_state:
        :param get_lane_segment_velocities: Skip building the lane-segment-based velocities
        :return: A tuple of (cartesian_trajectories, lane_based_velocity_limits) the latter is all zero
        if build_lane_segment_velocities is False
        """
        # group all specs and their indices by the relative lanes
        specs_by_rel_lane = defaultdict(list)
        indices_by_rel_lane = defaultdict(list)
        for i, spec in enumerate(action_specs):
            if spec is not None:
                specs_by_rel_lane[spec.relative_lane].append(spec)
                indices_by_rel_lane[spec.relative_lane].append(i)

        time_samples = np.arange(0, BP_ACTION_T_LIMITS[1], TRAJECTORY_TIME_RESOLUTION)
        ctrajectories = np.zeros((len(action_specs), len(time_samples), 6), dtype=float)
        # velocity limits according to lane segments
        lane_segment_velocity_limits = np.zeros((len(action_specs), len(time_samples)), dtype=float)
        # loop on the target relative lanes and calculate lateral accelerations for all relevant specs
        for rel_lane, lane_specs in specs_by_rel_lane.items():
            specs_t = np.array([spec.t for spec in lane_specs])
            pad_mode = np.array([spec.only_padding_mode for spec in lane_specs])
            goal_fstates = np.array([spec.as_fstate() for spec in lane_specs])

            frenet = behavioral_state.extended_lane_frames[rel_lane]  # the target GFF
            ego_fstate = behavioral_state.projected_ego_fstates[rel_lane]
            ego_fstates = np.tile(ego_fstate, len(lane_specs)).reshape((len(lane_specs), -1))

            # calculate polynomials
            poly_coefs_s, poly_coefs_d = KinematicUtils.calc_poly_coefs(specs_t, ego_fstates, goal_fstates, pad_mode)

            # create Frenet trajectories for s axis for all trajectories of rel_lane and for all time samples
            ftrajectories_s = QuinticPoly1D.polyval_with_derivatives(poly_coefs_s, time_samples)
            ftrajectories_d = QuinticPoly1D.polyval_with_derivatives(poly_coefs_d, time_samples)

            # Pad (extrapolate) short trajectories from spec.t until minimal action time.
            # Beyond the maximum between spec.t and minimal action time the Frenet trajectories are set to zero.
            ftrajectories = ActionSpecTrajectoryBuilder.pad_trajectories_beyond_spec(lane_specs, ftrajectories_s,
                                                                             ftrajectories_d, specs_t, pad_mode)

            # convert Frenet trajectories to cartesian trajectories
            ctrajectories[indices_by_rel_lane[rel_lane]] = frenet.ftrajectories_to_ctrajectories(ftrajectories)
            if get_lane_segment_velocities:
                lane_segment_velocity_limits[indices_by_rel_lane[rel_lane]] = ActionSpecTrajectoryBuilder.\
                    create_trajectory_lane_speed_limits(ftrajectories, frenet)
        return ctrajectories, lane_segment_velocity_limits


    @staticmethod
    def create_trajectory_lane_speed_limits(ftrajectories: np.ndarray, frenet: GeneralizedFrenetSerretFrame) -> np.ndarray:
        """
        :param ftrajectories: The frenet trajectories to which to calculate the nominal speeds
        :return: A matrix of (Trajectories x Time_samples) of lane-based maximal limits (e_v_nominal_speed).
        """
        # get lane_the ids
        lane_ids_list = frenet.convert_to_segment_states(ftrajectories)[0]
        max_velocities = {lane_id: MapUtils.get_lane(lane_id).e_v_nominal_speed
                          for lane_id in np.unique(lane_ids_list)}
        # creates an ndarray with the same shape as of `lane_ids_list`,
        # where each element is replaced by the maximal speed limit (according to lane)
        return np.vectorize(max_velocities.get)(lane_ids_list)

    @staticmethod
    def pad_trajectories_beyond_spec(action_specs: List[ActionSpec], ftrajectories_s: np.array, ftrajectories_d: np.array,
                                     T: np.array, in_padding_mode: np.array) -> np.array:
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
        last_pad_idxs = (np.maximum(T, MINIMUM_REQUIRED_TRAJECTORY_TIME_HORIZON) / TRAJECTORY_TIME_RESOLUTION).astype(int) + 1

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
            trajectory_s[last_pad_idx:] = 0
            trajectory_d[spec_t_idx:] = 0

        # return full Frenet trajectories
        return np.c_[ftrajectories_s, ftrajectories_d]

