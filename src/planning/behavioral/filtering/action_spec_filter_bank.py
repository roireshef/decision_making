from collections import defaultdict

from decision_making.src.global_constants import BP_ACTION_T_LIMITS, TRAJECTORY_TIME_RESOLUTION, EPS, \
    MAX_SAFETY_T_D_GRID_SIZE, LAT_ACC_LIMITS
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.behavioral_state import BehavioralState
from decision_making.src.planning.behavioral.data_objects import ActionSpec, RelativeLongitudinalPosition, RelativeLane
from decision_making.src.planning.behavioral.filtering.action_spec_filtering import \
    ActionSpecFilter
from decision_making.src.planning.types import FS_SX, FrenetState2D, FrenetTrajectories2D, FS_DX, FS_SV
from decision_making.src.planning.utils.optimal_control.poly1d import QuinticPoly1D, Poly1D
from decision_making.src.prediction.ego_aware_prediction.road_following_predictor import RoadFollowingPredictor
from decision_making.src.state.state import State
from rte.python.logger.AV_logger import AV_Logger
from typing import List
import numpy as np
from decision_making.src.planning.utils.safety_utils import SafetyUtils


class FilterIfNone(ActionSpecFilter):
    def filter(self, action_specs: List[ActionSpec], behavioral_state: BehavioralState, state: State) -> List[bool]:
        return [(action_spec and behavioral_state) is not None for action_spec in action_specs]


class FilterUnsafeExpectedTrajectory(ActionSpecFilter):
    """
    This filter checks full safety (by RSS) toward the followed vehicle
    """
    def filter(self, action_specs: List[ActionSpec], behavioral_state: BehavioralGridState, state: State) -> List[bool]:
        """
        This filter checks for each action_spec if it's trajectory is safe w.r.t. the followed vehicle.
        :param action_specs:
        :param behavioral_state:
        :return: boolean array of size len(action_specs)
        """
        safe_specs = FilterUnsafeExpectedTrajectory._check_actions_safety(action_specs, behavioral_state, state)
        return safe_specs.tolist()

    @staticmethod
    def _check_actions_safety(action_specs: List[ActionSpec], behavioral_state: BehavioralGridState, state: State) -> np.array:
        """
        Check RSS safety for all action specs, for which action_specs_mask is true.
        An action spec is considered safe if it's safe wrt all dynamic objects for all timestamps < spec.t.
        :param action_specs: list of action specifications
        :param behavioral_state:
        :return: boolean list of safe specifications. The list's size is equal to the original action_specs size.
        Specifications filtered by action_specs_mask are considered "unsafe".
        """
        logger = AV_Logger.get_logger('_check_actions_safety')
        predictor = RoadFollowingPredictor(logger)
        ego_size = behavioral_state.ego_state.size
        behavioral_state.marginal_safety_per_action = np.zeros(len(action_specs))

        # TODO: Use dynamic objects and remove state from the Filter signature
        dynamic_objects = [dynamic_object for cell_objects in behavioral_state.road_occupancy_grid.values()
                           for dynamic_object in cell_objects]

        # collect action specs into at most 3 groups, according to their target lane
        safe_specs = np.full(len(action_specs), True)
        specs_by_rel_lane = defaultdict(list)
        indices_by_rel_lane = defaultdict(list)
        for i, spec in enumerate(action_specs):
            if spec is not None:
                specs_by_rel_lane[spec.relative_lane].append(spec)
                indices_by_rel_lane[spec.relative_lane].append(i)

        for rel_lane in specs_by_rel_lane.keys():
            lane_specs = specs_by_rel_lane[rel_lane]
            lane_frenet = behavioral_state.extended_lane_frames[rel_lane]
            ego_init_fstate = behavioral_state.projected_ego_fstates[rel_lane]
            T_d_grid_size = MAX_SAFETY_T_D_GRID_SIZE

            # find relevant objects for safety: front and from the target lane
            relevant_objects = []
            front_cell = (rel_lane, RelativeLongitudinalPosition.FRONT)
            if front_cell in behavioral_state.road_occupancy_grid:
                relevant_objects.append(behavioral_state.road_occupancy_grid[front_cell][0].dynamic_object)
            else:
                # if there is no front cell in behavioral state, maybe there is a far front object beyond
                # lookahead horizon, which may be relevant for safety for long follow-lane actions
                lane_ids = np.array([obj.map_state.lane_id if obj._cached_map_state is not None else 0
                                     for obj in state.dynamic_objects])
                same_lane_orig_idxs = np.where(lane_frenet.has_segment_ids(lane_ids))[0]
                if len(same_lane_orig_idxs) > 0:
                    same_lane_fstates = np.array([state.dynamic_objects[idx].map_state.lane_fstate
                                                  for idx in same_lane_orig_idxs])
                    projected_obj_fstates = lane_frenet.convert_from_segment_states(same_lane_fstates,
                                                                                    lane_ids[same_lane_orig_idxs])
                    front_idxs = np.where(projected_obj_fstates[:, FS_SX] > ego_init_fstate[FS_SX])[0]
                    if len(front_idxs) > 0:
                        closest_idx = same_lane_orig_idxs[front_idxs[np.argmin(projected_obj_fstates[front_idxs, FS_SX])]]
                        relevant_objects.append(state.dynamic_objects[closest_idx])

            if rel_lane != RelativeLane.SAME_LANE:
                for lon_cell in RelativeLongitudinalPosition:
                    if (rel_lane, lon_cell) in behavioral_state.road_occupancy_grid:
                        relevant_objects.append(behavioral_state.road_occupancy_grid[(rel_lane, lon_cell)][0].dynamic_object)

            obj_sizes = []
            obj_map_states = []
            for obj in relevant_objects:
                obj_sizes.append(obj.size)
                obj_map_states.append(lane_frenet.convert_from_segment_state(obj.map_state.lane_fstate, obj.map_state.lane_id))
            if len(obj_sizes) == 0:
                continue  # if there are no objects on the target lane, then all actions are safe

            # predict objects' trajectories
            time_points = np.arange(0, BP_ACTION_T_LIMITS[1] + EPS, TRAJECTORY_TIME_RESOLUTION)
            obj_trajectories = predictor.predict_2d_frenet_states(np.array(obj_map_states), time_points)

            # generate baseline Frenet trajectories for the relative-lane actions, with T_d grid
            ftrajectories = FilterUnsafeExpectedTrajectory._generate_frenet_trajectories_for_relative_lane(
                ego_init_fstate, lane_specs, T_d_grid_size, time_points)

            # calculate safety for each trajectory, each object, each timestamp
            marginal_safety = SafetyUtils.get_safe_distances(ftrajectories, ego_size, obj_trajectories, obj_sizes)
            safe_times = (marginal_safety > 0).any(axis=-1)  # OR on lon & lat safe distances

            # trajectory is considered safe if it's safe wrt all dynamic objects for all timestamps
            safe_trajectories = safe_times.all(axis=(1, 2))

            # marginal safety is the difference between actual distance from obstacle and the minimal RSS-safe distance
            behavioral_state.marginal_safety_per_action[np.array(indices_by_rel_lane[rel_lane])] = \
                np.max(np.min(marginal_safety[..., 0], axis=(1, 2)).reshape(len(lane_specs), T_d_grid_size), axis=1)

            # find trajectories that are safe for any T_d
            safe_specs[np.array(indices_by_rel_lane[rel_lane])] = \
                safe_trajectories.reshape(len(lane_specs), T_d_grid_size).any(axis=1)  # OR on different T_d

        return safe_specs

    @staticmethod
    def _generate_frenet_trajectories_for_relative_lane(ego_init_fstate: FrenetState2D, lane_specs: List[ActionSpec],
                                                        T_d_grid_size: int, time_points: np.array) -> FrenetTrajectories2D:
        """
        Generate baseline Frenet trajectories for action specifications, with T_d grid.
        :param ego_init_fstate: initial ego frenet state
        :param lane_specs: action specifications list
        :param T_d_grid_size: the size of T_d grid
        :param time_points: array of time samples for the generated trajectories
        :return: the generated Frenet trajectories (3D numpy array)
        """
        # convert the specifications list to 2D matrix, where rows represent different specifications
        spec_arr = np.array([[spec.t, spec.s, spec.v] for i, spec in enumerate(lane_specs)])
        specs_t, specs_s, specs_v = np.split(spec_arr, 3, axis=1)
        specs_t, specs_s, specs_v = specs_t.flatten(), specs_s.flatten(), specs_v.flatten()

        # duplicate initial frenet states and create target frenet states based on the specifications
        zeros = np.zeros_like(specs_t)
        init_fstates = np.tile(ego_init_fstate, len(specs_t)).reshape(-1, len(ego_init_fstate))
        goal_fstates = np.c_[specs_s, specs_v, zeros, zeros, zeros, zeros]

        # calculate A_inv_d as a concatenation of inverse matrices for maximal T_d (= T_s) and for minimal T_d
        A_inv_s = QuinticPoly1D.inverse_time_constraints_tensor(specs_t)
        T_d_arr = FilterUnsafeExpectedTrajectory._calc_T_d_grid(ego_init_fstate[FS_DX:], specs_t, T_d_grid_size)
        A_inv_d = QuinticPoly1D.inverse_time_constraints_tensor(T_d_arr)

        # create ftrajectories_s and duplicated ftrajectories_d (for max_T_d and min_T_d)
        constraints_s = np.concatenate((init_fstates[:, :FS_DX], goal_fstates[:, :FS_DX]), axis=1)
        constraints_d = np.concatenate((init_fstates[:, FS_DX:], goal_fstates[:, FS_DX:]), axis=1)
        duplicated_constraints_d = np.tile(constraints_d, T_d_grid_size).reshape(-1, init_fstates.shape[-1])
        poly_coefs_s = QuinticPoly1D.zip_solve(A_inv_s, constraints_s)
        poly_coefs_d = QuinticPoly1D.zip_solve(A_inv_d, duplicated_constraints_d)
        ftrajectories_s = QuinticPoly1D.polyval_with_derivatives(poly_coefs_s, time_points)
        ftrajectories_d = QuinticPoly1D.polyval_with_derivatives(poly_coefs_d, time_points)
        # for any T_d < T_s, complement ftrajectories_d to the length of T_s by setting zeros
        last_t_d_indices = np.floor(T_d_arr / TRAJECTORY_TIME_RESOLUTION).astype(int) + 1
        for i, ftrajectory_d in enumerate(ftrajectories_d):
            ftrajectory_d[last_t_d_indices[i]:] = 0

        # extrapolate points beyond spec.t
        end_traj_indices = np.floor(specs_t / TRAJECTORY_TIME_RESOLUTION).astype(int) + 1
        for i, ftrajectory_s in enumerate(ftrajectories_s):
            ftrajectory_s[end_traj_indices[i]:, FS_SX] = specs_s[i] + (time_points[end_traj_indices[i]:] - specs_t[i]) * specs_v[i]
            ftrajectory_s[end_traj_indices[i]:, FS_SV:] = np.array([specs_v[i], 0])

        # duplicate each longitudinal trajectory T_d_num times to be aligned with lateral trajectories
        duplicated_ftrajectories_s = np.tile(ftrajectories_s, (1, T_d_grid_size, 1)).reshape(-1, len(time_points), 3)

        # return full Frenet trajectories
        return np.concatenate((duplicated_ftrajectories_s, ftrajectories_d), axis=-1)

    @staticmethod
    def _calc_T_d_grid(fstate_d: np.array, T_s: np.array, T_d_num: int) -> np.array:
        """
        Calculate the lower bound of the lateral time horizon T_d_low_bound and return a grid of possible lateral
        planning time values based on lateral acceleration limits.
        :param fstate_d: 1D array containing: current latitude, lateral velocity and lateral acceleration
        :param T_s: [m] longitudinal time horizon
        :return: numpy array (1D) of the possible lateral planning horizons
                The output array size is len(T_s) * T_d_num, such that for each T_s value there are T_d_num
                values of T_d.
        """
        T_d_grid = np.arange(TRAJECTORY_TIME_RESOLUTION, np.max(T_s) + EPS, TRAJECTORY_TIME_RESOLUTION)
        A_inv = QuinticPoly1D.inverse_time_constraints_tensor(T_d_grid)
        constraints_d = np.concatenate((fstate_d, np.zeros(3)))
        duplicated_constraints = np.tile(constraints_d, len(T_d_grid)).reshape(-1, 6)
        poly_coefs_d = QuinticPoly1D.zip_solve(A_inv, duplicated_constraints)
        acc_in_limits = Poly1D.are_accelerations_in_limits(poly_coefs_d, T_d_grid, LAT_ACC_LIMITS)
        T_d_min = T_d_grid[np.argmax(acc_in_limits)]
        # Create array of size len(T_s) * T_d_num. If for example T_d_num=3, then the output array looks:
        # [T_d_min, (T_s[0]+T_d_min)/2, T_s[0],
        #  T_d_min, (T_s[1]+T_d_min)/2, T_s[1],
        #  ...]
        T_d = np.repeat(T_s, T_d_num)
        T_d[0::T_d_num] = T_d_min
        T_d[1::T_d_num] = (T_d_min + T_s) / 2
        # T_d[2::T_d_num] = T_s
        return T_d
