from collections import defaultdict

from decision_making.src.global_constants import TRAJECTORY_TIME_RESOLUTION, EPS, LAT_ACC_LIMITS, BP_ACTION_T_LIMITS, \
    DX_OFFSET_MIN, DX_OFFSET_MAX
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.behavioral_state import BehavioralState
from decision_making.src.planning.behavioral.data_objects import ActionSpec, RelativeLongitudinalPosition, RelativeLane
from decision_making.src.planning.behavioral.filtering.action_spec_filtering import \
    ActionSpecFilter
import numpy as np
from decision_making.src.planning.behavioral.filtering.recipe_filter_bank import FilterLimitsViolatingTrajectory
from decision_making.src.planning.trajectory.frenet_constraints import FrenetConstraints
from decision_making.src.planning.trajectory.werling_planner import WerlingPlanner
from decision_making.src.planning.types import C_K, C_V, FS_SX, FS_DX
from decision_making.src.planning.utils.numpy_utils import NumpyUtils
from decision_making.src.planning.utils.optimal_control.poly1d import QuinticPoly1D
from decision_making.src.planning.utils.safety_utils import SafetyUtils
from decision_making.src.prediction.ego_aware_prediction.road_following_predictor import RoadFollowingPredictor
from rte.python.logger.AV_logger import AV_Logger

from typing import List


class FilterIfNone(ActionSpecFilter):
    def filter(self, action_specs: List[ActionSpec], behavioral_state: BehavioralState) -> List[bool]:
        return [(action_spec and behavioral_state) is not None for action_spec in action_specs]


class FilterByLateralAcceleration(ActionSpecFilter):
    def __init__(self, predicates_dir: str):
        self.predicates = FilterLimitsViolatingTrajectory.read_predicates(predicates_dir, 'limits')
        self.distances = FilterLimitsViolatingTrajectory.read_predicates(predicates_dir, 'distances')

    def filter(self, action_specs: List[ActionSpec], behavioral_state: BehavioralGridState) -> List[bool]:
        """
        Check violation of lateral acceleration for action_specs, and beyond action_specs check ability to brake
        before all future curves using any static action.
        :return: specs list that passed the lateral acceleration filter
        """
        # first check lateral acceleration limits for all baseline trajectories of all action_specs
        meet_limits = FilterByLateralAcceleration.check_lateral_acceleration_limits(action_specs, behavioral_state)

        # now check ability to break before future curves beyond the baseline specs' trajectories
        filtering_res = [False]*len(action_specs)
        for spec_idx, spec in enumerate(action_specs):

            if spec is None or not meet_limits[spec_idx]:
                continue

            target_lane_frenet = behavioral_state.extended_lane_frames[spec.relative_lane]  # the target GFF
            # get the Frenet point index near the goal action_spec.s
            spec_s_point_idx = target_lane_frenet.get_index_on_frame_from_s(np.array([spec.s]))[0][0]
            # find all Frenet points beyond spec.s, where velocity limit (by curvature) is lower then spec.v
            beyond_spec_frenet_idxs = np.array(range(spec_s_point_idx + 1, len(target_lane_frenet.k), 4))
            curvatures = np.maximum(np.abs(target_lane_frenet.k[beyond_spec_frenet_idxs, 0]), EPS)

            points_velocity_limits = np.sqrt(LAT_ACC_LIMITS[1] / curvatures)
            slow_points = np.where(points_velocity_limits < spec.v)[0]  # points that require braking after spec

            # if all points beyond the spec have velocity limit higher than spec.v, so no need to brake
            if len(slow_points) == 0:
                filtering_res[spec_idx] = True
                continue  # the spec passes the filter

            # check the ability to brake beyond the spec for all points with limited velocity
            is_able_to_brake = ActionSpecFilter.check_ability_to_brake_beyond_spec(
                spec, behavioral_state.extended_lane_frames[spec.relative_lane],
                beyond_spec_frenet_idxs[slow_points], points_velocity_limits[slow_points], self.distances)

            filtering_res[spec_idx] = is_able_to_brake

        return filtering_res

    @staticmethod
    def check_lateral_acceleration_limits(action_specs: List[ActionSpec], behavioral_state: BehavioralGridState) -> np.array:
        """
        check meeting of lateral acceleration limits for the given specs list
        :param action_specs:
        :param behavioral_state:
        :return: bool array of size len(action_specs)
        """
        # group all specs and their indices by the relative lanes
        specs_by_rel_lane = defaultdict(list)
        indices_by_rel_lane = defaultdict(list)
        for i, spec in enumerate(action_specs):
            if spec is not None:
                specs_by_rel_lane[spec.relative_lane].append(spec)
                indices_by_rel_lane[spec.relative_lane].append(i)

        time_samples = np.arange(0, BP_ACTION_T_LIMITS[1], TRAJECTORY_TIME_RESOLUTION)
        lateral_accelerations = np.zeros((len(action_specs), len(time_samples)))

        # loop on the target relative lanes and calculate lateral accelerations for all relevant specs
        for rel_lane in specs_by_rel_lane.keys():
            lane_specs = specs_by_rel_lane[rel_lane]
            specs_t = np.array([spec.t for spec in lane_specs])
            goal_fstates = np.array([spec.as_fstate() for spec in lane_specs])

            frenet = behavioral_state.extended_lane_frames[rel_lane]  # the target GFF
            ego_fstate = behavioral_state.projected_ego_fstates[rel_lane]
            ego_fstates = np.tile(ego_fstate, len(lane_specs)).reshape((len(lane_specs), -1))

            # calculate polynomial coefficients of the spec's Frenet trajectory for s axis
            A_inv = QuinticPoly1D.inverse_time_constraints_tensor(specs_t)
            poly_coefs_s = QuinticPoly1D.zip_solve(A_inv, np.hstack((ego_fstates[:, FS_SX:FS_DX], goal_fstates[:, FS_SX:FS_DX])))

            # create Frenet trajectories for s axis for all trajectories of rel_lane and for all time samples
            ftrajectories_s = QuinticPoly1D.polyval_with_derivatives(poly_coefs_s, time_samples)
            # assign near-zero velocity to ftrajectories_s beyond spec.t
            for i, trajectory in enumerate(ftrajectories_s):
                trajectory[int(specs_t[i] / TRAJECTORY_TIME_RESOLUTION) + 1:] = np.array([EPS, EPS, 0])
            # assign zeros to the lateral movement of ftrajectories
            ftrajectories = np.concatenate((ftrajectories_s, np.zeros_like(ftrajectories_s)), axis=-1)

            # convert Frenet to cartesian trajectories
            lane_center_ctrajectories = frenet.ftrajectories_to_ctrajectories(ftrajectories)

            # calculate lateral accelerations
            lateral_accelerations[np.array(indices_by_rel_lane[rel_lane])] = \
                lane_center_ctrajectories[..., C_K] * lane_center_ctrajectories[..., C_V] ** 2

        return NumpyUtils.is_in_limits(lateral_accelerations, LAT_ACC_LIMITS).all(axis=-1)


class FilterUnsafeExpectedTrajectory(ActionSpecFilter):
    """
    This filter checks full safety (by RSS) toward the followed vehicle
    """
    def __init__(self):
        self.time_sum = 0
        self.time_num = 0

    def filter(self, action_specs: List[ActionSpec], behavioral_state: BehavioralGridState) -> List[ActionSpec]:
        """
        This filter checks for each action_spec if it's trajectory is safe w.r.t. the followed vehicle.
        :param action_specs:
        :param behavioral_state:
        :return: boolean array of size len(action_specs)
        """
        import time
        st = time.time()
        safe_specs = FilterUnsafeExpectedTrajectory._check_actions_safety(action_specs, behavioral_state)
        if (RelativeLane.SAME_LANE, RelativeLongitudinalPosition.FRONT) in behavioral_state.road_occupancy_grid:
            self.time_sum += time.time() - st
            self.time_num += 1
            if self.time_num % 10 == 1:
                print('safety time: %f' % (self.time_sum/self.time_num))
        return [spec if safe_specs[i] else None for i, spec in enumerate(action_specs)]

    @staticmethod
    def _check_actions_safety(action_specs: List[ActionSpec], behavioral_state: BehavioralGridState) -> np.array:
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

        are_specs_safe = np.full(len(action_specs), True)

        specs_by_rel_lane = defaultdict(list)
        indices_by_rel_lane = defaultdict(list)
        for i, spec in enumerate(action_specs):
            if spec is not None:
                specs_by_rel_lane[spec.relative_lane].append(spec)
                indices_by_rel_lane[spec.relative_lane].append(i)

        for rel_lane in specs_by_rel_lane.keys():

            # if there is no front object on the target lane, then all actions are safe
            if (rel_lane, RelativeLongitudinalPosition.FRONT) not in behavioral_state.road_occupancy_grid:
                continue

            lane_specs = specs_by_rel_lane[rel_lane]
            lane_frenet = behavioral_state.extended_lane_frames[rel_lane]
            ego_init_fstate = behavioral_state.projected_ego_fstates[rel_lane]

            # convert the specifications list to 2D matrix, where rows represent different specifications
            spec_arr = np.array([[spec.t, spec.s, spec.v] for i, spec in enumerate(lane_specs) if spec is not None])
            specs_t, specs_s, specs_v = np.split(spec_arr, 3, axis=1)
            specs_t, specs_s, specs_v = specs_t.flatten(), specs_s.flatten(), specs_v.flatten()

            # duplicate initial frenet states and create target frenet states based on the specifications
            zeros = np.zeros_like(specs_t)
            init_fstates = np.tile(ego_init_fstate, len(specs_t)).reshape(-1, 6)
            goal_fstates = np.c_[specs_s, specs_v, zeros, zeros, zeros, zeros]

            # calculate A_inv_d as a concatenation of inverse matrices for maximal T_d (= T_s) and for minimal T_d
            A_inv_s = QuinticPoly1D.inverse_time_constraints_tensor(specs_t)
            T_d_num = 3
            T_d_arr = FilterUnsafeExpectedTrajectory._calc_T_d_grid(ego_init_fstate[FS_DX:], specs_t, T_d_num)
            A_inv_d = QuinticPoly1D.inverse_time_constraints_tensor(T_d_arr)

            # create ftrajectories_s and duplicated ftrajectories_d (for max_T_d and min_T_d)
            constraints_s = np.concatenate((init_fstates[:, :FS_DX], goal_fstates[:, :FS_DX]), axis=1)
            constraints_d = np.concatenate((init_fstates[:, FS_DX:], goal_fstates[:, FS_DX:]), axis=1)
            duplicated_constraints_d = np.tile(constraints_d, T_d_num).reshape(-1, 6)
            poly_coefs_s = QuinticPoly1D.zip_solve(A_inv_s, constraints_s)
            poly_coefs_d = QuinticPoly1D.zip_solve(A_inv_d, duplicated_constraints_d)
            time_points = np.arange(0, np.max(specs_t) + EPS, TRAJECTORY_TIME_RESOLUTION)
            ftrajectories_s = QuinticPoly1D.polyval_with_derivatives(poly_coefs_s, time_points)
            ftrajectories_d = QuinticPoly1D.polyval_with_derivatives(poly_coefs_d, time_points)
            # for any T_d < T_s, complement ftrajectories_d to the length of T_s by setting zeros
            last_t_d_indices = np.floor(T_d_arr / TRAJECTORY_TIME_RESOLUTION).astype(int) + 1
            for i, ftrajectory_d in enumerate(ftrajectories_d):
                ftrajectory_d[last_t_d_indices[i]:] = 0

            # set all points beyond spec.t at infinity, such that they will be safe and will not affect the result
            end_traj_indices = np.floor(specs_t / TRAJECTORY_TIME_RESOLUTION).astype(int) + 1
            for i, ftrajectory_s in enumerate(ftrajectories_s):
                ftrajectory_s[end_traj_indices[i]:] = np.array([np.inf, 0, 0])

            # duplicate each longitudinal trajectory T_d_num times to be aligned with lateral trajectories
            duplicated_ftrajectories_s = np.tile(ftrajectories_s, (1, T_d_num, 1)).reshape(-1, len(time_points), 3)

            # create full Frenet trajectories
            ftrajectories = np.concatenate((duplicated_ftrajectories_s, ftrajectories_d), axis=-1)

            # predict objects' trajectories
            obj = behavioral_state.road_occupancy_grid[(rel_lane, RelativeLongitudinalPosition.FRONT)][0].dynamic_object
            obj_map_state = lane_frenet.convert_from_segment_state(obj.map_state.lane_fstate, obj.map_state.lane_id)
            obj_sizes = [obj.size]
            obj_trajectories = predictor.predict_frenet_states(np.array([obj_map_state]), time_points)

            # calculate safety for each trajectory, each object, each timestamp
            safety_costs = SafetyUtils.get_safety_costs(ftrajectories, ego_size, obj_trajectories, obj_sizes)
            # for each triple (ego_trajectory, obj_trajectory, time_sample) choose minimal cost among all T_d
            safety_costs = safety_costs.reshape(len(specs_t), T_d_num, len(obj_trajectories), len(time_points)).min(axis=1)

            # safe_specs = (safety_costs < 1).all(axis=(1, 2))
            # print('safe_specs: %s; specs_v %s, specs_t %s\nvel: %s\nacc: %s\ndist: %s\ncosts: %s: ' %
            #       (safe_specs, specs_v, specs_t, ftrajectories_s[-2, ::4, 1], ftrajectories_s[-2, ::4, 2],
            #        obj_trajectories[0, ::4, 0] - ftrajectories_s[-2, ::4, 0],
            #        safety_costs[-2, 0, ::4]))

            # trajectory is considered safe if it's safe wrt all dynamic objects for all timestamps
            are_specs_safe[np.array(indices_by_rel_lane[rel_lane])] = (safety_costs < 1).all(axis=(1, 2))

        return are_specs_safe

    @staticmethod
    def _calc_T_d_grid(fstate_d: np.array, T_s: np.array, T_d_num: int) -> np.array:
        """
        Calculate the lower bound of the lateral time horizon T_d_low_bound and return a grid of possible lateral
        planning time values.
        :param fstate_d: 1D array containing: current latitude, lateral velocity and lateral acceleration
        :param T_s: [m] longitudinal time horizon
        :return: numpy array (1D) of the possible lateral planning horizons
        """
        dt = TRAJECTORY_TIME_RESOLUTION
        dx = min(abs(fstate_d[0] + DX_OFFSET_MIN), abs(fstate_d[0] - DX_OFFSET_MAX))
        fconstraints_t0 = FrenetConstraints(0, 0, 0, dx, fstate_d[1], fstate_d[2])
        fconstraints_tT = FrenetConstraints(0, 0, 0, 0, 0, 0)
        lower_bound_T_d = WerlingPlanner.low_bound_lat_horizon(fconstraints_t0, fconstraints_tT, dt)
        T_d = np.repeat(T_s, T_d_num)
        T_d[0::T_d_num] = lower_bound_T_d
        T_d[1::T_d_num] = (lower_bound_T_d + T_s) / 2
        # T_d[2::T_d_num] = T_s
        return T_d
