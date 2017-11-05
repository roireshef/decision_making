from logging import Logger

import numpy as np

from decision_making.src.exceptions import NoValidTrajectoriesFound, raises
from decision_making.src.global_constants import BEHAVIORAL_PLANNING_DEFAULT_SPEED_LIMIT
from decision_making.src.planning.behavioral.constants import BEHAVIORAL_PLANNING_TRAJECTORY_HORIZON, \
    BP_SPECIFICATION_T_MIN, BP_SPECIFICATION_T_MAX, BP_SPECIFICATION_T_RES, A_LON_MIN, \
    A_LON_MAX, A_LAT_MIN, A_LAT_MAX, SAFE_DIST_TIME_DELAY
from decision_making.src.planning.behavioral.semantic_actions_policy import SemanticActionsPolicy, \
    SemanticBehavioralState, RoadSemanticOccupancyGrid, SemanticAction, SemanticActionSpec, \
    SEMANTIC_CELL_LANE
from decision_making.src.planning.trajectory.optimal_control.optimal_control_utils import OptimalControlUtils
from decision_making.src.state.state import EgoState, State
from mapping.src.model.map_api import MapAPI

SEMANTIC_GRID_FRONT, SEMANTIC_GRID_ASIDE, SEMANTIC_GRID_BEHIND = 1, 0, -1
GRID_MID = 10

# The margin that we take from the front/read of the vehicle to define the front/rear partitions
SEMANTIC_OCCUPANCY_GRID_PARTITIONS_MARGIN_FROM_EGO = 1


class NovDemoBehavioralState(SemanticBehavioralState):
    def __init__(self, road_occupancy_grid: RoadSemanticOccupancyGrid, ego_state: EgoState):
        super().__init__(road_occupancy_grid=road_occupancy_grid)
        self.ego_state = ego_state

    @classmethod
    def create_from_state(cls, state: State, map_api: MapAPI, logger: Logger):
        """
        Occupy the occupancy grid.
        This method iterates over all dynamic objects, and fits them into the relevant cell
        in the semantic occupancy grid (semantic_lane, semantic_lon).
        Each cell holds a list of objects that are within the cell borders.
        In this particular implementation, we keep up to one dynamic object per cell, which is the closest ego.
         (e.g. in the cells in front of ego, we keep objects with minimal longitudinal distance
         relative to ego front, while in all other cells we keep the object with the maximal longitudinal distance from
         ego front).
        :param ego_state:
        :param dynamic_objects:
        :return: road semantic occupancy grid
        """
        ego_state = state.ego_state
        dynamic_objects = state.dynamic_objects

        default_navigation_plan = map_api.get_road_based_navigation_plan(
            current_road_id=ego_state.road_localization.road_id)

        ego_lane = ego_state.road_localization.lane_num

        semantic_occupancy_dict: RoadSemanticOccupancyGrid = dict()
        for dynamic_object in dynamic_objects:

            object_relative_localization = dynamic_object.get_relative_road_localization(
                ego_road_localization=ego_state.road_localization, ego_nav_plan=default_navigation_plan,
                map_api=map_api, logger=logger)
            object_lon_dist = object_relative_localization.rel_lon
            object_dist_from_front = object_lon_dist - ego_state.size.length
            object_relative_lane = int(dynamic_object.road_localization.lane_num - ego_lane)

            # Determine cell index in occupancy grid
            if object_relative_lane == 0:
                # Object is on same lane as ego
                if object_dist_from_front > 0.0:
                    # Object in front of vehicle
                    occupancy_index = (object_relative_lane, SEMANTIC_GRID_FRONT)

                else:
                    # Object behind vehicle
                    occupancy_index = (object_relative_lane, SEMANTIC_GRID_BEHIND)

            elif object_relative_lane == 1 or object_relative_lane == -1:
                # Object is one lane on the left/right

                if object_dist_from_front > SEMANTIC_OCCUPANCY_GRID_PARTITIONS_MARGIN_FROM_EGO:
                    # Object in front of vehicle
                    occupancy_index = (object_relative_lane, SEMANTIC_GRID_FRONT)

                elif object_lon_dist > -1 * SEMANTIC_OCCUPANCY_GRID_PARTITIONS_MARGIN_FROM_EGO:
                    # Object vehicle aside of ego
                    occupancy_index = (object_relative_lane, SEMANTIC_GRID_ASIDE)

                else:
                    # Object behind rear of vehicle
                    occupancy_index = (object_relative_lane, SEMANTIC_GRID_BEHIND)

            # Add object to occupancy grid
            # keeping only a single dynamic object per cell. List is used for future dev.
            if occupancy_index not in semantic_occupancy_dict:
                # add to occupancy grid
                semantic_occupancy_dict[occupancy_index] = [dynamic_object]
            else:
                object_in_cell = semantic_occupancy_dict[occupancy_index][0]
                object_in_grid_lon_dist = object_in_cell.get_relative_road_localization(
                    ego_road_localization=ego_state.road_localization,
                    ego_nav_plan=default_navigation_plan,
                    map_api=map_api, logger=logger).rel_lon
                object_in_grid_dist_from_front = object_in_grid_lon_dist - ego_state.size.length

                if occupancy_index[1] == SEMANTIC_GRID_FRONT:
                    # take the object with least lon
                    if object_lon_dist < object_in_grid_dist_from_front:
                        # replace object the the closer one
                        semantic_occupancy_dict[occupancy_index][0] = dynamic_object
                else:
                    # Assumption - taking the object with the largest long even in the ASIDE cells
                    # take the object with largest lon
                    if object_lon_dist > object_in_grid_dist_from_front:
                        # replace object the the closer one
                        semantic_occupancy_dict[occupancy_index][0] = dynamic_object

        return cls(semantic_occupancy_dict, ego_state)


class NovDemoPolicy(SemanticActionsPolicy):
    def _specify_action(self, behavioral_state: NovDemoBehavioralState,
                        semantic_action: SemanticAction) -> SemanticActionSpec:
        """
        For each semantic actions, generate a trajectory specifications that will be passed through to the TP
        :param behavioral_state:
        :param semantic_action:
        :return: semantic action spec
        """

        if semantic_action.target_obj is None:
            return self._specify_action_to_empty_cell(behavioral_state=behavioral_state,
                                                      semantic_action=semantic_action)
        else:
            return self._specify_action_towards_object(behavioral_state=behavioral_state,
                                                       semantic_action=semantic_action)

    # TODO: modify into a working+tested version
    def _specify_action_to_empty_cell(self, behavioral_state: NovDemoBehavioralState,
                                      semantic_action: SemanticAction) -> SemanticActionSpec:
        """
        Generate trajectory specification towards a target location in given cell considering ego speed, location.
        :param behavioral_state:
        :param semantic_action:
        :return:
        """
        road_lane_latitudes = self._map_api.get_center_lanes_latitudes(
            road_id=behavioral_state.ego_state.road_localization.road_id)
        target_lane = behavioral_state.ego_state.road_localization.lane_num + semantic_action.cell[SEMANTIC_CELL_LANE]
        target_lane_latitude = road_lane_latitudes[target_lane]

        target_relative_s = target_lane_latitude - behavioral_state.ego_state.road_localization.full_lat
        target_relative_d = BEHAVIORAL_PLANNING_DEFAULT_SPEED_LIMIT * BEHAVIORAL_PLANNING_TRAJECTORY_HORIZON

        return SemanticActionSpec(t=BEHAVIORAL_PLANNING_TRAJECTORY_HORIZON, v=BEHAVIORAL_PLANNING_DEFAULT_SPEED_LIMIT,
                                  s_rel=target_relative_s, d_rel=target_relative_d)

    @raises(NoValidTrajectoriesFound)
    def _specify_action_towards_object(self, behavioral_state: NovDemoBehavioralState,
                                       semantic_action: SemanticAction) -> SemanticActionSpec:
        """
        given a state and a high level SemanticAction towards an object, generate a SemanticActionSpec
        :param behavioral_state:
        :param semantic_action:
        :return:
        """

        # Extract relevant details from state on Ego
        ego_v_x = behavioral_state.ego_state.v_x
        ego_v_y = behavioral_state.ego_state.v_y

        ego_on_road = behavioral_state.ego_state.road_localization
        ego_theta_diff = ego_on_road.intra_lane_yaw  # relative to road

        ego_sx0 = ego_on_road.road_lon
        ego_sv0 = np.cos(ego_theta_diff) * ego_v_x + np.sin(ego_theta_diff) * ego_v_y
        ego_sa0 = 0.0  # TODO: to be changed to include acc

        ego_dx0 = ego_on_road.full_lat
        ego_dv0 = -np.sin(ego_theta_diff) * ego_v_x + np.cos(ego_theta_diff) * ego_v_y
        ego_da0 = 0.0  # TODO: to be changed to include acc

        # Extract relevant details from state on Reference-Object
        obj_on_road = semantic_action.target_obj.road_localization
        obj_v_x = semantic_action.target_obj.v_x
        obj_v_y = semantic_action.target_obj.v_y
        obj_theta_diff = obj_on_road.intra_lane_yaw  # relative to road

        obj_sx0 = obj_on_road.road_lon  # TODO: handle different road_ids
        obj_sv0 = np.cos(obj_theta_diff) * obj_v_x + np.sin(obj_theta_diff) * obj_v_y
        obj_sa0 = 0.0  # TODO: to be changed to include acc

        obj_dx0 = obj_on_road.full_lat

        obj_long_margin = semantic_action.target_obj.size.length

        for T in np.arange(BP_SPECIFICATION_T_MIN, BP_SPECIFICATION_T_MAX, BP_SPECIFICATION_T_RES):
            # TODO: should be cached in advance using OCU.QP1D.time_constraints_tensor
            A = OptimalControlUtils.QuinticPoly1D.time_constraints_matrix(T)
            A_inv = np.linalg.inv(A)

            # TODO: should be swapped with current implementation of Predictor
            obj_saT = obj_sa0
            obj_svT = obj_sv0 + obj_sa0 * T
            obj_sxT = obj_sx0 + obj_sv0 * T + obj_sa0 * T ** 2 / 2
            obj_dxT = obj_dx0  # assuming no agent's lateral movement

            # TODO: account for acc<>0 (from MobilEye's paper)
            safe_lon_dist = obj_svT * SAFE_DIST_TIME_DELAY

            # set of 6 constraints RHS values for quintic polynomial solution (S DIM)
            constraints_s = np.array([ego_sx0, ego_sv0, ego_sa0, obj_sxT, obj_svT, obj_saT - safe_lon_dist - obj_long_margin])
            constraints_d = np.array([ego_dx0, ego_dv0, ego_da0, obj_dxT, 0.0, 0.0])

            # solve for s(t) and d(t)
            poly_all_coefs_s = OptimalControlUtils.QuinticPoly1D.solve(A_inv, constraints_s[np.newaxis, :])[0]
            poly_all_coefs_d = OptimalControlUtils.QuinticPoly1D.solve(A_inv, constraints_d[np.newaxis, :])[0]

            # TODO: acceleration is computed in frenet frame and not cartesian. if road is curved, this is problematic
            if NovDemoPolicy._is_acceleration_in_limits(poly_all_coefs_s, T, A_LON_MIN, A_LON_MAX) and \
               NovDemoPolicy._is_acceleration_in_limits(poly_all_coefs_d, T, A_LAT_MIN, A_LAT_MAX):
                return SemanticActionSpec(t=T, v=obj_svT, s_rel=obj_sxT - ego_sx0, d_rel=obj_dxT - ego_dx0)

            raise NoValidTrajectoriesFound("No valid trajectories found. state: {}, action: {}"
                                           .format(behavioral_state, semantic_action))

    @staticmethod
    def _is_acceleration_in_limits(poly_all_coefs: np.ndarray, T: float,
                                   min_acc_threshold: float, max_acc_threshold: float) -> bool:
        """
        given a quintic polynomial coefficients vector, and restrictions
        on the acceleration values, return True if restrictions are met, False otherwise
        :param poly_all_coefs: 1D numpy array with s(t), s_dot(t) s_dotdot(t) concatenated
        :param T: planning time horizon [sec]
        :param min_acc_threshold: minimal allowed value of acceleration/deceleration [m/sec^2]
        :param max_acc_threshold: maximal allowed value of acceleration/deceleration [m/sec^2]
        :return: True if restrictions are met, False otherwise
        """
        # TODO: boundary values are redundant if they are provided by the user!
        # compute extrema points (add 0,T boundaries as well)
        acc_suspected_points_s = np.concatenate((
            OptimalControlUtils.QuinticPoly1D.find_second_der_extrema(poly_all_coefs[:6]),
            np.array([0.0, T])
        ))
        acc_suspected_values_s = np.polyval(poly_all_coefs[11:15], acc_suspected_points_s)

        # filter out extrema points out of [0, T]
        acc_inlimit_suspected_values_s = acc_suspected_values_s[np.greater_equal(acc_suspected_points_s, 0) &
                                                                np.less_equal(acc_suspected_points_s, T)]

        # check if extrema values are within [a_min, a_max] limits
        return np.all(np.greater_equal(acc_inlimit_suspected_values_s, min_acc_threshold) &
                      np.less_equal(acc_inlimit_suspected_values_s, max_acc_threshold))
