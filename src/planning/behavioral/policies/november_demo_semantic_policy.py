from logging import Logger
from typing import List

import math
import numpy as np

from decision_making.src.exceptions import BehavioralPlanningException
from decision_making.src.exceptions import NoValidTrajectoriesFound, raises
from decision_making.src.global_constants import BEHAVIORAL_PLANNING_DEFAULT_SPEED_LIMIT, TRAJECTORY_ARCLEN_RESOLUTION, \
    REFERENCE_TRAJECTORY_LENGTH
from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from decision_making.src.messages.trajectory_parameters import SigmoidFunctionParams, TrajectoryCostParams, \
    TrajectoryParams
from decision_making.src.messages.visualization.behavioral_visualization_message import BehavioralVisualizationMsg
from decision_making.src.planning.behavioral.constants import BEHAVIORAL_PLANNING_TRAJECTORY_HORIZON, \
    BP_SPECIFICATION_T_MIN, BP_SPECIFICATION_T_MAX, BP_SPECIFICATION_T_RES, A_LON_MIN, \
    A_LON_MAX, A_LAT_MIN, A_LAT_MAX, SAFE_DIST_TIME_DELAY, SEMANTIC_CELL_LON_FRONT, SEMANTIC_CELL_LON_REAR, \
    SEMANTIC_CELL_LON_SAME, SEMANTIC_CELL_LAT_SAME, SEMANTIC_CELL_LAT_LEFT, SEMANTIC_CELL_LAT_RIGHT, MIN_OVERTAKE_VEL, \
    LON_MARGIN_FROM_EGO, BEHAVIORAL_PLANNING_HORIZON, A_LON_EPS
from decision_making.src.planning.behavioral.constants import LATERAL_SAFETY_MARGIN_FROM_OBJECT
from decision_making.src.planning.behavioral.semantic_actions_policy import SemanticActionsPolicy, \
    SemanticBehavioralState, RoadSemanticOccupancyGrid, SemanticAction, SemanticActionSpec, SemanticActionType, \
    LAT_CELL
from decision_making.src.planning.trajectory.optimal_control.optimal_control_utils import OptimalControlUtils
from decision_making.src.planning.trajectory.trajectory_planning_strategy import TrajectoryPlanningStrategy
from decision_making.src.prediction.constants import LOOKAHEAD_MARGIN_DUE_TO_ROUTE_LINEARIZATION_APPROXIMATION
from decision_making.src.state.state import EgoState, State
from mapping.src.model.constants import ROAD_SHOULDERS_WIDTH
from mapping.src.model.map_api import MapAPI
from mapping.src.transformations.geometry_utils import CartesianFrame


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
        :return: road semantic occupancy grid
        """

        ego_state = state.ego_state
        dynamic_objects = state.dynamic_objects

        default_navigation_plan = map_api.get_road_based_navigation_plan(
            current_road_id=ego_state.road_localization.road_id)

        ego_lane = ego_state.road_localization.lane_num

        # Generate grid cells
        semantic_occupancy_dict: RoadSemanticOccupancyGrid = dict()
        optional_lane_keys = [-1, 0, 1]
        lanes_in_road = map_api.get_road(state.ego_state.road_localization.road_id).lanes_num
        filtered_lane_keys = list(
            filter(lambda relative_lane: 0 <= ego_lane + relative_lane < lanes_in_road, optional_lane_keys))

        optional_lon_keys = [SEMANTIC_CELL_LON_FRONT, SEMANTIC_CELL_LON_SAME, SEMANTIC_CELL_LON_REAR]
        for lon_key in optional_lon_keys:
            for lane_key in filtered_lane_keys:
                occupancy_index = (lane_key, lon_key)
                semantic_occupancy_dict[occupancy_index] = []

        # Allocate dynamic objects
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
                    occupancy_index = (object_relative_lane, SEMANTIC_CELL_LON_FRONT)

                else:
                    # Object behind vehicle
                    occupancy_index = (object_relative_lane, SEMANTIC_CELL_LON_REAR)

            elif object_relative_lane == 1 or object_relative_lane == -1:
                # Object is one lane on the left/right

                if object_dist_from_front > LON_MARGIN_FROM_EGO:
                    # Object in front of vehicle
                    occupancy_index = (object_relative_lane, SEMANTIC_CELL_LON_FRONT)

                elif object_lon_dist > -1 * LON_MARGIN_FROM_EGO:
                    # Object vehicle aside of ego
                    occupancy_index = (object_relative_lane, SEMANTIC_CELL_LON_SAME)

                else:
                    # Object behind rear of vehicle
                    occupancy_index = (object_relative_lane, SEMANTIC_CELL_LON_REAR)
            else:
                continue

            # Add object to occupancy grid
            # keeping only a single dynamic object per cell. List is used for future dev.
            # TODO: treat objects out of road
            if occupancy_index in semantic_occupancy_dict:

                if object_dist_from_front > BEHAVIORAL_PLANNING_HORIZON * \
                        0.5 * (ego_state.v_x + BEHAVIORAL_PLANNING_DEFAULT_SPEED_LIMIT):
                    continue

                if len(semantic_occupancy_dict[occupancy_index]) == 0:
                    # add to occupancy grid
                    semantic_occupancy_dict[occupancy_index].append(dynamic_object)
                else:
                    object_in_cell = semantic_occupancy_dict[occupancy_index][0]
                    object_in_grid_lon_dist = object_in_cell.get_relative_road_localization(
                        ego_road_localization=ego_state.road_localization,
                        ego_nav_plan=default_navigation_plan,
                        map_api=map_api, logger=logger).rel_lon
                    object_in_grid_dist_from_front = object_in_grid_lon_dist - ego_state.size.length

                    if occupancy_index[1] == SEMANTIC_CELL_LON_FRONT:
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
    def plan(self, state: State, nav_plan: NavigationPlanMsg):

        # TODO: this update is intended for visualization and should be moved to the Viz process
        # Update state: align all object to ego timestamp
        predicted_state = self._predictor.predict_state(state=state, prediction_timestamps=np.array(
            [state.ego_state.timestamp_in_sec]))
        state = predicted_state[0]

        # create road semantic grid from the raw State object
        behavioral_state = NovDemoBehavioralState.create_from_state(state=state, map_api=self._map_api,
                                                                    logger=self.logger)

        # iterate over the semantic grid and enumerate all relevant HL actions
        semantic_actions = self._enumerate_actions(behavioral_state=behavioral_state)

        # iterate over all HL actions and generate a specification (desired terminal: position, velocity, time-horizon)
        actions_spec = [self._specify_action(behavioral_state=behavioral_state, semantic_action=semantic_actions[idx])
                        for idx in range(len(semantic_actions))]

        # evaluate all action-specifications by computing a cost for each
        action_costs = self._eval_actions(behavioral_state=behavioral_state, semantic_actions=semantic_actions,
                                          actions_spec=actions_spec)

        # select an action-specification with minimal cost
        selected_action_index = int(np.argmax(action_costs))
        selected_action_spec = actions_spec[selected_action_index]

        # translate the selected action-specification into a full specification for the TP
        reference_trajectory = NovDemoPolicy._generate_reference_route(map_api=self._map_api,
                                                                       behavioral_state=behavioral_state,
                                                                       action_spec=selected_action_spec,
                                                                       navigation_plan=nav_plan)
        trajectory_parameters = NovDemoPolicy._generate_trajectory_specs(map_api=self._map_api,
                                                                         behavioral_state=behavioral_state,
                                                                         action_spec=selected_action_spec,
                                                                         reference_route=reference_trajectory)

        visualization_message = BehavioralVisualizationMsg(reference_route=reference_trajectory)

        self.logger.debug("Chosen behavioral semantic action is %s, %s",
                          semantic_actions[selected_action_index].__dict__, selected_action_spec.__dict__)

        return trajectory_parameters, visualization_message

    def _enumerate_actions(self, behavioral_state: NovDemoBehavioralState) -> List[SemanticAction]:
        """
        Enumerate the list of possible semantic actions to be generated.
        :param behavioral_state:
        :return:
        """

        semantic_actions: List[SemanticAction] = list()

        ego_lane = behavioral_state.ego_state.road_localization.lane_num
        optional_lane_keys = [-1, 0, 1]
        lanes_in_road = self._map_api.get_road(behavioral_state.ego_state.road_localization.road_id).lanes_num
        filtered_lane_keys = list(
            filter(lambda relative_lane: 0 <= ego_lane + relative_lane < lanes_in_road, optional_lane_keys))

        # Generate actions towards each of the cells in front of ego
        for relative_lane_key in filtered_lane_keys:
            for longitudinal_key in [SEMANTIC_CELL_LON_FRONT]:
                semantic_cell = (relative_lane_key, longitudinal_key)
                if semantic_cell in behavioral_state.road_occupancy_grid and len(
                        behavioral_state.road_occupancy_grid[semantic_cell]) > 0:
                    # Select first (closest) object in cell
                    target_obj = behavioral_state.road_occupancy_grid[semantic_cell][0]
                else:
                    # There are no objects in cell
                    target_obj = None

                semantic_action = SemanticAction(cell=semantic_cell, target_obj=target_obj,
                                                 action_type=SemanticActionType.FOLLOW)

                semantic_actions.append(semantic_action)

        return semantic_actions

    def _specify_action(self, behavioral_state: NovDemoBehavioralState,
                        semantic_action: SemanticAction) -> SemanticActionSpec:
        """
        For each semantic actions, generate a trajectory specifications that will be passed through to the TP
        :param behavioral_state:
        :param semantic_action:
        :return: semantic action spec
        """

        if semantic_action.target_obj is None:
            return NovDemoPolicy._specify_action_to_empty_cell(map_api=self._map_api,
                                                               behavioral_state=behavioral_state,
                                                               semantic_action=semantic_action)
        else:
            return NovDemoPolicy._specify_action_towards_object(map_api=self._map_api,
                                                                behavioral_state=behavioral_state,
                                                                semantic_action=semantic_action)

    def _eval_actions(self, behavioral_state: NovDemoBehavioralState, semantic_actions: List[SemanticAction],
                      actions_spec: List[SemanticActionSpec]) -> np.ndarray:
        """
        Evaluate the generated actions using the actions' spec and SemanticBehavioralState containing semantic grid.
        Gets a list of actions to evaluate so and returns a vector representing their costs.
        A set of actions is provided, enabling assessing them dependently.
        Note: the semantic actions were generated using the behavioral state and don't necessarily capture
         all relevant details in the scene. Therefore the evaluation is done using the behavioral state.
        :param behavioral_state: semantic behavioral state state, containing the semantic grid
        :param semantic_actions: semantic actions list
        :param actions_spec: specifications of semantic actions
        :return: numpy array of costs of semantic actions. Only one action is 1, the rest are zero.
        """

        if len(semantic_actions) != len(actions_spec):
            self.logger.error(
                "The input arrays have different sizes: len(semantic_actions)=%d, len(actions_spec)=%d",
                len(semantic_actions), len(actions_spec))
            raise BehavioralPlanningException(
                "The input arrays have different sizes: len(semantic_actions)=%d, len(actions_spec)=%d",
                len(semantic_actions), len(actions_spec))

        # get indices of semantic_actions array for 3 actions: goto-right, straight, goto-left
        current_lane_action_ind = NovDemoPolicy._get_action_ind_by_lane(semantic_actions, actions_spec,
                                                                        SEMANTIC_CELL_LAT_SAME)
        left_lane_action_ind = NovDemoPolicy._get_action_ind_by_lane(semantic_actions, actions_spec,
                                                                     SEMANTIC_CELL_LAT_LEFT)
        right_lane_action_ind = NovDemoPolicy._get_action_ind_by_lane(semantic_actions, actions_spec,
                                                                      SEMANTIC_CELL_LAT_RIGHT)

        # The cost for each action is assigned so that the preferred policy would be:
        # Go to right if right and current lanes are fast enough.
        # Go to left if the current lane is slow and the left lane is faster than the current.
        # Otherwise remain on the current lane.

        # TODO - this needs to come from map
        desired_vel = BEHAVIORAL_PLANNING_DEFAULT_SPEED_LIMIT

        # boolean whether the forward-right cell is fast enough (may be empty grid cell)
        is_forward_right_fast = right_lane_action_ind is not None and \
                                desired_vel - actions_spec[right_lane_action_ind].v < MIN_OVERTAKE_VEL
        # boolean whether the right cell near ego is occupied
        is_right_occupied = True
        if (SEMANTIC_CELL_LAT_RIGHT, SEMANTIC_CELL_LON_SAME) in behavioral_state.road_occupancy_grid:
            is_right_occupied = len(behavioral_state.road_occupancy_grid[(SEMANTIC_CELL_LAT_RIGHT,
                                                                          SEMANTIC_CELL_LON_SAME)]) > 0

        # boolean whether the forward cell is fast enough (may be empty grid cell)
        is_forward_fast = current_lane_action_ind is not None and \
                          desired_vel - actions_spec[current_lane_action_ind].v < MIN_OVERTAKE_VEL

        # boolean whether the forward-left cell is faster than the forward cell
        is_forward_left_faster = left_lane_action_ind is not None and (current_lane_action_ind is None or
                                                                       actions_spec[left_lane_action_ind].v -
                                                                       actions_spec[
                                                                           current_lane_action_ind].v >= MIN_OVERTAKE_VEL)
        # boolean whether the left cell near ego is occupied
        is_left_occupied = True
        if (SEMANTIC_CELL_LAT_LEFT, SEMANTIC_CELL_LON_SAME) in behavioral_state.road_occupancy_grid:
            is_left_occupied = len(behavioral_state.road_occupancy_grid[(SEMANTIC_CELL_LAT_LEFT,
                                                                         SEMANTIC_CELL_LON_SAME)]) > 0

        costs = np.zeros(len(semantic_actions))

        # move right if both straight and right lanes are fast
        # if is_forward_right_fast and (is_forward_fast or current_lane_action_ind is None) and not is_right_occupied:
        if is_forward_right_fast and not is_right_occupied:
            costs[right_lane_action_ind] = 1.
        # move left if straight is slow and the left is faster than straight
        elif not is_forward_fast and (
                    is_forward_left_faster or current_lane_action_ind is None) and not is_left_occupied:
            costs[left_lane_action_ind] = 1.
        else:
            costs[current_lane_action_ind] = 1.
        return costs

    @staticmethod
    def _generate_trajectory_specs(map_api: MapAPI, behavioral_state: NovDemoBehavioralState,
                                   action_spec: SemanticActionSpec, reference_route: np.ndarray) -> TrajectoryParams:
        """
        Generate trajectory specification (cost parameters) for trajectory planner
        :param behavioral_state: processed behavioral state
        :param reference_route: [nx3] numpy array of (x, y, z, yaw) states
        :return: Trajectory cost specifications [TrajectoryParameters]
        """

        # Get road details
        road_width = map_api.get_road(behavioral_state.ego_state.road_localization.road_id).road_width

        # Create target state
        target_path_latitude = action_spec.d_rel + behavioral_state.ego_state.road_localization.full_lat

        reference_route_x_y_yaw = CartesianFrame.add_yaw(reference_route)
        target_state_x_y_yaw = reference_route_x_y_yaw[-1, :]
        target_state_velocity = action_spec.v
        target_state = np.array(
            [target_state_x_y_yaw[0], target_state_x_y_yaw[1], target_state_x_y_yaw[2], target_state_velocity])

        # Define cost parameters
        # TODO: assign proper cost parameters
        infinite_sigmoid_cost = 2.0 * 1e2  # TODO: move to constants file
        deviation_from_road_cost = 1.0 * 1e2  # TODO: move to constants file
        deviation_to_shoulder_cost = 1.0 * 1e2  # TODO: move to constants file
        zero_sigmoid_cost = 0.0  # TODO: move to constants file
        road_sigmoid_k_param = 1000.0  # TODO: move to constants file
        sigmoid_k_param = 20.0  # TODO: move to constants file

        # lateral distance in [m] from ref. path to rightmost edge of lane
        left_margin = right_margin = behavioral_state.ego_state.size.width / 2 + LATERAL_SAFETY_MARGIN_FROM_OBJECT
        right_lane_offset = target_path_latitude - right_margin
        # lateral distance in [m] from ref. path to rightmost edge of lane
        left_lane_offset = (road_width - target_path_latitude) - left_margin
        # as stated above, for shoulders
        right_shoulder_offset = target_path_latitude - right_margin
        # as stated above, for shoulders
        left_shoulder_offset = (road_width - target_path_latitude) - left_margin
        # as stated above, for whole road including shoulders
        right_road_offset = right_shoulder_offset + ROAD_SHOULDERS_WIDTH
        # as stated above, for whole road including shoulders
        left_road_offset = left_shoulder_offset + ROAD_SHOULDERS_WIDTH

        # Set road-structure-based cost parameters
        right_lane_cost = SigmoidFunctionParams(w=zero_sigmoid_cost, k=road_sigmoid_k_param,
                                                offset=right_lane_offset)  # Zero cost
        left_lane_cost = SigmoidFunctionParams(w=zero_sigmoid_cost, k=road_sigmoid_k_param,
                                               offset=left_lane_offset)  # Zero cost
        right_shoulder_cost = SigmoidFunctionParams(w=deviation_to_shoulder_cost, k=road_sigmoid_k_param,
                                                    offset=right_shoulder_offset)  # Very high cost
        left_shoulder_cost = SigmoidFunctionParams(w=deviation_to_shoulder_cost, k=road_sigmoid_k_param,
                                                   offset=left_shoulder_offset)  # Very high cost
        right_road_cost = SigmoidFunctionParams(w=deviation_from_road_cost, k=road_sigmoid_k_param,
                                                offset=right_road_offset)  # Very high cost
        left_road_cost = SigmoidFunctionParams(w=deviation_from_road_cost, k=road_sigmoid_k_param,
                                               offset=left_road_offset)  # Very high cost

        # Set objects parameters
        # dilate each object by cars length + safety margin
        objects_dilation_size = behavioral_state.ego_state.size.length + LATERAL_SAFETY_MARGIN_FROM_OBJECT
        objects_cost = SigmoidFunctionParams(w=infinite_sigmoid_cost, k=sigmoid_k_param,
                                             offset=objects_dilation_size)  # Very high (inf) cost

        dist_from_goal_lon_sq_cost = 1.0 * 1e2
        dist_from_goal_lat_sq_cost = 1.5 * 1e2
        dist_from_ref_sq_cost = 0.0

        # TODO: set velocity and acceleration limits properly
        velocity_limits = np.array([0.0, 60.0])  # [m/s]. not a constant because it might be learned. TBD
        acceleration_limits = np.array(
            [A_LON_MIN - A_LON_EPS, A_LON_MAX + A_LON_EPS])  # [m/s^2]. not a constant because it might be learned. TBD
        cost_params = TrajectoryCostParams(left_lane_cost=left_lane_cost,
                                           right_lane_cost=right_lane_cost,
                                           left_road_cost=left_road_cost,
                                           right_road_cost=right_road_cost,
                                           left_shoulder_cost=left_shoulder_cost,
                                           right_shoulder_cost=right_shoulder_cost,
                                           obstacle_cost=objects_cost,
                                           dist_from_goal_lon_sq_cost=dist_from_goal_lon_sq_cost,
                                           dist_from_goal_lat_sq_cost=dist_from_goal_lat_sq_cost,
                                           dist_from_ref_sq_cost=dist_from_ref_sq_cost,
                                           velocity_limits=velocity_limits,
                                           acceleration_limits=acceleration_limits)

        trajectory_parameters = TrajectoryParams(reference_route=reference_route,
                                                 time=action_spec.t + behavioral_state.ego_state.timestamp_in_sec,
                                                 target_state=target_state,
                                                 cost_params=cost_params,
                                                 strategy=TrajectoryPlanningStrategy.HIGHWAY)

        return trajectory_parameters

    # TODO: rethink the design of this function
    @staticmethod
    def _specify_action_to_empty_cell(map_api: MapAPI, behavioral_state: NovDemoBehavioralState,
                                      semantic_action: SemanticAction) -> SemanticActionSpec:
        """
        Generate trajectory specification towards a target location in given cell considering ego speed, location.
        :param behavioral_state:
        :param semantic_action:
        :return:
        """
        road_lane_latitudes = map_api.get_center_lanes_latitudes(
            road_id=behavioral_state.ego_state.road_localization.road_id)
        target_lane = behavioral_state.ego_state.road_localization.lane_num + semantic_action.cell[LAT_CELL]
        target_lane_latitude = road_lane_latitudes[target_lane]

        # BEHAVIORAL_PLANNING_DEFAULT_SPEED_LIMIT * BEHAVIORAL_PLANNING_HORIZON
        target_relative_s = BEHAVIORAL_PLANNING_HORIZON * \
                            0.5 * (behavioral_state.ego_state.v_x + BEHAVIORAL_PLANNING_DEFAULT_SPEED_LIMIT)
        target_relative_d = target_lane_latitude - behavioral_state.ego_state.road_localization.full_lat

        return SemanticActionSpec(t=BEHAVIORAL_PLANNING_HORIZON, v=BEHAVIORAL_PLANNING_DEFAULT_SPEED_LIMIT,
                                  s_rel=target_relative_s, d_rel=target_relative_d)

    @staticmethod
    @raises(NoValidTrajectoriesFound)
    def _specify_action_towards_object(behavioral_state: NovDemoBehavioralState,
                                       semantic_action: SemanticAction, map_api: MapAPI) -> SemanticActionSpec:
        """
        given a state and a high level SemanticAction towards an object, generate a SemanticActionSpec
        :type map_api:
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
        road_lane_latitudes = map_api.get_center_lanes_latitudes(road_id=obj_on_road.road_id)
        obj_center_lane_latitude = road_lane_latitudes[obj_on_road.lane_num]
        # TODO: rotate speed v_x, v_y to road coordinated to get the actual lon/lat speed
        # obj_v_x = semantic_action.target_obj.road_longitudinal_speed
        # obj_v_y = semantic_action.target_obj.road_lateral_speed
        # obj_theta_diff = obj_on_road.intra_lane_yaw  # relative to road

        obj_sx0 = obj_on_road.road_lon  # TODO: handle different road_ids
        obj_sv0 = semantic_action.target_obj.road_longitudinal_speed
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

            # TODO: account for acc<>0 (from MobilEye's paper)
            safe_lon_dist = obj_svT * SAFE_DIST_TIME_DELAY

            # set of 6 constraints RHS values for quintic polynomial solution (S DIM)
            constraints_s = np.array(
                [ego_sx0, ego_sv0, ego_sa0, obj_sxT - safe_lon_dist - obj_long_margin, obj_svT, obj_saT])
            constraints_d = np.array([ego_dx0, ego_dv0, ego_da0, obj_center_lane_latitude, 0.0, 0.0])

            # solve for s(t) and d(t)
            poly_coefs_s = OptimalControlUtils.QuinticPoly1D.solve(A_inv, constraints_s[np.newaxis, :])[0]
            poly_coefs_d = OptimalControlUtils.QuinticPoly1D.solve(A_inv, constraints_d[np.newaxis, :])[0]

            # TODO: acceleration is computed in frenet frame and not cartesian. if road is curved, this is problematic
            if NovDemoPolicy._is_acceleration_in_limits(poly_coefs_s, T, A_LON_MIN, A_LON_MAX) and \
                    NovDemoPolicy._is_acceleration_in_limits(poly_coefs_d, T, A_LAT_MIN, A_LAT_MAX):
                return SemanticActionSpec(t=T, v=obj_svT, s_rel=constraints_s[3] - ego_sx0,
                                          d_rel=constraints_d[3] - ego_dx0)

        raise NoValidTrajectoriesFound("No valid trajectories found. action: {}, state: {}, "
                                       .format(semantic_action.__dict__, behavioral_state.__dict__))

    @staticmethod
    def _is_acceleration_in_limits(poly_coefs: np.ndarray, T: float,
                                   min_acc_threshold: float, max_acc_threshold: float) -> bool:
        """
        given a quintic polynomial coefficients vector, and restrictions
        on the acceleration values, return True if restrictions are met, False otherwise
        :param poly_coefs: 1D numpy array with s(t), s_dot(t) s_dotdot(t) concatenated
        :param T: planning time horizon [sec]
        :param min_acc_threshold: minimal allowed value of acceleration/deceleration [m/sec^2]
        :param max_acc_threshold: maximal allowed value of acceleration/deceleration [m/sec^2]
        :return: True if restrictions are met, False otherwise
        """
        # TODO: a(0) and a(T) checks are omitted as they they are provided by the user.
        # compute extrema points, by finding the roots of the 3rd derivative (which is itself a 2nd degree polynomial)
        acc_suspected_points_s = np.roots(np.polyder(poly_coefs, m=3))
        acceleration_poly_coefs = np.polyder(poly_coefs, m=2)
        acc_suspected_values_s = np.polyval(acceleration_poly_coefs, acc_suspected_points_s)

        # filter out extrema points out of [0, T]
        acc_inlimit_suspected_values_s = acc_suspected_values_s[np.greater_equal(acc_suspected_points_s, 0) &
                                                                np.less_equal(acc_suspected_points_s, T)]

        # check if extrema values are within [a_min, a_max] limits
        return np.all(np.greater_equal(acc_inlimit_suspected_values_s, min_acc_threshold) &
                      np.less_equal(acc_inlimit_suspected_values_s, max_acc_threshold))

    @staticmethod
    def _get_action_ind_by_lane(semantic_actions: List[SemanticAction], actions_spec: List[SemanticActionSpec],
                                cell_lane: int):
        """
        Given semantic actions array and relative lane index, return index of action matching to the given lane.
        For lane change action, verify that the action time is not smaller than MIN_CHANGE_LANE_TIME.
        :param semantic_actions: array of semantic actions
        :param actions_spec: array of actions spec
        :param cell_lane: the relative lane index (-1 right ,0 straight, 1 left)
        :return: the action index or None if the action does not exist
        """
        action_ind = [i for i, action in enumerate(semantic_actions) if action.cell[LAT_CELL] == cell_lane]
        # verify that change lane time is large enough
        if len(action_ind) > 0:
            return action_ind[0]
        else:
            return None

    @staticmethod
    def _generate_reference_route(map_api: MapAPI, behavioral_state: NovDemoBehavioralState,
                                  action_spec: SemanticActionSpec, navigation_plan: NavigationPlanMsg) -> np.ndarray:
        """
        :param behavioral_state: processed behavioral state
        :param action_spec: the goal of the action
        :return: [nx3] array of reference_route (x,y,yaw) [m,m,rad] in global coordinates
        """

        target_lane_latitude = action_spec.d_rel + behavioral_state.ego_state.road_localization.full_lat
        target_relative_longitude = action_spec.s_rel

        lookahead_distance = target_relative_longitude + LOOKAHEAD_MARGIN_DUE_TO_ROUTE_LINEARIZATION_APPROXIMATION

        lookahead_path = map_api.get_uniform_path_lookahead(
            road_id=behavioral_state.ego_state.road_localization.road_id,
            lat_shift=target_lane_latitude,
            starting_lon=behavioral_state.ego_state.road_localization.road_lon,
            lon_step=TRAJECTORY_ARCLEN_RESOLUTION,
            steps_num=int(np.round(lookahead_distance / TRAJECTORY_ARCLEN_RESOLUTION)),
            navigation_plan=navigation_plan)
        reference_route_xy = lookahead_path

        # interpolate and create uniformly spaced path
        reference_route_xy_resampled, _ = CartesianFrame.resample_curve(curve=reference_route_xy,
                                                                        step_size=TRAJECTORY_ARCLEN_RESOLUTION,
                                                                        desired_curve_len=target_relative_longitude,
                                                                        preserve_step_size=False)

        return reference_route_xy_resampled
