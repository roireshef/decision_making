import numpy as np
from logging import Logger
from typing import List

from decision_making.src import global_constants
from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from decision_making.src.messages.trajectory_parameters import SigmoidFunctionParams, TrajectoryCostParams, \
    TrajectoryParams
from decision_making.src.planning.behavioral.constants import LATERAL_SAFETY_MARGIN_FROM_OBJECT
from decision_making.src.planning.behavioral.semantic_actions_policy import SemanticActionsPolicy, \
    SemanticBehavioralState, RoadSemanticOccupancyGrid, SemanticActionSpec
from decision_making.src.planning.trajectory.trajectory_planning_strategy import TrajectoryPlanningStrategy
from decision_making.src.state.state import EgoState, State, DynamicObject
from mapping.src.model.constants import ROAD_SHOULDERS_WIDTH
from mapping.src.model.map_api import MapAPI
from mapping.src.transformations.geometry_utils import CartesianFrame

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
        :return: a new and updated BehavioralState
        """
        semantic_occupancy_grid = cls._generate_semantic_occupancy_grid(ego_state=state.ego_state,
                                                                        dynamic_objects=state.dynamic_objects,
                                                                        map_api=map_api,
                                                                        logger=logger)

        return NovDemoBehavioralState(road_occupancy_grid=semantic_occupancy_grid, ego_state=state.ego_state)

    @staticmethod
    def _generate_semantic_occupancy_grid(ego_state: EgoState, dynamic_objects: List[
        DynamicObject], map_api: MapAPI, logger: Logger) -> RoadSemanticOccupancyGrid:
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

        return semantic_occupancy_dict


class NovDemoPolicy(SemanticActionsPolicy):
    def plan(self, state: State, nav_plan: NavigationPlanMsg):
        behavioral_state = NovDemoBehavioralState.create_from_state(state=state, map_api=self._map_api,
                                                                    logger=self.logger)
        semantic_actions = self._enumerate_actions(behavioral_state=behavioral_state)
        actions_spec = [self._specify_action(behavioral_state=behavioral_state, semantic_action=semantic_actions[x]) for
                        x in range(len(semantic_actions))]
        action_values = self._eval_actions(behavioral_state=behavioral_state, semantic_actions=semantic_actions,
                                           actions_spec=actions_spec)

        selected_action_index = self._select_best(action_specs=actions_spec, costs=action_values)
        selected_action_spec = actions_spec[selected_action_index]

        reference_trajectory = self.__generate_reference_route(behavioral_state=behavioral_state,
                                                               action_spec=selected_action_spec,
                                                               navigation_plan=nav_plan)

        trajectory_parameters = self._generate_trajectory_specs(behavioral_state=behavioral_state,
                                                            action_spec=selected_action_spec,
                                                            reference_route=reference_trajectory)

        return trajectory_parameters


    def __generate_reference_route(self, behavioral_state: NovDemoBehavioralState, action_spec: SemanticActionSpec,
                                   navigation_plan: NavigationPlanMsg) -> np.ndarray:
        """
        :param behavioral_state: processed behavioral state
        :param target_lane_latitude: road latitude of reference route in [m]
        :return: [nx3] array of reference_route (x,y,yaw) [m,m,rad] in global coordinates
        """

        target_lane_latitude = action_spec.d_rel + behavioral_state.ego_state.road_localization.full_lat
        target_relative_longitude = action_spec.s_rel

        lookahead_path = self._map_api.get_uniform_path_lookahead(
            road_id=behavioral_state.ego_state.road_localization.road_id,
            lat_shift=target_lane_latitude,
            starting_lon=behavioral_state.ego_state.road_localization.road_lon,
            lon_step=global_constants.TRAJECTORY_ARCLEN_RESOLUTION,
            steps_num=int(np.round(target_relative_longitude /
                                   global_constants.TRAJECTORY_ARCLEN_RESOLUTION)),
            navigation_plan=navigation_plan)
        reference_route_xy = lookahead_path

        # interpolate and create uniformly spaced path
        reference_route_xy_resampled, _ = \
            CartesianFrame.resample_curve(curve=reference_route_xy,
                                          step_size=global_constants.TRAJECTORY_ARCLEN_RESOLUTION,
                                          desired_curve_len=global_constants.REFERENCE_TRAJECTORY_LENGTH,
                                          preserve_step_size=False)

        return reference_route_xy_resampled

    def _generate_trajectory_specs(self, behavioral_state: NovDemoBehavioralState, action_spec: SemanticActionSpec,
                                   reference_route: np.ndarray) -> TrajectoryParams:
        """
        Generate trajectory specification (cost) for trajectory planner
        :param behavioral_state: processed behavioral state
        :param target_path_latitude: road latitude of reference route in [m] from right-side of road
        :param safe_speed: safe speed in [m/s] (ACDA)
        :param reference_route: [nx3] numpy array of (x, y, z, yaw) states
        :return: Trajectory cost specifications [TrajectoryParameters]
        """

        # Get road details
        road_width = self._map_api.get_road(behavioral_state.ego_state.ego_road_id).road_width

        # Create target state
        target_path_latitude = action_spec.d_rel + behavioral_state.ego_state.road_localization.full_lat

        reference_route_x_y_yaw = CartesianFrame.add_yaw(reference_route)
        target_state_x_y_yaw = reference_route_x_y_yaw[-1, :]
        target_state_velocity = action_spec.v
        target_state = np.array(
            [target_state_x_y_yaw[0], target_state_x_y_yaw[1], target_state_x_y_yaw[2], target_state_velocity])

        # Define cost parameters
        # TODO: assign proper cost parameters
        infinite_sigmoid_cost = 5000.0  # not a constant because it might be learned. TBD
        deviation_from_road_cost = 10 ** -3  # not a constant because it might be learned. TBD
        deviation_to_shoulder_cost = 10 ** -3  # not a constant because it might be learned. TBD
        zero_sigmoid_cost = 0.0  # not a constant because it might be learned. TBD
        sigmoid_k_param = 10.0

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
        right_lane_cost = SigmoidFunctionParams(w=zero_sigmoid_cost, k=sigmoid_k_param,
                                                offset=right_lane_offset)  # Zero cost
        left_lane_cost = SigmoidFunctionParams(w=zero_sigmoid_cost, k=sigmoid_k_param,
                                               offset=left_lane_offset)  # Zero cost
        right_shoulder_cost = SigmoidFunctionParams(w=deviation_to_shoulder_cost, k=sigmoid_k_param,
                                                    offset=right_shoulder_offset)  # Very high cost
        left_shoulder_cost = SigmoidFunctionParams(w=deviation_to_shoulder_cost, k=sigmoid_k_param,
                                                   offset=left_shoulder_offset)  # Very high cost
        right_road_cost = SigmoidFunctionParams(w=deviation_from_road_cost, k=sigmoid_k_param,
                                                offset=right_road_offset)  # Very high cost
        left_road_cost = SigmoidFunctionParams(w=deviation_from_road_cost, k=sigmoid_k_param,
                                               offset=left_road_offset)  # Very high cost

        # Set objects parameters
        # dilate each object by cars length + safety margin
        objects_dilation_size = behavioral_state.ego_state.size.length + LATERAL_SAFETY_MARGIN_FROM_OBJECT
        objects_cost = SigmoidFunctionParams(w=infinite_sigmoid_cost, k=sigmoid_k_param,
                                             offset=objects_dilation_size)  # Very high (inf) cost

        distance_from_reference_route_sq_factor = 0.4
        # TODO: set velocity and acceleration limits properly
        velocity_limits = np.array([0.0, 50.0])  # [m/s]. not a constant because it might be learned. TBD
        acceleration_limits = np.array([-5.0, 5.0])  # [m/s^2]. not a constant because it might be learned. TBD
        cost_params = TrajectoryCostParams(left_lane_cost=left_lane_cost,
                                           right_lane_cost=right_lane_cost,
                                           left_road_cost=left_road_cost,
                                           right_road_cost=right_road_cost,
                                           left_shoulder_cost=left_shoulder_cost,
                                           right_shoulder_cost=right_shoulder_cost,
                                           obstacle_cost=objects_cost,
                                           dist_from_ref_sq_cost_coef=distance_from_reference_route_sq_factor,
                                           velocity_limits=velocity_limits,
                                           acceleration_limits=acceleration_limits)

        trajectory_parameters = TrajectoryParams(reference_route=reference_route,
                                                 time=action_spec.t,
                                                 target_state=target_state,
                                                 cost_params=cost_params,
                                                 strategy=TrajectoryPlanningStrategy.HIGHWAY)

        return trajectory_parameters
