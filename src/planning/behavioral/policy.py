from abc import ABCMeta, abstractmethod

import numpy as np

from decision_making.src import global_constants
from decision_making.src.messages.trajectory_parameters import TrajectoryParameters, TrajectoryCostParams
from decision_making.src.messages.visualization.behavioral_visualization_message import BehavioralVisualizationMsg
from decision_making.src.planning.behavioral.behavioral_state import BehavioralState
from decision_making.src.planning.utils import geometry_utils
from decision_making.src.planning.utils.geometry_utils import CartesianFrame
from decision_making.src.state.enriched_state import EnrichedState
from rte.python.logger.AV_logger import AV_Logger


class Policy(metaclass=ABCMeta):
    def __init__(self, logger: AV_Logger, policy_params: dict):
        self._policy_params = policy_params
        self.logger = logger

    @abstractmethod
    def plan(self, behavioral_state: BehavioralState) -> (TrajectoryParameters, BehavioralVisualizationMsg):
        pass


class DefaultPolicy(Policy):
    def __init__(self, logger: AV_Logger, policy_params: dict):
        super().__init__(logger=logger, policy_params=policy_params)
        self.behavioral_state = None

    def __high_level_planning(self, behavioral_state: BehavioralState) -> float:
        """
        Generates a high-level plan
        :return:
        selected_latitude: target latitude for driving, measured in [lanes]
        """

        lanes_in_current_road, _, _, _ = behavioral_state.map.get_road_details(behavioral_state.current_road_id)
        # remain in right most lane
        # return lanes_in_current_road

        current_lane_latitude = behavioral_state.current_lane + 0.5

        behavior_param_margin_from_road_edge = 0.2
        behavior_param_prefer_other_lanes_where_blocking_object_distance_less_than = 40
        behavior_param_prefer_other_lanes_if_improvement_is_greater_than = 5
        behavior_param_prefer_any_lane_center_if_blocking_object_distance_greater_than = 25
        behavior_param_assume_blocking_object_at_rear_if_distance_less_than = 0

        # Create a grid in latitude of lane offsets that will define latitude of the target trajectory.
        latitude_offset_grid_relative_to_current_center_lane = np.array([-1, -0.5, -0.25, 0, 0.25, 0.5, 1])
        absolute_latitude_offset_grid = current_lane_latitude + latitude_offset_grid_relative_to_current_center_lane
        indexes_past_right_road_margin = \
            np.where(absolute_latitude_offset_grid > lanes_in_current_road - behavior_param_margin_from_road_edge)[0]
        indexes_past_left_road_margin = \
            np.where(absolute_latitude_offset_grid < behavior_param_margin_from_road_edge)[0]
        valid_absolute_latitude_offset_grid = np.delete(absolute_latitude_offset_grid, np.concatenate(
            (indexes_past_right_road_margin, indexes_past_left_road_margin)))

        # Fetch each latitude offset attributes (free-space, blocking objects, etc.)
        lane_options = len(valid_absolute_latitude_offset_grid)
        closest_blocking_object_in_lane = np.inf * np.ones(shape=[lane_options, 1])

        # Assign object to optional lanes
        # TODO: set actual dilation
        CAR_WIDTH_DILATION_IN_LANES = 0.4
        CAR_LENGTH_DILATION_IN_METERS = 2
        for blocking_object in behavioral_state.static_objects:
            relative_lon = blocking_object['relative_lon']
            if relative_lon < behavior_param_assume_blocking_object_at_rear_if_distance_less_than:
                # If we passed an obstacle, treat it as at inf
                relative_lon = np.inf
            object_latitude_in_lanes = behavioral_state.map.convert_lat_in_meters_to_lat_in_lanes(
                road_id=blocking_object['road_id'], lat_in_meters=blocking_object['full_lat'])
            object_width_in_lanes = behavioral_state.map.convert_lat_in_meters_to_lat_in_lanes(
                road_id=blocking_object['road_id'], lat_in_meters=blocking_object['width'])
            leftmost_edge_in_lanes = object_latitude_in_lanes - 0.5 * object_width_in_lanes
            leftmost_edge_in_lanes_dilated = leftmost_edge_in_lanes - 0.5 * CAR_WIDTH_DILATION_IN_LANES

            rightmost_edge_lanes = object_latitude_in_lanes + 0.5 * object_width_in_lanes
            rightmost_edge_lanes_dilated = rightmost_edge_lanes + 0.5 * CAR_WIDTH_DILATION_IN_LANES

            affected_lanes = np.where((valid_absolute_latitude_offset_grid > leftmost_edge_in_lanes_dilated) & (
                valid_absolute_latitude_offset_grid < rightmost_edge_lanes_dilated))[0]
            closest_blocking_object_in_lane[affected_lanes] = np.minimum(
                closest_blocking_object_in_lane[affected_lanes],
                relative_lon)

        # High-level policy:
        # Choose a proper action (latitude offset from current center lane)
        current_center_lane_index_in_grid = np.where(valid_absolute_latitude_offset_grid == current_lane_latitude)[0][0]
        center_of_lane = (
            np.round(valid_absolute_latitude_offset_grid - 0.5) == valid_absolute_latitude_offset_grid - 0.5)
        other_center_lane_indexes_in_grid = \
            np.where((valid_absolute_latitude_offset_grid != current_lane_latitude) & center_of_lane)[
                0]  # check if integer

        chosen_action = current_center_lane_index_in_grid
        object_distance_in_current_lane = closest_blocking_object_in_lane[current_center_lane_index_in_grid]
        other_lanes_are_available = len(other_center_lane_indexes_in_grid) > 0
        best_center_of_lane_index_in_grid = other_center_lane_indexes_in_grid[
            np.argmax(valid_absolute_latitude_offset_grid[other_center_lane_indexes_in_grid])]
        best_center_of_lane_distance_from_object = closest_blocking_object_in_lane[best_center_of_lane_index_in_grid]

        # Prefer current center lane if nearest object is far enough
        chosen_action = current_center_lane_index_in_grid

        # Choose other lane only if improvement is sufficient
        if other_lanes_are_available and (
            object_distance_in_current_lane < behavior_param_prefer_other_lanes_where_blocking_object_distance_less_than) and (
                    best_center_of_lane_distance_from_object > object_distance_in_current_lane + behavior_param_prefer_other_lanes_if_improvement_is_greater_than):
            chosen_action = best_center_of_lane_index_in_grid

        # If blocking object is too close: choose any valid lateral offset
        if (
            object_distance_in_current_lane < behavior_param_prefer_any_lane_center_if_blocking_object_distance_greater_than) and (
                    best_center_of_lane_distance_from_object < behavior_param_prefer_any_lane_center_if_blocking_object_distance_greater_than):
            chosen_action = np.argmax(closest_blocking_object_in_lane)

        # Draw in RViz
        relevant_options_array = list()
        selected_option = list()
        for lat_option in valid_absolute_latitude_offset_grid:
            lat_option_in_lanes = lat_option
            lat_option_in_meters = behavioral_state.map.convert_lat_in_lanes_to_lat_in_meters(
                behavioral_state.current_road_id,
                lat_option_in_lanes)
            lookahead_path = behavioral_state.map.get_path_lookahead(road_id=behavioral_state.current_road_id,
                                                                     lon=behavioral_state.current_long,
                                                                     lat=lat_option_in_meters,
                                                                     max_lookahead_distance=global_constants.BEHAVIORAL_PLANNING_LOOKAHEAD_DIST,
                                                                     direction=1)

            lookahead_path = lookahead_path.transpose()
            lookahead_path_len = lookahead_path.shape[0]
            reference_route_xyz = np.concatenate((lookahead_path, np.zeros(shape=[lookahead_path_len, 1])), axis=1)

            if lat_option == valid_absolute_latitude_offset_grid[chosen_action]:
                selected_option.append(reference_route_xyz)
            else:
                relevant_options_array.append(reference_route_xyz)

        # self.rviz_object.updateSelectedOption(trajArr=selected_option)
        # self.rviz_object.updateLookaheadOptions(trajArr=relevant_options_array)

        # return best lane
        selected_latitude = valid_absolute_latitude_offset_grid[chosen_action]
        return selected_latitude


    def plan(self, behavioral_state: BehavioralState) -> (TrajectoryParameters, BehavioralVisualizationMsg):

        if behavioral_state is None:
            # Only happens on init
            return None, None

        ###################################################
        # High-level planning
        ###################################################
        target_lat_in_lanes = self.__high_level_planning()
        target_lane_latitude = behavioral_state.map.convert_lat_in_lanes_to_lat_in_meters(
            road_id=behavioral_state.current_road_id,
            lat_in_lanes=target_lat_in_lanes)

        ###################################################
        # Calculate reference route for driving
        ###################################################
        lookahead_path = behavioral_state.map.get_path_lookahead(road_id=behavioral_state.current_road_id,
                                                                 lon=behavioral_state.current_long,
                                                                 lat=target_lane_latitude,
                                                                 max_lookahead_distance=global_constants.REFERENCE_TRAJECTORY_LENGTH,
                                                                 direction=1)
        reference_route = lookahead_path.transpose()
        reference_route_len = reference_route.shape[0]

        # Transform into car's frame
        reference_route_xyz = np.concatenate((reference_route, np.zeros(shape=[reference_route_len, 1])), axis=1)
        reference_route_xyz_in_cars_frame = geometry_utils.CartesianFrame.get_vector_in_objective_frame(
            target_vector=reference_route_xyz.transpose(),
            ego_position=behavioral_state.current_position,
            ego_orientation=behavioral_state.current_orientation)
        reference_route_xy_in_cars_frame = reference_route_xyz_in_cars_frame[0:2, :].transpose()

        # interpolate and create uniformly spaced path
        reference_route_xy_in_cars_frame = CartesianFrame.resample_curve(curve=reference_route_xy_in_cars_frame,
                                                                         step_size=global_constants.TRAJECTORY_ARCLEN_RESOLUTION,
                                                                         desired_curve_len=global_constants.REFERENCE_TRAJECTORY_LENGTH,
                                                                         preserve_step_size=False)

        reference_route_xytheta_in_cars_frame = CartesianFrame.add_yaw_and_derivatives(
            reference_route_xy_in_cars_frame)[:, 0:3]

        #############################################
        # Calculating safe speed according to ACDA
        #############################################
        set_safety_lookahead_dist_by_ego_vel = False
        acda_safe_speed = compute_acda(behavioral_state.static_objects, behavioral_state.dynamic_objects,
                                       enriched_ego_state, reference_route_xy_in_cars_frame)
        safe_speed = min(acda_safe_speed, global_constants.BEHAVIORAL_PLANNING_CONSTANT_DRIVE_VELOCITY)

        if safe_speed < 0:
            self.logger.info("safe speed < 0")

        safe_speed = max(safe_speed, 0)

        lanes_num, road_width, _, _ = behavioral_state.map.get_road_details(behavioral_state.current_road_id)

        ###################################################
        # Generate specs for trajectory planner
        ###################################################
        target_state_xytheta = reference_route_xytheta_in_cars_frame[-1, :]
        target_state_velocity = safe_speed
        target_state = np.array(
            [target_state_xytheta[0], target_state_xytheta[1], target_state_xytheta[2], target_state_velocity])
        left_lane_offset = target_lane_latitude
        right_lane_offset = road_width - target_lane_latitude

        cost_params = TrajectoryCostParams(0, 0, 0, 0, 0, left_lane_offset,
                                           right_lane_offset, 0, 0, 0, 0, 0, 0, 0)
        trajectory_parameters = TrajectoryParameters(reference_route=reference_route_xytheta_in_cars_frame,
                                                     target_state=target_state,
                                                     cost_params=cost_params)

        visualization_message = BehavioralVisualizationMsg(reference_route=reference_route)
        return trajectory_parameters, visualization_message
