from logging import Logger
import numpy as np

from decision_making.src import global_constants
from decision_making.src.messages.trajectory_parameters import TrajectoryCostParams
from decision_making.src.messages.trajectory_parameters import TrajectoryParameters
from decision_making.src.messages.visualization.behavioral_visualization_message import BehavioralVisualizationMsg
from decision_making.src.planning.behavioral.behavioral_state import BehavioralState
from decision_making.src.planning.behavioral.policy import Policy, PolicyConfig
from decision_making.src.planning.trajectory.trajectory_planning_strategy import TrajectoryPlanningStrategy
from decision_making.src.planning.utils import geometry_utils
from decision_making.src.planning.utils.acda import AcdaApi
from decision_making.src.planning.utils.geometry_utils import CartesianFrame


class DefaultPolicyConfig(PolicyConfig):
    def __init__(self, margin_from_road_edge: float = 0.2,
                 prefer_other_lanes_where_blocking_object_distance_less_than: float = 40.0,
                 prefer_other_lanes_if_improvement_is_greater_than: float = 5.0,
                 prefer_any_lane_center_if_blocking_object_distance_greater_than: float = 25.0,
                 assume_blocking_object_at_rear_if_distance_less_than: float = 0.0):
        self.margin_from_road_edge = margin_from_road_edge
        super().__init__()

        self.prefer_other_lanes_where_blocking_object_distance_less_than = \
            prefer_other_lanes_where_blocking_object_distance_less_than
        self.prefer_other_lanes_if_improvement_is_greater_than = \
            prefer_other_lanes_if_improvement_is_greater_than
        self.prefer_any_lane_center_if_blocking_object_distance_greater_than = \
            prefer_any_lane_center_if_blocking_object_distance_greater_than
        self.assume_blocking_object_at_rear_if_distance_less_than = \
            assume_blocking_object_at_rear_if_distance_less_than


class DefaultPolicy(Policy):

    def __init__(self, logger: Logger, policy_config: DefaultPolicyConfig):
        super().__init__(logger=logger, policy_config=policy_config)

    def plan(self, behavioral_state: BehavioralState) -> (TrajectoryParameters, BehavioralVisualizationMsg):
        if behavioral_state.current_timestamp is None:
            # Only happens on init
            return None, None

        # High-level planning
        target_lat_in_lanes = self.__high_level_planning(behavioral_state)
        target_lane_latitude = behavioral_state.map.convert_lat_in_lanes_to_lat_in_meters(
            road_id=behavioral_state.current_road_id,
            lat_in_lanes=target_lat_in_lanes)

        # Calculate reference route for driving
        reference_route_x_y_z, reference_route_in_cars_frame_x_y_yaw = DefaultPolicy.__generate_reference_route(
            behavioral_state,
            target_lane_latitude)

        # Calculate safe speed according to ACDA
        acda_safe_speed = AcdaApi.compute_acda(objects_on_road=behavioral_state.dynamic_objects,
                                               ego_state=behavioral_state.ego_state,
                                               lookahead_path=reference_route_in_cars_frame_x_y_yaw[:, 0:2])
        safe_speed = min(acda_safe_speed, global_constants.BEHAVIORAL_PLANNING_CONSTANT_DRIVE_VELOCITY)

        if safe_speed < 0:
            self.logger.warning("safe speed < 0")

        safe_speed = max(safe_speed, 0)

        # Generate specs for trajectory planner
        trajectory_parameters = \
            DefaultPolicy.__generate_trajectory_specs(behavioral_state=behavioral_state,
                                                      safe_speed=safe_speed,
                                                      target_lane_latitude=target_lane_latitude,
                                                      reference_route=reference_route_in_cars_frame_x_y_yaw)

        visualization_message = BehavioralVisualizationMsg(reference_route=reference_route_x_y_z)
        return trajectory_parameters, visualization_message

    def __high_level_planning(self, behavioral_state: BehavioralState) -> float:
        """
        Generates a high-level plan
        :param behavioral_state: processed behavioral state
        selected_latitude: target latitude for driving, measured in [lanes]
        """

        from decision_making.src.planning.behavioral.policy_features import PolicyFeatures

        lanes_in_current_road, road_width, _, _ = behavioral_state.map.get_road_details(
            behavioral_state.ego_state.road_localization.road_id)
        lane_width_in_meters = float(road_width) / lanes_in_current_road
        # remain in right most lane
        # return lanes_in_current_road

        current_lane_latitude = behavioral_state.ego_state.road_localization.lane + 0.5

        # Load policy parameters config
        with self._policy_config as pc:

            # Create a grid in latitude of lane offsets that will define latitude of the target trajectory.
            latitude_offset_grid_relative_to_current_center_lane = np.array([-1, -0.5, -0.25, 0, 0.25, 0.5, 1])
            absolute_latitude_offset_grid_in_lanes = current_lane_latitude + \
                                                     latitude_offset_grid_relative_to_current_center_lane
            num_of_latitude_options = len(latitude_offset_grid_relative_to_current_center_lane)
            latitude_options_in_lanes = [absolute_latitude_offset_grid_in_lanes[ind] for ind in
                                         range(num_of_latitude_options)
                                         if ((absolute_latitude_offset_grid_in_lanes[ind] >
                                              lanes_in_current_road - pc.margin_from_road_edge)
                                             and (absolute_latitude_offset_grid_in_lanes[ind]
                                                  < 0.0 + pc.margin_from_road_edge))]
            latitude_options_in_meters = lane_width_in_meters * latitude_options_in_lanes

            # For each lateral offset, find closest blocking object on lane
            closest_blocking_object_in_lane = \
                PolicyFeatures.get_closest_object_on_lane(policy_config=self._policy_config,
                                                          behavioral_state=behavioral_state,
                                                          lat_options=latitude_options_in_meters)

            # High-level policy:
            # Choose a proper action (latitude offset from current center lane)
            current_center_lane_index_in_grid = \
                np.where(latitude_options_in_lanes == current_lane_latitude)[0][0]
            center_of_lane = (
                np.round(latitude_options_in_lanes - 0.5) == latitude_options_in_lanes - 0.5)
            other_center_lane_indexes_in_grid = \
                np.where((latitude_options_in_lanes != current_lane_latitude) & center_of_lane)[
                    0]  # check if integer

            object_distance_in_current_lane = closest_blocking_object_in_lane[current_center_lane_index_in_grid]
            other_lanes_are_available = len(other_center_lane_indexes_in_grid) > 0
            best_center_of_lane_index_in_grid = other_center_lane_indexes_in_grid[
                np.argmax(latitude_options_in_lanes[other_center_lane_indexes_in_grid])]
            best_center_of_lane_distance_from_object = closest_blocking_object_in_lane[
                best_center_of_lane_index_in_grid]

            # Prefer current center lane if nearest object is far enough
            chosen_action = current_center_lane_index_in_grid

            # Choose other lane only if improvement is sufficient
            if other_lanes_are_available \
                    and (
                                object_distance_in_current_lane <
                                pc.prefer_other_lanes_where_blocking_object_distance_less_than) \
                    and (
                                best_center_of_lane_distance_from_object >
                                    object_distance_in_current_lane +
                                    pc.prefer_other_lanes_if_improvement_is_greater_than):
                chosen_action = best_center_of_lane_index_in_grid

            # If blocking object is too close: choose any valid lateral offset
            if (object_distance_in_current_lane < pc.prefer_any_lane_center_if_blocking_object_distance_greater_than) \
                    and (best_center_of_lane_distance_from_object <
                             pc.prefer_any_lane_center_if_blocking_object_distance_greater_than):
                chosen_action = np.argmax(closest_blocking_object_in_lane)

        # Debug objects
        relevant_options_array = list()
        selected_option = list()
        for lat_option_in_meters in latitude_options_in_meters:
            # Generate lookahead path per each lateral option for debugging and visualization purposes
            lookahead_path = behavioral_state.map.get_path_lookahead(
                road_id=behavioral_state.ego_state.road_localization.road_id,
                lon=behavioral_state.ego_state.road_localization.road_lon, lat=lat_option_in_meters,
                max_lookahead_distance=global_constants.BEHAVIORAL_PLANNING_LOOKAHEAD_DIST, direction=1)

            lookahead_path = lookahead_path.transpose()
            lookahead_path_len = lookahead_path.shape[0]
            reference_route_xyz = np.concatenate((lookahead_path, np.zeros(shape=[lookahead_path_len, 1])), axis=1)

            if lat_option_in_meters == latitude_options_in_meters[chosen_action]:
                selected_option.append(reference_route_xyz)
            else:
                relevant_options_array.append(reference_route_xyz)

        # return best lane
        selected_latitude = latitude_options_in_lanes[chosen_action]
        return selected_latitude

    @staticmethod
    def __generate_reference_route(behavioral_state: BehavioralState, target_lane_latitude: float) -> (
            np.array, np.array):
        """
        :param behavioral_state: processed behavioral state
        :param target_lane_latitude: road latitude of reference route in [m]
        :return: [nx3] array of reference_route (x,y,z) [m,m,m] in world coordinates,
         [nx3] array of reference_route (x,y,yaw) [m,m,rad] in cars coordinates
        """
        lookahead_path = behavioral_state.map.get_path_lookahead(road_id=behavioral_state.current_road_id,
                                                                 lon=behavioral_state.current_long,
                                                                 lat=target_lane_latitude,
                                                                 max_lookahead_distance=global_constants.REFERENCE_TRAJECTORY_LENGTH,
                                                                 direction=1)
        reference_route_xy = lookahead_path.transpose()
        reference_route_len = reference_route_xy.shape[0]

        # Transform into car's frame
        reference_route_x_y_z = np.concatenate((reference_route_xy, np.zeros(shape=[reference_route_len, 1])),
                                               axis=1)
        reference_route_xyz_in_cars_frame = geometry_utils.CartesianFrame.get_vector_in_objective_frame(
            target_vector=reference_route_x_y_z.transpose(),
            ego_position=behavioral_state.current_position,
            ego_orientation=behavioral_state.current_orientation)
        reference_route_xy_in_cars_frame = reference_route_xyz_in_cars_frame[0:2, :].transpose()

        # interpolate and create uniformly spaced path
        reference_route_xy_in_cars_frame = \
            CartesianFrame.resample_curve(curve=reference_route_xy_in_cars_frame,
                                          step_size=global_constants.TRAJECTORY_ARCLEN_RESOLUTION,
                                          desired_curve_len=global_constants.REFERENCE_TRAJECTORY_LENGTH,
                                          preserve_step_size=False)

        reference_route_in_cars_frame_x_y_yaw = CartesianFrame.add_yaw_and_derivatives(
            reference_route_xy_in_cars_frame)[:, 0:3]

        return reference_route_x_y_z, reference_route_in_cars_frame_x_y_yaw

    @staticmethod
    def __generate_trajectory_specs(behavioral_state: BehavioralState, target_lane_latitude: float,
                                    safe_speed: float, reference_route: np.ndarray) -> TrajectoryParameters:
        """
        Generate trajectory specification (cost) for trajectory planner
        :param behavioral_state: processed behavioral state
        :param target_lane_latitude: road latitude of reference route in [m]
        :param safe_speed: safe speed in [m/s] (ACDA)
        :param reference_route: [nx3] numpy array of (x, y, z, yaw) states
        :return: Trajectory cost specifications [TrajectoryParameters]
        """

        lanes_num, road_width, _, _ = behavioral_state.map.get_road_details(behavioral_state.current_road_id)

        target_state_x_y_yaw = reference_route[-1, :]
        target_state_velocity = safe_speed
        target_state = np.array(
            [target_state_x_y_yaw[0], target_state_x_y_yaw[1], target_state_x_y_yaw[2], target_state_velocity])
        left_lane_offset = road_width - target_lane_latitude
        right_lane_offset = target_lane_latitude

        # TODO: assign proper cost parameters
        cost_params = TrajectoryCostParams(0, 0, 0, 0, 0, left_lane_offset,
                                           right_lane_offset, 0, 0, 0, 0, 0, 0, 0)
        trajectory_parameters = TrajectoryParameters(reference_route=reference_route,
                                                     target_state=target_state,
                                                     cost_params=cost_params,
                                                     strategy=TrajectoryPlanningStrategy.HIGHWAY)

        return trajectory_parameters
