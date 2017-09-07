from logging import Logger
import numpy as np

from decision_making.src import global_constants
from decision_making.src.global_constants import ROAD_SHOULDERS_WIDTH
from decision_making.src.messages.trajectory_parameters import TrajectoryCostParams, SigmoidFunctionParams
from decision_making.src.messages.trajectory_parameters import TrajectoryParams
from decision_making.src.messages.visualization.behavioral_visualization_message import BehavioralVisualizationMsg
from decision_making.src.planning.behavioral.behavioral_state import BehavioralState
from decision_making.src.planning.behavioral.constants import POLICY_ACTION_SPACE_ADDITIVE_LATERAL_OFFSETS_IN_LANES, \
    LATERAL_SAFETY_MARGIN_FROM_OBJECT
from decision_making.src.planning.behavioral.default_policy_config import DefaultPolicyConfig
from decision_making.src.planning.behavioral.policy import Policy, PolicyConfig
from decision_making.src.planning.behavioral.policy_features import DefaultPolicyFeatures
from decision_making.src.planning.trajectory.trajectory_planning_strategy import TrajectoryPlanningStrategy
from decision_making.src.planning.utils import geometry_utils
from decision_making.src.planning.utils.acda import AcdaApi
from decision_making.src.planning.utils.geometry_utils import CartesianFrame


class DefaultPolicy(Policy):
    """
    The policy chooses a single reference path from paths alongside the road with different lateral offset.
    The offset is selected according to a logical process that takes into account the distance from ego
    to the closest object on each path (free-space).
    The selected lateral offset then defines the reference route that is forwarded to the trajectory planner.
    """

    def __init__(self, logger: Logger, policy_config: DefaultPolicyConfig):
        super().__init__(logger=logger, policy_config=policy_config)

    def plan(self, behavioral_state: BehavioralState) -> (TrajectoryParams, BehavioralVisualizationMsg):
        """
        This policy first calls to __high_level_planning that returns a desired lateral offset for driving.
        On the basis of the desired lateral offset, the policy defines a target state and cost parameters
          that will be forwarded to the trajectory planner.
        :param behavioral_state:
        :return: trajectory parameters for trajectories evaluation, visualization object
        """
        if behavioral_state.ego_timestamp is None:
            # supposed to be prevented in the facade
            self.logger.warning("Invalid behavioral state: behavioral_state.ego_timestamp is None")
            return None, None

        # High-level planning
        target_path_offset, target_path_latitude = self.__high_level_planning(behavioral_state)

        # Calculate reference route for driving
        reference_route_x_y_z, reference_route_in_cars_frame_x_y = DefaultPolicy.__generate_reference_route(
            behavioral_state,
            target_path_offset)

        # Calculate safe speed according to ACDA
        acda_safe_speed = AcdaApi.compute_acda(objects_on_road=behavioral_state.dynamic_objects_on_road,
                                               ego_state=behavioral_state.ego_state,
                                               navigation_plan=behavioral_state.navigation_plan,
                                               map_api=behavioral_state.map,
                                               lookahead_path=reference_route_in_cars_frame_x_y)
        safe_speed = min(acda_safe_speed, global_constants.BEHAVIORAL_PLANNING_CONSTANT_DRIVE_VELOCITY)

        if safe_speed < 0:
            self.logger.warning("safe speed < 0")

        safe_speed = max(safe_speed, 0)

        # Generate specs for trajectory planner
        trajectory_parameters = \
            DefaultPolicy._generate_trajectory_specs(behavioral_state=behavioral_state,
                                                     safe_speed=safe_speed,
                                                     target_path_latitude=target_path_latitude,
                                                     reference_route=reference_route_in_cars_frame_x_y)

        visualization_message = BehavioralVisualizationMsg(reference_route=reference_route_x_y_z)
        return trajectory_parameters, visualization_message

    def __high_level_planning(self, behavioral_state: BehavioralState) -> (float, float):
        """
        Generates a high-level plan
        :param behavioral_state: processed behavioral state
        :return target latitude for driving in [lanes], target latitude for driving in [m]
        """

        num_lanes = behavioral_state.map.get_road(behavioral_state.ego_road_id).lanes_num
        lane_width = behavioral_state.map.get_road(behavioral_state.ego_road_id).lane_width
        # remain in right most lane
        # return lanes_in_current_road

        current_center_lane_offset = behavioral_state.ego_state.road_localization.lane_num + 0.5

        #Load policy parameters config

        # Creates a grid of latitude locations on road, which will be used to determine
        # the target latitude of the driving trajectory

        # generated_path_offsets_grid is a grid of optional lateral offsets in [lanes]
        generated_path_offsets_grid = \
            DefaultPolicy.__generate_latitudes_grid(num_of_lanes=num_lanes,
                                                    current_lane_latitude=current_center_lane_offset,
                                                    policy_config=self._policy_config)
        path_absolute_latitudes = lane_width * generated_path_offsets_grid

        # For each latitude, find closest blocking object on lane
        closest_blocking_object_on_path = \
            DefaultPolicyFeatures.get_closest_object_on_path(policy_config=self._policy_config,
                                                             behavioral_state=behavioral_state,
                                                             lat_options=path_absolute_latitudes)

        # Choose a proper action (latitude offset from current center lane)
        selected_action, selected_offset = DefaultPolicy.__select_latitude_from_grid(
            path_absolute_offsets=generated_path_offsets_grid, current_lane_offset=current_center_lane_offset,
            closest_object_in_lane=closest_blocking_object_on_path, policy_config=self._policy_config)
        selected_latitude = selected_offset * lane_width

        return selected_offset, selected_latitude

    @staticmethod
    def __select_latitude_from_grid(path_absolute_offsets: np.array, current_lane_offset: float,
                                    closest_object_in_lane: np.array, policy_config: PolicyConfig) -> (
            float, float):
        """
        Select the best lateral offset to be taken, according to policy
        :param path_absolute_offsets: grid of latitudes [lanes]
        :param current_lane_offset: current latitude [lanes]
        :param closest_object_in_lane: array of distance to closest object per lane [m]
        :param policy_config: policy parameters
        :return: bets action index, best lateral offset [lanes]
        """
        num_of_valid_latitude_options = len(path_absolute_offsets)

        current_center_lane_index_in_grid = \
            np.where(path_absolute_offsets == current_lane_offset)[0][0]
        # check which options are in the center of lane
        center_of_lane = np.isclose(np.mod(path_absolute_offsets - 0.5, 1.0),
                                    np.zeros(shape=[num_of_valid_latitude_options]))
        other_center_lane_indexes_in_grid = \
            np.where((path_absolute_offsets != current_lane_offset) & center_of_lane)[
                0]  # check if integer

        object_distance_in_current_lane = closest_object_in_lane[current_center_lane_index_in_grid]
        is_other_lanes_available = len(other_center_lane_indexes_in_grid) > 0

        # the best center of lane is where the blocking object is most far
        best_center_of_lane_index_in_grid = other_center_lane_indexes_in_grid[
            np.argmax(closest_object_in_lane[other_center_lane_indexes_in_grid])]
        best_center_of_lane_distance_from_object = closest_object_in_lane[
            best_center_of_lane_index_in_grid]

        # Prefer current center lane if nearest object is far enough
        selected_action = current_center_lane_index_in_grid

        # Choose other lane only if improvement is sufficient
        if is_other_lanes_available \
                and (object_distance_in_current_lane <
                         policy_config.prefer_other_lanes_where_blocking_object_distance_less_than) \
                and (best_center_of_lane_distance_from_object >
                             object_distance_in_current_lane +
                             policy_config.prefer_other_lanes_if_improvement_is_greater_than):
            selected_action = best_center_of_lane_index_in_grid

        # If blocking object is too close: choose any valid lateral offset
        if (object_distance_in_current_lane <
                policy_config.prefer_any_lane_center_if_blocking_object_distance_greater_than) \
                and (best_center_of_lane_distance_from_object <
                         policy_config.prefer_any_lane_center_if_blocking_object_distance_greater_than):
            selected_action = np.argmax(closest_object_in_lane)

        # return best lane
        selected_offset = path_absolute_offsets[selected_action]

        return selected_action, selected_offset

    @staticmethod
    def __generate_latitudes_grid(num_of_lanes: float, current_lane_latitude: float,
                                  policy_config: PolicyConfig) -> np.array:
        """
        This function creates a grid of latitude locations on road, which will be used as
        a discrete action space that determines the target latitude of the driving trajectory.
        :param num_of_lanes: number of lanes on road
        :param current_lane_latitude: current road localization
        :param policy_config: policy parameters
        :return:
        """
        latitude_offset_grid_relative_to_current_center_lane = np.array(
            POLICY_ACTION_SPACE_ADDITIVE_LATERAL_OFFSETS_IN_LANES)
        absolute_latitude_offset_grid_in_lanes = current_lane_latitude + \
                                                 latitude_offset_grid_relative_to_current_center_lane
        num_of_latitude_options = len(latitude_offset_grid_relative_to_current_center_lane)
        rightmost_edge_of_road = 0.0  # in lanes
        leftmost_edge_of_road = num_of_lanes  # in lanes

        # The actions is a grid of different lateral offsets that will be used
        # as reference route for the trajectory planner. the different options of actions
        # is stored in 'latitude_options_in_lanes'
        latitude_options_in_lanes = [absolute_latitude_offset_grid_in_lanes[ind] for ind in
                                     range(num_of_latitude_options)
                                     if ((absolute_latitude_offset_grid_in_lanes[ind] <
                                          leftmost_edge_of_road - policy_config.margin_from_road_edge)
                                         and (absolute_latitude_offset_grid_in_lanes[ind]
                                              > rightmost_edge_of_road + policy_config.margin_from_road_edge))]
        latitude_options_in_lanes = np.array(latitude_options_in_lanes)

        return latitude_options_in_lanes

    @staticmethod
    def __generate_reference_route(behavioral_state: BehavioralState, target_lane_latitude: float) -> (
            np.array, np.array):
        """
        :param behavioral_state: processed behavioral state
        :param target_lane_latitude: road latitude of reference route in [m]
        :return: [nx3] array of reference_route (x,y,z) [m,m,m] in world coordinates,
         [nx3] array of reference_route (x,y,yaw) [m,m,rad] in cars coordinates
        """
        lookahead_path = behavioral_state.map.get_uniform_path_lookahead(
            road_id=behavioral_state.ego_road_id,
            lat_shift=target_lane_latitude,
            starting_lon=behavioral_state.ego_state.road_localization.road_lon,
            lon_step=global_constants.TRAJECTORY_ARCLEN_RESOLUTION,
            steps_num=int(np.round(global_constants.REFERENCE_TRAJECTORY_LENGTH / global_constants.TRAJECTORY_ARCLEN_RESOLUTION)),
            navigation_plan=behavioral_state.navigation_plan)
        reference_route_xy = lookahead_path
        reference_route_len = reference_route_xy.shape[0]

        # Transform into car's frame
        reference_route_x_y_z = np.concatenate((reference_route_xy, np.zeros(shape=[reference_route_len, 1])),
                                               axis=1)
        reference_route_xyz_in_cars_frame = geometry_utils.CartesianFrame.convert_global_to_relative_frame(
            global_pos=reference_route_x_y_z, frame_position=behavioral_state.ego_position,
            frame_orientation=behavioral_state.ego_orientation)
        reference_route_xy_in_cars_frame = reference_route_xyz_in_cars_frame[:, 0:2]

        # interpolate and create uniformly spaced path
        reference_route_xy_in_cars_frame, _ = \
            CartesianFrame.resample_curve(curve=reference_route_xy_in_cars_frame,
                                          step_size=global_constants.TRAJECTORY_ARCLEN_RESOLUTION,
                                          desired_curve_len=global_constants.REFERENCE_TRAJECTORY_LENGTH,
                                          preserve_step_size=False)

        return reference_route_x_y_z, reference_route_xy_in_cars_frame

    @staticmethod
    def _generate_trajectory_specs(behavioral_state: BehavioralState, target_path_latitude: float,
                                   safe_speed: float, reference_route: np.ndarray) -> TrajectoryParams:
        """
        Generate trajectory specification (cost) for trajectory planner
        :param behavioral_state: processed behavioral state
        :param target_path_latitude: road latitude of reference route in [m]
        :param safe_speed: safe speed in [m/s] (ACDA)
        :param reference_route: [nx3] numpy array of (x, y, z, yaw) states
        :return: Trajectory cost specifications [TrajectoryParameters]
        """

        # Get road details
        lane_width = behavioral_state.map.get_road(behavioral_state.ego_road_id).lane_width
        road_width = behavioral_state.map.get_road(behavioral_state.ego_road_id).width

        # Create target state
        reference_route_x_y_yaw = CartesianFrame.add_yaw(reference_route)
        target_state_x_y_yaw = reference_route_x_y_yaw[-1, :]
        target_state_velocity = safe_speed
        target_state = np.array(
            [target_state_x_y_yaw[0], target_state_x_y_yaw[1], target_state_x_y_yaw[2], target_state_velocity])

        # TODO: assign proper cost parameters
        infinite_sigmoid_cost = 1000.0  # not a constant because it might be learned. TBD
        zero_sigmoid_cost = 0.0  # not a constant because it might be learned. TBD
        sigmoid_k_param = 20.0

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
        right_shoulder_cost = SigmoidFunctionParams(w=infinite_sigmoid_cost, k=sigmoid_k_param,
                                                    offset=right_shoulder_offset)  # Very high (inf) cost
        left_shoulder_cost = SigmoidFunctionParams(w=infinite_sigmoid_cost, k=sigmoid_k_param,
                                                   offset=left_shoulder_offset)  # Very high (inf) cost
        right_road_cost = SigmoidFunctionParams(w=infinite_sigmoid_cost, k=sigmoid_k_param,
                                                offset=right_road_offset)  # Very high (inf) cost
        left_road_cost = SigmoidFunctionParams(w=infinite_sigmoid_cost, k=sigmoid_k_param,
                                               offset=left_road_offset)  # Very high (inf) cost

        # Set objects parameters
        # dilate each object by cars length + safety margin
        objects_dilation_size = behavioral_state.ego_state.size.length + LATERAL_SAFETY_MARGIN_FROM_OBJECT
        objects_cost = SigmoidFunctionParams(w=infinite_sigmoid_cost, k=sigmoid_k_param,
                                             offset=objects_dilation_size)  # Very high (inf) cost

        distance_from_reference_route_sq_factor = 1.0
        # TODO: set velocity and acceleration limits properly
        velocity_limits = np.array([0.0, 100.0])  # [m/s]. not a constant because it might be learned. TBD
        acceleration_limits = np.array([-10.0, 10.0])  # [m/s^2]. not a constant because it might be learned. TBD
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

        trajectory_execution_time = global_constants.REFERENCE_TRAJECTORY_LENGTH / safe_speed
        trajectory_parameters = TrajectoryParams(reference_route=reference_route,
                                                 time=trajectory_execution_time,
                                                 target_state=target_state,
                                                 cost_params=cost_params,
                                                 strategy=TrajectoryPlanningStrategy.HIGHWAY)

        return trajectory_parameters

    @staticmethod
    def __visualize_high_level_policy(path_absolute_latitudes: np.array, behavioral_state: BehavioralState,
                                      selected_action: float) -> None:
        # TODO: implement visualization
        relevant_options_array = list()
        selected_option = list()
        for lat_option_in_meters in path_absolute_latitudes:
            # Generate lookahead path per each lateral option for debugging and visualization purposes
            lookahead_path = behavioral_state.map.get_path_lookahead(
                road_id=behavioral_state.ego_road_id,
                lon=behavioral_state.ego_state.road_localization.road_lon, lat=lat_option_in_meters,
                max_lookahead_distance=global_constants.BEHAVIORAL_PLANNING_LOOKAHEAD_DIST, direction=1)

            lookahead_path = lookahead_path.transpose()
            lookahead_path_len = lookahead_path.shape[0]
            reference_route_xyz = np.concatenate((lookahead_path, np.zeros(shape=[lookahead_path_len, 1])), axis=1)

            if lat_option_in_meters == path_absolute_latitudes[selected_action]:
                selected_option.append(reference_route_xyz)
            else:
                relevant_options_array.append(reference_route_xyz)
