from logging import Logger
from typing import Type, List, Dict, Tuple

import numpy as np

from decision_making.src import global_constants
from decision_making.src.exceptions import VehicleOutOfRoad, NoValidLanesFound, raises
from decision_making.src.global_constants import BEHAVIORAL_PLANNING_DEFAULT_SPEED_LIMIT
from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from decision_making.src.messages.trajectory_parameters import TrajectoryCostParams, SigmoidFunctionParams
from decision_making.src.messages.trajectory_parameters import TrajectoryParams
from decision_making.src.messages.visualization.behavioral_visualization_message import BehavioralVisualizationMsg
from decision_making.src.planning.behavioral.behavioral_state import BehavioralState, DynamicObjectOnRoad
from decision_making.src.planning.behavioral.constants import POLICY_ACTION_SPACE_ADDITIVE_LATERAL_OFFSETS_IN_LANES, \
    LATERAL_SAFETY_MARGIN_FROM_OBJECT, MAX_PLANNING_DISTANCE_FORWARD, \
    MAX_PLANNING_DISTANCE_BACKWARD, BEHAVIORAL_PLANNING_HORIZON, BEHAVIORAL_PLANNING_TIME_RESOLUTION, \
    BEHAVIORAL_PLANNING_TRAJECTORY_HORIZON
from decision_making.src.planning.behavioral.default_policy_config import DefaultPolicyConfig
from decision_making.src.planning.behavioral.policy import Policy, PolicyConfig
from decision_making.src.planning.trajectory.trajectory_planning_strategy import TrajectoryPlanningStrategy
from decision_making.src.planning.utils.acda import AcdaApi
from decision_making.src.planning.utils.acda_constants import TIME_GAP
from decision_making.src.prediction.predictor import Predictor
from decision_making.src.state.state import EgoState, State, DynamicObject
from mapping.src.model.constants import ROAD_SHOULDERS_WIDTH
from mapping.src.model.map_api import MapAPI
from mapping.src.transformations.geometry_utils import CartesianFrame

######################################################################################################
#################   Default Policy Behavioral State
######################################################################################################

SEMANTIC_GRID_FRONT, SEMANTIC_GRID_ASIDE, SEMANTIC_GRID_BEHIND = 1.0, 0.0, -1.0
GRID_MID = 10

# The margin that we take from the front/read of the vehicle to define the front/rear partitions
SEMANTIC_OCCUPANCY_GRID_PARTITIONS_MARGIN_FROM_EGO = 1.0


class RoadSemanticOccupancyGrid:
    """
    This class holds a semantic occupancy grid. We assume that the road is partitioned into semantic areas,
     and the class holds an occupancy grid that associates object to these areas.
    """

    def __init__(self, road_occupancy_grid: Dict[Tuple[float], List[DynamicObjectOnRoad]]):
        """
        :param road_occupancy_grid: A dictionary that maps a partition to a list of dynamic objects.
        Dictionary of Tuple( relative_lane, longitudinal_index ) -> List [ DynamicObjectOnRoad ]
         - relative_lane is a float that describes the lane number, relative to ego. For example: {-1.0, 0.0, 1.0}
         - longitudinal_index is a float that describes the longitudinal partition of the grid.
            For example, the grid can be partitioned in the following way:
            {-1.0: behind ego, 0.0: aside, 1.0: infront of ego}.

        """
        self.road_occupancy_grid = road_occupancy_grid


class SemanticActionGrid:
    """
    This class holds semantic actions (reference routes) that are associated to the road occupancy grid.
    The road is partitioned into semantic areas, each holds feasible actions that the ego vehicle can
    execute towards these cells.
    Dictionary of Tuple( relative_lane, longitudinal_index ) -> List [ trajectories ]
     - relative_lane, longitudinal_index as defined in RoadSemanticOccupancyGrid class
     - trajectory is a numpy array of floats, of size Nx4. Each row is (x, y, yaw, v)
    """

    def __init__(self, semantic_actions_grid: Dict[Tuple[float], List[np.ndarray]]):
        """
        :param road_occupancy_grid: A dictionary that maps a partition to a list of dynamic objects.
        """
        self.semantic_actions_grid = semantic_actions_grid


class DefaultBehavioralState(BehavioralState):
    def __init__(self, logger: Logger, map_api: MapAPI, navigation_plan: NavigationPlanMsg, ego_state: EgoState,
                 dynamic_objects_on_road: List[DynamicObjectOnRoad],
                 road_semantic_occupancy_grid: RoadSemanticOccupancyGrid=None) -> None:
        """
        Behavioral state generates and stores relevant state features that will be used for planning
        :param logger: logger
        :param map_api: map API
        :param navigation_plan: car's navigation plan
        :param ego_state: updated ego state
        :param road_semantic_occupancy_grid:
        """

        self.logger = logger

        # Navigation
        self.map = map_api
        self.navigation_plan = navigation_plan

        # Ego state features
        self.ego_timestamp = ego_state.timestamp
        self.ego_state = ego_state

        self.ego_yaw = ego_state.yaw
        self.ego_position = np.array([ego_state.x, ego_state.y, ego_state.z])
        self.ego_orientation = np.array(CartesianFrame.convert_yaw_to_quaternion(ego_state.yaw))
        self.ego_velocity = np.linalg.norm([ego_state.v_x, ego_state.v_y])
        self.ego_road_id = ego_state.road_localization.road_id
        self.ego_on_road = ego_state.road_localization.road_id is not None
        if not self.ego_on_road:
            self.logger.warning("Car is off road.")

        # Dynamic objects and their relative locations
        self.dynamic_objects_on_road = dynamic_objects_on_road

        # The semantic road occupancy grid
        self.road_semantic_occupancy_grid = road_semantic_occupancy_grid

    def update_behavioral_state(self, state: State, navigation_plan: NavigationPlanMsg):
        """
        This method updates the behavioral state according to the new world state and navigation plan.
         It fetches relevant features that will be used for the decision-making process.
        :param navigation_plan: new navigation plan of vehicle
        :param state: new world state
        :return: a new and updated BehavioralState
        """
        ego_state = state.ego_state

        # Filter static & dynamic objects that are relevant to car's navigation
        dynamic_objects_on_road = []
        for dynamic_obj in state.dynamic_objects:
            # Get object's relative road localization
            relative_road_localization = dynamic_obj.get_relative_road_localization(
                ego_road_localization=ego_state.road_localization, ego_nav_plan=navigation_plan,
                map_api=self.map, logger=self.logger)

            # filter objects with out of decision-making range
            if MAX_PLANNING_DISTANCE_FORWARD > \
                    relative_road_localization.rel_lon > \
                    MAX_PLANNING_DISTANCE_BACKWARD:
                dynamic_object_on_road = DynamicObjectOnRoad(dynamic_object_properties=dynamic_obj,
                                                             relative_road_localization=relative_road_localization)
                dynamic_objects_on_road.append(dynamic_object_on_road)

        road_semantic_occupancy_grid = DefaultBehavioralState.generate_semantic_occupancy_grid(ego_state,
                                                                                               dynamic_objects_on_road)

        return DefaultBehavioralState(logger=self.logger, map_api=self.map, navigation_plan=navigation_plan,
                                      ego_state=ego_state, dynamic_objects_on_road=dynamic_objects_on_road,
                                      road_semantic_occupancy_grid=road_semantic_occupancy_grid)

    @staticmethod
    def generate_semantic_occupancy_grid(ego_state: EgoState, dynamic_objects_on_road: List[
        DynamicObjectOnRoad]) -> RoadSemanticOccupancyGrid:
        """
        Occupy the occupancy grid.
        :param ego_state:
        :param dynamic_objects_on_road:
        :return: road semantic occupancy grid
        """

        ego_lane = ego_state.road_localization.lane_num

        # TODO - document assumptions
        semantic_occupancy_dict: Dict[Tuple[float], List[DynamicObjectOnRoad]] = dict()
        for dynamic_objects_on_road in dynamic_objects_on_road:
            # TODO - make sure ego localization is compatible. Consider creating a global constant representing the dist to front/back of ego.
            object_lon_dist = dynamic_objects_on_road.relative_road_localization.rel_lon
            object_dist_from_front = object_lon_dist - ego_state.size.length
            object_relative_lane = dynamic_objects_on_road.road_localization.lane_num - ego_lane

            # Determine cell index in occupancy grid
            if object_relative_lane == 0.0:
                # Object is on same lane as ego
                if object_dist_from_front > 0:
                    # Object in front of vehicle
                    occupancy_index = (object_relative_lane, SEMANTIC_GRID_FRONT)

                else:
                    # Object behind vehicle
                    occupancy_index = (object_relative_lane, SEMANTIC_GRID_BEHIND)

            elif object_relative_lane == 1.0 or object_relative_lane == -1.0:
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
            # TODO - keeping only a single dynamic object per cell. List is used for future dev.
            if occupancy_index not in semantic_occupancy_dict:
                # add to occupancy grid
                semantic_occupancy_dict[occupancy_index] = [dynamic_objects_on_road]
            else:
                object_in_grid_lon_dist = semantic_occupancy_dict[occupancy_index][
                    0].relative_road_localization.rel_lon

                if occupancy_index[1] == SEMANTIC_GRID_FRONT:
                    # take the object with least lon
                    if object_lon_dist < object_in_grid_lon_dist:
                        # replace object the the closer one
                        semantic_occupancy_dict[occupancy_index][0] = dynamic_objects_on_road
                else:
                    # Assumption - taking the object with the largest long even in the ASIDE cells
                    # take the object with largest lon
                    if object_lon_dist > object_in_grid_lon_dist:
                        # replace object the the closer one
                        semantic_occupancy_dict[occupancy_index][0] = dynamic_objects_on_road

        return RoadSemanticOccupancyGrid(semantic_occupancy_dict)


######################################################################################################
#################   Default Policy Features
######################################################################################################

class DefaultPolicyFeatures:
    @staticmethod
    def generate_object_following_trajectory(ego_state: EgoState, dynamic_object: DynamicObjectOnRoad,
                                             desired_speed: float, map_api: MapAPI,
                                             nav_plan: NavigationPlanMsg,
                                             predictor: Predictor, logger: Logger) -> List[
        np.ndarray]:
        """
        Uses prediction to generates semantic actions that are related to a certain dynamic object on road.
         e.g. a reference route that merges in front / behind a given vehicle
        :param desired_speed:
        :param logger:
        :param predictor:
        :param map_api:
        :param ego_state:
        :param dynamic_object:
        :param nav_plan:
        :return: list of numpy arrays (Nx4) describing the reference trajectories. each row is (x,y,yaw,v)
        """

        # Predict object trajectory
        prediction_timestamps = np.arange(0.0, BEHAVIORAL_PLANNING_HORIZON, BEHAVIORAL_PLANNING_TRAJECTORY_HORIZON)
        predicted_object_trajectory = predictor.predict_object_trajectories(dynamic_object=dynamic_object,
                                                                            prediction_timestamps=prediction_timestamps,
                                                                            nav_plan=nav_plan)

        # Get the relative position to ego at the objects' final state
        object_predicted_state = predictor.convert_predictions_to_dynamic_objects(dynamic_object=dynamic_object,
                                                                                  predictions=
                                                                                  predicted_object_trajectory[
                                                                                      -1], prediction_timestamps=
                                                                                  prediction_timestamps[-1])[0]
        object_final_state_relative_road_localization = object_predicted_state.get_relative_road_localization(
            ego_road_localization=ego_state.road_localization, ego_nav_plan=nav_plan, map_api=map_api, logger=logger)

        object_predicted_lat = object_final_state_relative_road_localization.rel_lat
        object_predicted_lon = object_final_state_relative_road_localization.rel_lon
        ego_predicted_lon = ego_state.road_longitudinal_speed * BEHAVIORAL_PLANNING_TRAJECTORY_HORIZON

        target_speed = np.min([desired_speed, object_predicted_state.road_longitudinal_speed])

        safe_distance = AcdaApi.calc_forward_sight_distance(static_objects=[object_predicted_state],
                                                            ego_state=ego_state)

        # Generate reference trajectory towards the rear of the objects final state
        if object_predicted_lon - ego_predicted_lon < safe_distance:
            pass




    @staticmethod
    def _generate_safe_brake_trajectory(ego_state: EgoState, predicted_object: DynamicObjectOnRoad):
        """
        Decelerate down to a safe speed with short lookahead distance (to ensure safety)
        :param ego_state:
        :param predicted_object:
        :return: list of numpy arrays (Nx4) describing the reference trajectories. each row is (x,y,yaw,v)
        """
        pass

    @staticmethod
    def _contract_following_distance(ego_state: EgoState, predicted_object: DynamicObjectOnRoad):
        """
        When following distance is greater than the target following distance, generate
        a long-term plan that will narrow the distance between ego and the object to be followed
        :param ego_state:
        :param predicted_object:
        :return: list of numpy arrays (Nx4) describing the reference trajectories. each row is (x,y,yaw,v)
        """
        pass

    @staticmethod
    def _enlarge_following_distance(ego_state: EgoState, predicted_object: DynamicObjectOnRoad):
        """
        When following distance is smaller than the target following distance (but still safe), generate
        a long-term plan that will enlarge the distance between ego and the object to be followed.
        Later, we can call the contract following distance
        :param ego_state:
        :param predicted_object:
        :return: list of numpy arrays (Nx4) describing the reference trajectories. each row is (x,y,yaw,v)
        """
        pass
        pass

    @staticmethod
    def generate_reference_trajectory_towards_location(ego_state: EgoState, target_road_offset: np.ndarray,
                                                       desired_speed: float, map_api: MapAPI,
                                                       nav_plan: NavigationPlanMsg,
                                                       predictor: Predictor) -> np.ndarray:
        """
        Generates trajectory towards a target location / state
        :param ego_state: ego initial state
        :param target_road_offset: offset in (lat, lon) in [m] in road coordinates
        :param nav_plan: ego navigation plan
        :param predictor: ego motion predictor
        :return: numpy array (Nx4) describing the reference trajectory. each row is (x,y,yaw,v)
        """

    @staticmethod
    def get_closest_object_on_path(policy_config: DefaultPolicyConfig, behavioral_state: DefaultBehavioralState,
                                   lat_options: np.array) -> np.array:
        """
        Gets the closest object on lane per each lateral offset in lat_options
        :param policy_config: policy parameters configuration
        :param behavioral_state:
        :param lat_options: grid of lateral offsets on road [m]
        :return: array with closest object per each lateral location options
        """

        # Fetch each latitude offset attributes (free-space, blocking objects, etc.)
        num_of_lat_options = len(lat_options)
        closest_blocking_object_per_option = np.inf * np.ones(shape=[num_of_lat_options, 1])

        # Assign object to optional lanes
        for blocking_object in behavioral_state.dynamic_objects_on_road:

            relative_lon = blocking_object.relative_road_localization.rel_lon
            if relative_lon < policy_config.assume_blocking_object_at_rear_if_distance_less_than:
                # If we passed an obstacle, treat it as at inf
                relative_lon = np.inf

            # get leftmost and rightmost edge of object
            object_leftmost_edge = blocking_object.road_localization.full_lat \
                                   + 0.5 * blocking_object.size.width
            object_leftmost_edge_dilated = object_leftmost_edge + (
                LATERAL_SAFETY_MARGIN_FROM_OBJECT + 0.5 * behavioral_state.ego_state.size.width)

            object_rightmost_edge = blocking_object.road_localization.full_lat \
                                    - 0.5 * blocking_object.size.width
            object_rightmost_edge_dilated = object_rightmost_edge - (
                LATERAL_SAFETY_MARGIN_FROM_OBJECT + 0.5 * behavioral_state.ego_state.size.width)

            # check which lateral offsets are affected
            affected_lanes = np.where((lat_options < object_leftmost_edge_dilated) & (
                lat_options > object_rightmost_edge_dilated))[0]

            # assign closest object to each lateral offset
            closest_blocking_object_per_option[affected_lanes] = np.minimum(
                closest_blocking_object_per_option[affected_lanes],
                relative_lon)

        return closest_blocking_object_per_option


#####################################################################################################
#################   Default Policy
######################################################################################################

class DefaultPolicy(Policy):
    """
    The policy chooses a single reference path from paths alongside the road with different lateral offset.
    The offset is selected according to a logical process that takes into account the distance from ego
    to the closest object on each path (free-space).
    The selected lateral offset then defines the reference route that is forwarded to the trajectory planner.
    """

    def __init__(self, logger: Logger, policy_config: DefaultPolicyConfig, behavioral_state: DefaultBehavioralState,
                 predictor: Type[Predictor], map_api: MapAPI):
        """
        see base class
        """
        super().__init__(logger=logger, policy_config=policy_config, behavioral_state=behavioral_state,
                         predictor=predictor, map_api=map_api)

    @raises(VehicleOutOfRoad, NoValidLanesFound)
    def plan(self, state: State, nav_plan: NavigationPlanMsg) -> (TrajectoryParams, BehavioralVisualizationMsg):
        """
        This policy first calls to __high_level_planning that returns a desired lateral offset for driving.
        On the basis of the desired lateral offset, the policy defines a target state and cost parameters
          that will be forwarded to the trajectory planner.
        :param nav_plan: ego navigation plan
        :param state: world state
        :return: trajectory parameters for trajectories evaluation, visualization object
        """

        self._behavioral_state = self._behavioral_state.update_behavioral_state(state, nav_plan)

        if self._behavioral_state.ego_timestamp is None:
            # supposed to be prevented in the facade
            self.logger.warning("Invalid behavioral state: behavioral_state.ego_timestamp is None")
            return None, None

        # High-level planning
        target_path_offset, target_path_latitude = self.__high_level_planning(self._behavioral_state)

        # Calculate reference route for driving
        reference_route_xy = DefaultPolicy.__generate_reference_route(
            self._behavioral_state,
            target_path_latitude)

        # Calculate safe speed according to ACDA
        acda_safe_speed = AcdaApi.compute_acda(objects_on_road=self._behavioral_state.dynamic_objects_on_road,
                                               ego_state=self._behavioral_state.ego_state,
                                               lookahead_path=reference_route_xy)
        safe_speed = min(acda_safe_speed, global_constants.BEHAVIORAL_PLANNING_DEFAULT_SPEED_LIMIT)

        if safe_speed < 0:
            self.logger.warning("safe speed < 0")

        safe_speed = max(safe_speed, 0)

        # Generate specs for trajectory planner
        trajectory_parameters = \
            DefaultPolicy._generate_trajectory_specs(behavioral_state=self._behavioral_state,
                                                     safe_speed=safe_speed,
                                                     target_path_latitude=target_path_latitude,
                                                     reference_route=reference_route_xy)

        self.logger.debug("Actual reference route[0] is {} and target_path_latitude is {}"
                          .format(reference_route_xy[0], target_path_latitude))

        visualization_message = BehavioralVisualizationMsg(reference_route=reference_route_xy)
        return trajectory_parameters, visualization_message

    def __generate_semantic_action_space(self, behavioral_state: DefaultBehavioralState) -> SemanticActionGrid:
        """
        Creates a grid of possible semantic actions that the agent can choose from.
        Actions are characterized by reference trajectory and organized in semantic road grid.
        :param behavioral_state:
        :return: list of numpy arrays (Nx4) describing trajectories. each row is (x,y,yaw,v).
         all options are organized in SemanticActionGrid
        """

        # Initialize an empty semantic action space structure
        semantic_actions_grid: Dict[Tuple(float), List(np.ndarray)] = dict()

        for grid_cell_key, dynamic_objects_in_cell in self._behavioral_state.road_semantic_occupancy_grid.road_occupancy_grid:

            # Soft traffic rules
            # TODO: define max speed induced by neighboring cars on the left to prevent bypassing from the right
            max_speed_induced_by_left_lanes = np.inf

            # Define desired speed
            desired_speed_in_cell = np.min(
                [BEHAVIORAL_PLANNING_DEFAULT_SPEED_LIMIT, max_speed_induced_by_left_lanes])

            if len(dynamic_objects_in_cell) > 0:
                # Create reference trajectories towards the front / back of objects at each cell
                semantic_actions_towards_object = DefaultPolicyFeatures.generate_object_following_trajectory(
                    ego_state=behavioral_state.ego_state,
                    dynamic_object=dynamic_objects_in_cell[0], desired_speed=desired_speed_in_cell,
                    map_api=self._map_api, nav_plan=behavioral_state.navigation_plan,
                    predictor=self._predictor, logger=self.logger)

                # Add actions to grid
                semantic_actions_grid[grid_cell_key] = semantic_actions_towards_object
            else:
                # Create actions towards empty road grid cells
                if grid_cell_key[1] == 1.0:
                    # handle forward cells
                    grid_cell_relative_lane = int(grid_cell_key[0])
                    grid_cell_lane = behavioral_state.ego_state.road_localization.lane_num + grid_cell_relative_lane
                    grid_cell_lane_lat = behavioral_state.map.get_center_lanes_latitudes[grid_cell_lane]

                    target_road_offset = np.array(
                        [grid_cell_lane_lat, MAX_PLANNING_DISTANCE_FORWARD])  # lat, lon offsets
                    semantic_action_towards_location = DefaultPolicyFeatures.generate_reference_trajectory_towards_location(
                        ego_state=behavioral_state.ego_state, target_road_offset=target_road_offset,
                        desired_speed=desired_speed_in_cell, map_api=self._map_api,
                        nav_plan=behavioral_state.navigation_plan, predictor=self._predictor)

                    # Add action to grid
                    semantic_actions_grid[grid_cell_key] = [semantic_action_towards_location]

        # TODO: filter relevant semantic actions: decide which are safe and beneficial.

        # Return the semantic action grid
        semantic_actions = SemanticActionGrid(semantic_actions_grid=semantic_actions_grid)
        return semantic_actions

    @raises(VehicleOutOfRoad, NoValidLanesFound)
    def __high_level_planning(self, behavioral_state: DefaultBehavioralState) -> (float, float):
        """
        Generates a high-level plan
        :param behavioral_state: processed behavioral state
        :return target latitude for driving in [lanes], target latitude for driving in [m]
        """

        lane_width = behavioral_state.map.get_road(behavioral_state.ego_road_id).lane_width
        # remain in right most lane
        # return lanes_in_current_road

        current_center_lane_offset = behavioral_state.ego_state.road_localization.lane_num + 0.5

        # Load policy parameters config

        # Creates a grid of latitude locations on road, which will be used to determine
        # the target latitude of the driving trajectory

        # generated_path_offsets_grid is a grid of optional lateral offsets in [lanes]
        path_absolute_latitudes = behavioral_state.map.get_center_lanes_latitudes(behavioral_state.ego_road_id)
        generated_path_offsets_grid = path_absolute_latitudes / lane_width

        # For each latitude, find closest blocking object on lane
        closest_blocking_object_on_path = \
            DefaultPolicyFeatures.get_closest_object_on_path(policy_config=self._policy_config,
                                                             behavioral_state=behavioral_state,
                                                             lat_options=path_absolute_latitudes)

        self.logger.debug("BEHAVIORAL: closest_blocking_object_on_path: {}".format(closest_blocking_object_on_path))
        # Choose a proper action (latitude offset from current center lane)
        selected_action, selected_offset = DefaultPolicy.__select_latitude_from_grid(
            path_absolute_offsets=generated_path_offsets_grid, current_lane_offset=current_center_lane_offset,
            closest_object_in_lane=closest_blocking_object_on_path, policy_config=self._policy_config)
        selected_latitude = selected_offset * lane_width
        self.logger.debug("BEHAVIORAL: selected_latitude: {}".format(selected_latitude))

        self.logger.debug("DefaultPolicy.__high_level_planning is considering latitudes: {} (lanes {}) - "
                          "latitude {} is chosen (lane {})".format(path_absolute_latitudes, generated_path_offsets_grid,
                                                                   selected_latitude, selected_offset))

        return selected_offset, selected_latitude

    @staticmethod
    @raises(VehicleOutOfRoad, NoValidLanesFound)
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
        try:
            num_of_valid_latitude_options = len(path_absolute_offsets)

            current_center_lane_index_in_grid = \
                np.where(path_absolute_offsets == current_lane_offset)[0][0]
            # check which options are in the center of lane
            center_of_lane = np.isclose(np.mod(path_absolute_offsets - 0.5, 1.0),
                                        np.zeros(shape=[num_of_valid_latitude_options]))
        except IndexError as ie:
            raise VehicleOutOfRoad("VehicleOutOfRoad in __select_latitude_from_grid with: path_absolute_offsets={}, "
                                   "current_lane_offset={}. {}".format(path_absolute_offsets, current_lane_offset, ie))
        try:
            other_center_lane_indexes_in_grid = \
                np.where((path_absolute_offsets != current_lane_offset) & center_of_lane)[0]  # check if integer
        except IndexError as ie:
            raise NoValidLanesFound("NoValidLanesFound in __select_latitude_from_grid with: path_absolute_offsets={},"
                                    "current_lane_offset={}, center_of_lane={}"
                                    .format(path_absolute_offsets, current_lane_offset, center_of_lane))

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
                and (best_center_of_lane_distance_from_object >
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
    def __generate_reference_route(behavioral_state: DefaultBehavioralState, target_lane_latitude: float) -> np.ndarray:
        """
        :param behavioral_state: processed behavioral state
        :param target_lane_latitude: road latitude of reference route in [m]
        :return: [nx3] array of reference_route (x,y,yaw) [m,m,rad] in global coordinates
        """
        lookahead_path = behavioral_state.map.get_uniform_path_lookahead(
            road_id=behavioral_state.ego_road_id,
            lat_shift=target_lane_latitude,
            starting_lon=behavioral_state.ego_state.road_localization.road_lon,
            lon_step=global_constants.TRAJECTORY_ARCLEN_RESOLUTION,
            steps_num=int(np.round(global_constants.REFERENCE_TRAJECTORY_LENGTH_EXTENDED /
                                   global_constants.TRAJECTORY_ARCLEN_RESOLUTION)),
            navigation_plan=behavioral_state.navigation_plan)
        reference_route_xy = lookahead_path

        # interpolate and create uniformly spaced path
        reference_route_xy_resampled, _ = \
            CartesianFrame.resample_curve(curve=reference_route_xy,
                                          step_size=global_constants.TRAJECTORY_ARCLEN_RESOLUTION,
                                          desired_curve_len=global_constants.REFERENCE_TRAJECTORY_LENGTH,
                                          preserve_step_size=False)

        return reference_route_xy_resampled

    @staticmethod
    def _generate_trajectory_specs(behavioral_state: DefaultBehavioralState, target_path_latitude: float,
                                   safe_speed: float, reference_route: np.ndarray) -> TrajectoryParams:
        """
        Generate trajectory specification (cost) for trajectory planner
        :param behavioral_state: processed behavioral state
        :param target_path_latitude: road latitude of reference route in [m] from right-side of road
        :param safe_speed: safe speed in [m/s] (ACDA)
        :param reference_route: [nx3] numpy array of (x, y, z, yaw) states
        :return: Trajectory cost specifications [TrajectoryParameters]
        """

        # Get road details
        lane_width = behavioral_state.map.get_road(behavioral_state.ego_road_id).lane_width
        road_width = behavioral_state.map.get_road(behavioral_state.ego_road_id).road_width

        # Create target state
        reference_route_x_y_yaw = CartesianFrame.add_yaw(reference_route)
        target_state_x_y_yaw = reference_route_x_y_yaw[-1, :]
        target_state_velocity = safe_speed
        target_state = np.array(
            [target_state_x_y_yaw[0], target_state_x_y_yaw[1], target_state_x_y_yaw[2], target_state_velocity])

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

        current_speed = np.linalg.norm([behavioral_state.ego_state.v_x, behavioral_state.ego_state.v_y])
        euclidean_dist_to_target = np.linalg.norm(target_state[:2] - np.array([behavioral_state.ego_state.x,
                                                                               behavioral_state.ego_state.y]))
        trajectory_execution_time = 2 * euclidean_dist_to_target / (safe_speed + current_speed)

        trajectory_parameters = TrajectoryParams(reference_route=reference_route,
                                                 time=trajectory_execution_time,
                                                 target_state=target_state,
                                                 cost_params=cost_params,
                                                 strategy=TrajectoryPlanningStrategy.HIGHWAY)

        return trajectory_parameters

    @staticmethod
    def __visualize_high_level_policy(path_absolute_latitudes: np.array, behavioral_state: DefaultBehavioralState,
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
