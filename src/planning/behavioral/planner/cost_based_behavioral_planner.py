from abc import abstractmethod, ABCMeta
from logging import Logger
from typing import Optional

import numpy as np
import six

from decision_making.src.global_constants import PREDICTION_LOOKAHEAD_COMPENSATION_RATIO, TRAJECTORY_ARCLEN_RESOLUTION, \
    SHOULDER_SIGMOID_OFFSET, DEVIATION_FROM_LANE_COST, LANE_SIGMOID_K_PARAM, SHOULDER_SIGMOID_K_PARAM, \
    DEVIATION_TO_SHOULDER_COST, DEVIATION_FROM_ROAD_COST, ROAD_SIGMOID_K_PARAM, OBSTACLE_SIGMOID_COST, \
    OBSTACLE_SIGMOID_K_PARAM, DEVIATION_FROM_GOAL_COST, GOAL_SIGMOID_K_PARAM, GOAL_SIGMOID_OFFSET, \
    DEVIATION_FROM_GOAL_LAT_LON_RATIO, LON_JERK_COST_WEIGHT, LAT_JERK_COST_WEIGHT, VELOCITY_LIMITS, LON_ACC_LIMITS, LAT_ACC_LIMITS
from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from decision_making.src.messages.trajectory_parameters import TrajectoryParams, TrajectoryCostParams, \
    SigmoidFunctionParams
from decision_making.src.planning.behavioral.action_space.action_space import ActionSpace
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import ActionSpec, ActionRecipe
from decision_making.src.planning.behavioral.evaluators.action_evaluator import ActionSpecEvaluator, \
    ActionRecipeEvaluator
from decision_making.src.planning.behavioral.evaluators.value_approximator import ValueApproximator
from decision_making.src.planning.behavioral.filtering.action_spec_filtering import ActionSpecFiltering
from decision_making.src.planning.behavioral.semantic_actions_utils import SemanticActionsUtils
from decision_making.src.planning.trajectory.trajectory_planner import SamplableTrajectory
from decision_making.src.planning.trajectory.trajectory_planning_strategy import TrajectoryPlanningStrategy
from decision_making.src.planning.types import FS_DA, FS_SA, FS_SX, FS_DX
from decision_making.src.planning.trajectory.werling_planner import SamplableWerlingTrajectory
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame
from decision_making.src.planning.utils.map_utils import MapUtils
from decision_making.src.planning.utils.optimal_control.poly1d import QuinticPoly1D
from decision_making.src.prediction.predictor import Predictor
from decision_making.src.state.state import State, ObjectSize, NewEgoState
from mapping.src.model.constants import ROAD_SHOULDERS_WIDTH
from mapping.src.model.lane import Lane
from mapping.src.service.map_service import MapService


@six.add_metaclass(ABCMeta)
class CostBasedBehavioralPlanner:
    def __init__(self, action_space: ActionSpace, recipe_evaluator: Optional[ActionRecipeEvaluator],
                 action_spec_evaluator: Optional[ActionSpecEvaluator],
                 action_spec_validator: Optional[ActionSpecFiltering],
                 value_approximator: ValueApproximator, predictor: Predictor, logger: Logger):
        self.action_space = action_space
        self.recipe_evaluator = recipe_evaluator
        self.action_spec_evaluator = action_spec_evaluator
        self.action_spec_validator = action_spec_validator or ActionSpecFiltering()
        self.value_approximator = value_approximator
        self.predictor = predictor
        self.logger = logger

        self._last_action: Optional[ActionRecipe] = None
        self._last_action_spec: Optional[ActionSpec] = None

    @abstractmethod
    def plan(self, state: State, nav_plan: NavigationPlanMsg):
        """
        Given current state and navigation plan, plans the next semantic action to be carried away. This method makes
        use of Planner components such as Evaluator,Validator and Predictor for enumerating, specifying
        and evaluating actions. Its output will be further handled and used to create a trajectory in Trajectory Planner
        and has the form of TrajectoryParams, which includes the reference route, target time, target state to be in,
        cost params and strategy.
        :param state:
        :param nav_plan:
        :return: a tuple: (TrajectoryParams for TP,BehavioralVisualizationMsg for e.g. VizTool)
        """
        pass

    @staticmethod
    def _generate_trajectory_specs(behavioral_state: BehavioralGridState,
                                   action_spec: ActionSpec,
                                   navigation_plan: NavigationPlanMsg) -> TrajectoryParams:
        """
        Generate trajectory specification for trajectory planner given a SemanticActionSpec. This also
        generates the reference route that will be provided to the trajectory planner.
         Given the target longitude and latitude, we create a reference route in global coordinates, where:
         latitude is constant and equal to the target latitude;
         longitude starts from ego current longitude, and end in the target longitude.
        :param behavioral_state: processed behavioral state
        :param navigation_plan: navigation plan of the rest of the roads to be followed (used to create a ref. route)
        :return: Trajectory cost specifications [TrajectoryParameters]
        """
        map_api = MapService.get_instance()
        ego: NewEgoState = behavioral_state.ego_state

        # Add a margin to the lookahead path of dynamic objects to avoid extrapolation
        # caused by the curve linearization approximation in the resampling process
        # The compensation here is multiplicative because of the different curve-fittings we use:
        # in BP we use piecewise-linear and in TP we use cubic-fit.
        # Due to that, a point's longitude-value will be different between the 2 curves.
        # This error is accumulated depending on the actual length of the curvature -
        # when it is long, the error will potentially be big.
        lookahead_distance = action_spec.s * PREDICTION_LOOKAHEAD_COMPENSATION_RATIO

        ego_road = map_api.get_road(ego.map_state.road_id)
        # TODO: here we assume ego and the goal share the same road. Use nav_plan instead
        goal_segment = ego_road.get_segment_by_lon(action_spec.s)
        goal_lane = goal_segment.get_lane_by_lat(action_spec.s - goal_segment.s_start, action_spec.d)
        goal_lane_frame = goal_lane.get_as_frame()._center_frame

        # TODO: remove it when Frenet frame will be transferred to TP
        longitudes = np.arange(0, lookahead_distance, TRAJECTORY_ARCLEN_RESOLUTION)
        lookahead_fpoints = np.c_[longitudes, np.repeat(0, len(longitudes))]  # center-lane points in goal_lane frenet
        center_lane_reference_route = goal_lane_frame.fpoints_to_cpoints(lookahead_fpoints)

        # Convert goal state to Cartesian state. Use ego lane frenet-frame, like in specify
        road_id = ego.map_state.road_id
        road_lane_latitudes = MapService.get_instance().get_center_lanes_latitudes(road_id)
        ego_lane_num = MapService.get_instance().get_lane(ego.map_state.lane_id).ordinal
        ego_lane_frame = map_api.get_lane(ego.map_state.lane_id).get_as_frame()._center_frame
        goal_cstate = ego_lane_frame.fstate_to_cstate(np.array([action_spec.s, action_spec.v, 0,
                                                      action_spec.d - road_lane_latitudes[ego_lane_num], 0, 0]))

        # this assumes the target falls on the reference route
        cost_params = CostBasedBehavioralPlanner._generate_cost_params(ego.size, road_id, action_spec.d, action_spec.s)

        trajectory_parameters = TrajectoryParams(reference_route=center_lane_reference_route,
                                                 time=action_spec.t + ego.timestamp_in_sec,
                                                 target_state=goal_cstate,
                                                 cost_params=cost_params,
                                                 strategy=TrajectoryPlanningStrategy.HIGHWAY)
        return trajectory_parameters

    @staticmethod
    def generate_baseline_trajectory(ego: NewEgoState, action_spec: ActionSpec) -> SamplableTrajectory:
        """
        Creates a SamplableTrajectory as a reference trajectory for a given ActionSpec, assuming T_d=T_s
        :param ego: ego object
        :param action_spec: action specification that contains all relevant info about the action's terminal state
        :return: a SamplableWerlingTrajectory object
        """
        # Note: We create the samplable trajectory as a reference trajectory of the current action.from
        # We assume correctness only of the longitudinal axis, and set T_d to be equal to T_s.
        road_rhs_frenet = MapUtils.get_road_rhs_frenet(ego)

        # project ego vehicle onto the road
        ego_cstate = MapUtils.convert_map_to_cartesian_state(ego.map_state.lane_state)
        ego_init_fstate = road_rhs_frenet.cstate_to_fstate(ego_cstate)

        target_fstate = np.array([action_spec.s, action_spec.v, 0, action_spec.d, 0, 0])

        A_inv = np.linalg.inv(QuinticPoly1D.time_constraints_matrix(action_spec.t))

        constraints_s = np.concatenate((ego_init_fstate[FS_SX:(FS_SA + 1)], target_fstate[FS_SX:(FS_SA + 1)]))
        constraints_d = np.concatenate((ego_init_fstate[FS_DX:(FS_DA + 1)], target_fstate[FS_DX:(FS_DA + 1)]))

        poly_coefs_s = QuinticPoly1D.solve(A_inv, constraints_s[np.newaxis, :])[0]
        poly_coefs_d = QuinticPoly1D.solve(A_inv, constraints_d[np.newaxis, :])[0]

        return SamplableWerlingTrajectory(timestamp_in_sec=ego.timestamp_in_sec,
                                          T_s=action_spec.t,
                                          T_d=action_spec.t,
                                          frenet_frame=road_rhs_frenet,
                                          poly_s_coefs=poly_coefs_s,
                                          poly_d_coefs=poly_coefs_d)

    @staticmethod
    def _generate_cost_params(ego_size: ObjectSize, ego_road_id: int, reference_route_lat: float,
                              goal_road_lon: float) -> TrajectoryCostParams:

        map_api = MapService.get_instance()

        # TODO: here we assume ego and the goal are on the same road. Use nav_plan instead
        ego_road = map_api.get_road(ego_road_id)
        goal_segment = ego_road.get_segment_by_lon(goal_road_lon)
        segment_lon = goal_road_lon - goal_segment.s_start

        # TODO: here we assume a constant lane width along the trajectory
        _, lane_right_bound, lane_left_bound = goal_segment.find_lane_lateral_bounds(segment_lon, reference_route_lat)
        right_lane_offset = reference_route_lat - lane_right_bound - ego_size.width / 2
        left_lane_offset = lane_left_bound - reference_route_lat - ego_size.width / 2

        road_width = ego_road.get_width(goal_road_lon)
        # as stated above, for shoulders
        right_shoulder_offset = reference_route_lat - ego_size.width / 2 + SHOULDER_SIGMOID_OFFSET
        # as stated above, for shoulders
        left_shoulder_offset = road_width - reference_route_lat - ego_size.width / 2 + SHOULDER_SIGMOID_OFFSET

        # as stated above, for whole road including shoulders
        right_road_offset = reference_route_lat - ego_size.width / 2 + ROAD_SHOULDERS_WIDTH
        # as stated above, for whole road including shoulders
        left_road_offset = road_width - reference_route_lat - ego_size.width / 2 + ROAD_SHOULDERS_WIDTH

        # Set road-structure-based cost parameters
        right_lane_cost = SigmoidFunctionParams(w=DEVIATION_FROM_LANE_COST, k=LANE_SIGMOID_K_PARAM,
                                                offset=right_lane_offset)  # Zero cost
        left_lane_cost = SigmoidFunctionParams(w=DEVIATION_FROM_LANE_COST, k=LANE_SIGMOID_K_PARAM,
                                               offset=left_lane_offset)  # Zero cost
        right_shoulder_cost = SigmoidFunctionParams(w=DEVIATION_TO_SHOULDER_COST, k=SHOULDER_SIGMOID_K_PARAM,
                                                    offset=right_shoulder_offset)  # Very high cost
        left_shoulder_cost = SigmoidFunctionParams(w=DEVIATION_TO_SHOULDER_COST, k=SHOULDER_SIGMOID_K_PARAM,
                                                   offset=left_shoulder_offset)  # Very high cost
        right_road_cost = SigmoidFunctionParams(w=DEVIATION_FROM_ROAD_COST, k=ROAD_SIGMOID_K_PARAM,
                                                offset=right_road_offset)  # Very high cost
        left_road_cost = SigmoidFunctionParams(w=DEVIATION_FROM_ROAD_COST, k=ROAD_SIGMOID_K_PARAM,
                                               offset=left_road_offset)  # Very high cost

        # Set objects parameters
        # dilate each object by ego length + safety margin
        objects_dilation_length = SemanticActionsUtils.get_ego_lon_margin(ego_size)
        objects_dilation_width = SemanticActionsUtils.get_ego_lat_margin(ego_size)
        objects_cost_x = SigmoidFunctionParams(w=OBSTACLE_SIGMOID_COST, k=OBSTACLE_SIGMOID_K_PARAM,
                                               offset=objects_dilation_length)  # Very high (inf) cost
        objects_cost_y = SigmoidFunctionParams(w=OBSTACLE_SIGMOID_COST, k=OBSTACLE_SIGMOID_K_PARAM,
                                               offset=objects_dilation_width)  # Very high (inf) cost
        dist_from_goal_cost = SigmoidFunctionParams(w=DEVIATION_FROM_GOAL_COST, k=GOAL_SIGMOID_K_PARAM,
                                                    offset=GOAL_SIGMOID_OFFSET)
        dist_from_goal_lat_factor = DEVIATION_FROM_GOAL_LAT_LON_RATIO

        cost_params = TrajectoryCostParams(obstacle_cost_x=objects_cost_x,
                                           obstacle_cost_y=objects_cost_y,
                                           left_lane_cost=left_lane_cost,
                                           right_lane_cost=right_lane_cost,
                                           left_shoulder_cost=left_shoulder_cost,
                                           right_shoulder_cost=right_shoulder_cost,
                                           left_road_cost=left_road_cost,
                                           right_road_cost=right_road_cost,
                                           dist_from_goal_cost=dist_from_goal_cost,
                                           dist_from_goal_lat_factor=dist_from_goal_lat_factor,
                                           lon_jerk_cost=LON_JERK_COST_WEIGHT,
                                           lat_jerk_cost=LAT_JERK_COST_WEIGHT,
                                           velocity_limits=VELOCITY_LIMITS,
                                           lon_acceleration_limits=LON_ACC_LIMITS,
                                           lat_acceleration_limits=LAT_ACC_LIMITS)

        return cost_params
