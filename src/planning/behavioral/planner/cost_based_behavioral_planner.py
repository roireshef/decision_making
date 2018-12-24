import numpy as np
import six
from abc import abstractmethod, ABCMeta
from logging import Logger
from typing import Optional, List

import rte.python.profiler as prof
from decision_making.src.global_constants import PREDICTION_LOOKAHEAD_COMPENSATION_RATIO, TRAJECTORY_ARCLEN_RESOLUTION, \
    SHOULDER_SIGMOID_OFFSET, DEVIATION_FROM_LANE_COST, LANE_SIGMOID_K_PARAM, SHOULDER_SIGMOID_K_PARAM, \
    DEVIATION_TO_SHOULDER_COST, DEVIATION_FROM_ROAD_COST, ROAD_SIGMOID_K_PARAM, OBSTACLE_SIGMOID_COST, \
    OBSTACLE_SIGMOID_K_PARAM, DEVIATION_FROM_GOAL_COST, GOAL_SIGMOID_K_PARAM, GOAL_SIGMOID_OFFSET, \
    DEVIATION_FROM_GOAL_LAT_LON_RATIO, LON_JERK_COST_WEIGHT, LAT_JERK_COST_WEIGHT, VELOCITY_LIMITS, LON_ACC_LIMITS, \
    LAT_ACC_LIMITS, LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT, LATERAL_SAFETY_MARGIN_FROM_OBJECT, REFERENCE_ROUTE_MARGINS
from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from decision_making.src.messages.trajectory_parameters import TrajectoryParams, TrajectoryCostParams, \
    SigmoidFunctionParams
from decision_making.src.planning.behavioral.action_space.action_space import ActionSpace
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import ActionSpec, ActionRecipe, RelativeLane
from decision_making.src.planning.behavioral.evaluators.action_evaluator import ActionSpecEvaluator, \
    ActionRecipeEvaluator
from decision_making.src.planning.behavioral.evaluators.value_approximator import ValueApproximator
from decision_making.src.planning.behavioral.filtering.action_spec_filtering import ActionSpecFiltering
from decision_making.src.planning.trajectory.samplable_trajectory import SamplableTrajectory
from decision_making.src.planning.trajectory.samplable_werling_trajectory import SamplableWerlingTrajectory
from decision_making.src.planning.trajectory.trajectory_planning_strategy import TrajectoryPlanningStrategy
from decision_making.src.planning.types import FS_DA, FS_SA, FS_SX, FS_DX
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame
from decision_making.src.planning.utils.generalized_frenet_serret_frame import GeneralizedFrenetSerretFrame, \
    FrenetSubSegment
from decision_making.src.planning.utils.optimal_control.poly1d import QuinticPoly1D
from decision_making.src.prediction.ego_aware_prediction.ego_aware_predictor import EgoAwarePredictor
from decision_making.src.state.map_state import MapState
from decision_making.src.state.state import State, ObjectSize, EgoState
from decision_making.src.utils.map_utils import MapUtils
from mapping.src.exceptions import UpstreamLaneNotFound
from mapping.src.model.constants import ROAD_SHOULDERS_WIDTH, LARGE_DISTANCE_FROM_SHOULDER


@six.add_metaclass(ABCMeta)
class CostBasedBehavioralPlanner:
    def __init__(self, action_space: ActionSpace, recipe_evaluator: Optional[ActionRecipeEvaluator],
                 action_spec_evaluator: Optional[ActionSpecEvaluator],
                 action_spec_validator: Optional[ActionSpecFiltering],
                 value_approximator: ValueApproximator, predictor: EgoAwarePredictor, logger: Logger):
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
    def choose_action(self, state: State, behavioral_state: BehavioralGridState, action_recipes: List[ActionRecipe],
                      recipes_mask: List[bool]):
        """
        upon receiving an input state, return an action specification and its respective index in the given list of
        action recipes.
        :param recipes_mask: A list of boolean values, which are True if respective action recipe in
        input argument action_recipes is valid, else False.
        :param state: the current world state
        :param behavioral_state: processed behavioral state
        :param action_recipes: a list of enumerated semantic actions [ActionRecipe].
        :return: a tuple of the selected action index and selected action spec itself (int, ActionSpec).
        """
        pass

    @abstractmethod
    def plan(self, state: State, nav_plan: NavigationPlanMsg):
        """
        Given current state and navigation plan, plans the next semantic action to be carried away. This method makes
        use of Planner components such as Evaluator,Validator and Predictor for enumerating, specifying
        and evaluating actions. Its output will be further handled and used to create a trajectory in Trajectory Planner
        and has the form of TrajectoryParams, which includes the reference route, target time, target state to be in,
        cost params and strategy.
        :param state: the current world state
        :param nav_plan:
        :return: a tuple: (TrajectoryParams for TP,BehavioralVisualizationMsg for e.g. VizTool)
        """
        pass

    @prof.ProfileFunction()
    def _generate_terminal_states(self, state: State, action_specs: List[ActionSpec], mask: np.ndarray) -> \
            [BehavioralGridState]:
        """
        Given current state and action specifications, generate a corresponding list of future states using the
        predictor. Uses mask over list of action specifications to avoid unnecessary computation
        :param state: the current world state
        :param action_specs: list of action specifications
        :param mask: 1D mask vector (boolean) for filtering valid action specifications
        :return: a list of terminal states
        """
        # TODO: implement after M0
        return [None] * len(action_specs)

    @staticmethod
    @prof.ProfileFunction()
    def _generate_trajectory_specs(behavioral_state: BehavioralGridState, action_spec: ActionSpec,
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
        ego = behavioral_state.ego_state
        # calculate adjacent lane_ids
        relative_lane_ids = MapUtils.get_relative_lane_ids(ego.map_state.lane_id)
        # calculate adjacent lane_id matching to the action_spec
        spec_lane_id = relative_lane_ids[action_spec.relative_lane]

        # calculate projected ego frenet state
        projected_fstates = BehavioralGridState.project_ego_on_adjacent_lanes(ego)
        projected_ego_fstate = projected_fstates[action_spec.relative_lane]  # ego projected on the relative lane

        # goal frenet state relative to spec_lane_id
        goal_fstate = np.array([action_spec.s, action_spec.v, 0, action_spec.d, 0, 0])

        # set the reference route to start with a margin before the current longitudinal position of the vehicle
        suggested_ref_route_start = projected_ego_fstate[FS_SX] - REFERENCE_ROUTE_MARGINS

        # TODO: remove this hack when using a real map from SP
        # if there is no long enough road behind ego, set ref_route_start = 0
        ref_route_start = suggested_ref_route_start \
            if suggested_ref_route_start >= 0 or MapUtils.does_map_exist_backward(spec_lane_id, -suggested_ref_route_start) \
            else 0

        # calculate reference route length
        forward_lookahead = action_spec.s - ref_route_start + REFERENCE_ROUTE_MARGINS
        ref_route_length = forward_lookahead * PREDICTION_LOOKAHEAD_COMPENSATION_RATIO

        # create Generalized Frenet Frame (GFF) for TP, which is a part of TrajectoryParams
        # TODO: remove it and use GFFs from BehavioralGridState
        action_lane_gff = MapUtils.get_lookahead_frenet_frame(
            lane_id=spec_lane_id, starting_lon=ref_route_start, lookahead_dist=ref_route_length,
            navigation_plan=navigation_plan)

        # create cost params, which are part of TrajectoryParams
        cost_params = CostBasedBehavioralPlanner._generate_cost_params(
            map_state=MapState(goal_fstate, spec_lane_id),
            ego_size=ego.size
        )

        # Calculate cartesian coordinates of action_spec's target (according to target-lane frenet_frame)
        # TODO: remove it, when TP will obtain frenet frame
        goal_cstate = MapUtils.get_lane_frenet_frame(spec_lane_id).fstate_to_cstate(goal_fstate)

        # create TrajectoryParams for TP
        trajectory_parameters = TrajectoryParams(reference_route=action_lane_gff,
                                                 time=action_spec.t + ego.timestamp_in_sec,
                                                 target_state=goal_cstate,
                                                 cost_params=cost_params,
                                                 strategy=TrajectoryPlanningStrategy.HIGHWAY,
                                                 bp_time=ego.timestamp)

        return trajectory_parameters

    @staticmethod
    @prof.ProfileFunction()
    def generate_baseline_trajectory(ego: EgoState, action_spec: ActionSpec) -> SamplableTrajectory:
        """
        Creates a SamplableTrajectory as a reference trajectory for a given ActionSpec, assuming T_d=T_s
        :param ego: ego object
        :param action_spec: action specification that contains all relevant info about the action's terminal state
        :return: a SamplableWerlingTrajectory object
        """
        # Note: We create the samplable trajectory as a reference trajectory of the current action.from
        # We assume correctness only of the longitudinal axis, and set T_d to be equal to T_s.

        # project ego on target lane frenet_frame
        projected_fstates = BehavioralGridState.project_ego_on_adjacent_lanes(ego)
        projected_ego_fstate = projected_fstates[action_spec.relative_lane]  # ego projected on the relative lane

        relative_lane_ids = MapUtils.get_relative_lane_ids(ego.map_state.lane_id)
        spec_lane_id = relative_lane_ids[action_spec.relative_lane]

        target_fstate = np.array([action_spec.s, action_spec.v, 0, action_spec.d, 0, 0])

        A_inv = np.linalg.inv(QuinticPoly1D.time_constraints_matrix(action_spec.t))

        constraints_s = np.concatenate((projected_ego_fstate[FS_SX:(FS_SA + 1)], target_fstate[FS_SX:(FS_SA + 1)]))
        constraints_d = np.concatenate((projected_ego_fstate[FS_DX:(FS_DA + 1)], target_fstate[FS_DX:(FS_DA + 1)]))

        poly_coefs_s = QuinticPoly1D.solve(A_inv, constraints_s[np.newaxis, :])[0]
        poly_coefs_d = QuinticPoly1D.solve(A_inv, constraints_d[np.newaxis, :])[0]

        return SamplableWerlingTrajectory(timestamp_in_sec=ego.timestamp_in_sec,
                                          T_s=action_spec.t,
                                          T_d=action_spec.t,
                                          frenet_frame=MapUtils.get_lane_frenet_frame(spec_lane_id),
                                          poly_s_coefs=poly_coefs_s,
                                          poly_d_coefs=poly_coefs_d)

    @staticmethod
    def _generate_cost_params(map_state: MapState, ego_size: ObjectSize) -> TrajectoryCostParams:
        """
        Generate cost specification for trajectory planner
        :param map_state: MapState of the goal
        :param ego_size: ego size used to extract margins (for dilation of other objects on road)
        :return: a TrajectoryCostParams instance that encodes all parameters for TP cost computation.
        """
        relative_lanes = MapUtils.get_relative_lane_ids(map_state.lane_id)
        is_rightmost_lane = relative_lanes[RelativeLane.RIGHT_LANE] is None
        is_leftmost_lane = relative_lanes[RelativeLane.LEFT_LANE] is None

        # TODO: here we assume a constant lane width from the current state to the goal
        dist_from_right_lane_border, dist_from_left_lane_border = \
            MapUtils.get_dist_to_lane_borders(map_state.lane_id, map_state.lane_fstate[FS_SX])

        dist_from_right_road_border = dist_from_right_lane_border if is_rightmost_lane else LARGE_DISTANCE_FROM_SHOULDER
        dist_from_left_road_border = dist_from_left_lane_border if is_leftmost_lane else LARGE_DISTANCE_FROM_SHOULDER

        # lateral distance in [m] from ref. path to rightmost edge of lane
        right_lane_offset = dist_from_right_lane_border - ego_size.width / 2
        # lateral distance in [m] from ref. path to leftmost edge of lane
        left_lane_offset = dist_from_left_lane_border - ego_size.width / 2
        # as stated above, for shoulders
        right_shoulder_offset = dist_from_right_road_border - ego_size.width / 2 + SHOULDER_SIGMOID_OFFSET
        # as stated above, for shoulders
        left_shoulder_offset = dist_from_left_road_border - ego_size.width / 2 + SHOULDER_SIGMOID_OFFSET
        # as stated above, for whole road including shoulders
        right_road_offset = dist_from_right_road_border - ego_size.width / 2 + ROAD_SHOULDERS_WIDTH
        # as stated above, for whole road including shoulders
        left_road_offset = dist_from_left_road_border - ego_size.width / 2 + ROAD_SHOULDERS_WIDTH

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
        objects_cost_x = SigmoidFunctionParams(w=OBSTACLE_SIGMOID_COST, k=OBSTACLE_SIGMOID_K_PARAM,
                                               offset=LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT)  # Very high (inf) cost
        objects_cost_y = SigmoidFunctionParams(w=OBSTACLE_SIGMOID_COST, k=OBSTACLE_SIGMOID_K_PARAM,
                                               offset=LATERAL_SAFETY_MARGIN_FROM_OBJECT)  # Very high (inf) cost
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
