import numpy as np
import six
from abc import abstractmethod, ABCMeta
from decision_making.src.messages.route_plan_message import RoutePlan
from decision_making.src.planning.utils.kinematics_utils import KinematicUtils
from logging import Logger
from typing import Optional, List

import rte.python.profiler as prof
from decision_making.src.global_constants import SHOULDER_SIGMOID_OFFSET, DEVIATION_FROM_LANE_COST, \
    LANE_SIGMOID_K_PARAM, SHOULDER_SIGMOID_K_PARAM, DEVIATION_TO_SHOULDER_COST, DEVIATION_FROM_ROAD_COST, \
    ROAD_SIGMOID_K_PARAM, OBSTACLE_SIGMOID_COST, OBSTACLE_SIGMOID_K_PARAM, DEVIATION_FROM_GOAL_COST, \
    GOAL_SIGMOID_K_PARAM, GOAL_SIGMOID_OFFSET, DEVIATION_FROM_GOAL_LAT_LON_RATIO, LON_JERK_COST_WEIGHT, \
    LAT_JERK_COST_WEIGHT, VELOCITY_LIMITS, LON_ACC_LIMITS, LAT_ACC_LIMITS, LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT, \
    LATERAL_SAFETY_MARGIN_FROM_OBJECT, MINIMUM_REQUIRED_TRAJECTORY_TIME_HORIZON, LARGE_DISTANCE_FROM_SHOULDER, \
    ROAD_SHOULDERS_WIDTH, BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED, TP_DESIRED_VELOCITY_DEVIATION
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
from decision_making.src.planning.types import FS_DA, FS_SA, FS_SX, FS_DX, FrenetState2D
from decision_making.src.planning.utils.optimal_control.poly1d import QuinticPoly1D
from decision_making.src.prediction.ego_aware_prediction.ego_aware_predictor import EgoAwarePredictor
from decision_making.src.state.map_state import MapState
from decision_making.src.state.state import State, ObjectSize
from decision_making.src.utils.map_utils import MapUtils


@six.add_metaclass(ABCMeta)
class CostBasedBehavioralPlanner:
    def __init__(self, action_space: ActionSpace, recipe_evaluator: Optional[ActionRecipeEvaluator],
                 action_spec_evaluator: Optional[ActionSpecEvaluator],
                 action_spec_validator: Optional[ActionSpecFiltering],
                 value_approximator: ValueApproximator, predictor: EgoAwarePredictor, logger: Logger):
        self.action_space = action_space
        self.recipe_evaluator = recipe_evaluator
        self.action_spec_evaluator = action_spec_evaluator
        self.action_spec_validator = action_spec_validator or ActionSpecFiltering(filters=None, logger=logger)
        self.value_approximator = value_approximator
        self.predictor = predictor
        self.logger = logger

        self._last_action: Optional[ActionRecipe] = None
        self._last_action_spec: Optional[ActionSpec] = None

    @abstractmethod
    def choose_action(self, state: State, behavioral_state: BehavioralGridState, action_recipes: List[ActionRecipe],
                      recipes_mask: List[bool], route_plan: RoutePlan):
        """
        upon receiving an input state, return an action specification and its respective index in the given list of
        action recipes.
        :param recipes_mask: A list of boolean values, which are True if respective action recipe in
        input argument action_recipes is valid, else False.
        :param state: the current world state
        :param behavioral_state: processed behavioral state
        :param action_recipes: a list of enumerated semantic actions [ActionRecipe].
        :param route_plan -  a route_plane message
        :return: a tuple of the selected action index and selected action spec itself (int, ActionSpec).
        """
        pass

    @abstractmethod
    def plan(self, state: State, route_plan: RoutePlan):
        """
        Given current state and a route plan, plans the next semantic action to be carried away. This method makes
        use of Planner components such as Evaluator,Validator and Predictor for enumerating, specifying
        and evaluating actions. Its output will be further handled and used to create a trajectory in Trajectory Planner
        and has the form of TrajectoryParams, which includes the reference route, target time, target state to be in,
        cost params and strategy.
        :param state: the current world state
        :param route_plan: A route plan message
        :return: a tuple: (TrajectoryParams for TP,BehavioralVisualizationMsg for e.g. VizTool)
        """
        pass

    @prof.ProfileFunction()
    def _generate_terminal_states(self, state: State, behavioral_state: BehavioralGridState,
                                  action_specs: List[ActionSpec], mask: np.ndarray, route_plan: RoutePlan) \
            -> List[BehavioralGridState]:
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
    def _generate_trajectory_specs(behavioral_state: BehavioralGridState, action_spec: ActionSpec) -> TrajectoryParams:
        """
        Generate trajectory specification for trajectory planner given a SemanticActionSpec. This also
        generates the reference route that will be provided to the trajectory planner.
         Given the target longitude and latitude, we create a reference route in global coordinates, where:
         latitude is constant and equal to the target latitude;
         longitude starts from ego current longitude, and end in the target longitude.
        :param behavioral_state: processed behavioral state
        :return: Trajectory cost specifications [TrajectoryParameters]
        """
        ego = behavioral_state.ego_state
        # get action's unified frame (GFF)
        action_frame = behavioral_state.extended_lane_frames[action_spec.relative_lane]

        # goal Frenet state w.r.t. spec_lane_id
        projected_goal_fstate = action_spec.as_fstate()

        # calculate trajectory cost_params using original goal map_state (from the map)
        goal_segment_id, goal_segment_fstate = action_frame.convert_to_segment_state(projected_goal_fstate)
        cost_params = CostBasedBehavioralPlanner._generate_cost_params(map_state=MapState(goal_segment_fstate, goal_segment_id),
                                                                       ego_size=ego.size)
        # Calculate cartesian coordinates of action_spec's target (according to target-lane frenet_frame)
        goal_cstate = action_frame.fstate_to_cstate(projected_goal_fstate)

        goal_time = action_spec.t + ego.timestamp_in_sec
        trajectory_end_time = max(MINIMUM_REQUIRED_TRAJECTORY_TIME_HORIZON + ego.timestamp_in_sec, goal_time)

        # create TrajectoryParams for TP
        trajectory_parameters = TrajectoryParams(reference_route=action_frame,
                                                 target_time=goal_time,
                                                 target_state=goal_cstate,
                                                 cost_params=cost_params,
                                                 strategy=TrajectoryPlanningStrategy.HIGHWAY,
                                                 trajectory_end_time=trajectory_end_time,
                                                 bp_time=ego.timestamp)

        return trajectory_parameters

    @staticmethod
    @prof.ProfileFunction()
    def generate_baseline_trajectory(timestamp: float, action_spec: ActionSpec, trajectory_parameters: TrajectoryParams,
                                     ego_fstate: FrenetState2D) -> SamplableTrajectory:
        """
        Creates a SamplableTrajectory as a reference trajectory for a given ActionSpec, assuming T_d=T_s
        :param timestamp: [s] ego timestamp in seconds
        :param action_spec: action specification that contains all relevant info about the action's terminal state
        :param trajectory_parameters: the parameters (of the required trajectory) that will be sent to TP
        :param ego_fstate: ego Frenet state w.r.t. reference_route
        :return: a SamplableWerlingTrajectory object
        """
        # Note: We create the samplable trajectory as a reference trajectory of the current action.
        goal_fstate = action_spec.as_fstate()
        if action_spec.only_padding_mode:
            # in case of very short action, create samplable trajectory using linear polynomials from ego time,
            # such that it passes through the goal at goal time
            ego_by_goal_state = KinematicUtils.create_ego_by_goal_state(goal_fstate, action_spec.t)
            poly_coefs_s, poly_coefs_d = KinematicUtils.create_linear_profile_polynomial_pair(ego_by_goal_state)
        else:
            # We assume correctness only of the longitudinal axis, and set T_d to be equal to T_s.
            A_inv = np.linalg.inv(QuinticPoly1D.time_constraints_matrix(action_spec.t))

            constraints_s = np.concatenate((ego_fstate[FS_SX:(FS_SA + 1)], goal_fstate[FS_SX:(FS_SA + 1)]))
            constraints_d = np.concatenate((ego_fstate[FS_DX:(FS_DA + 1)], goal_fstate[FS_DX:(FS_DA + 1)]))

            poly_coefs_s = QuinticPoly1D.solve(A_inv, constraints_s[np.newaxis, :])[0]
            poly_coefs_d = QuinticPoly1D.solve(A_inv, constraints_d[np.newaxis, :])[0]

        minimal_horizon = trajectory_parameters.trajectory_end_time - timestamp

        return SamplableWerlingTrajectory(timestamp_in_sec=timestamp,
                                          T_s=action_spec.t,
                                          T_d=action_spec.t,
                                          T_extended=minimal_horizon,
                                          frenet_frame=trajectory_parameters.reference_route,
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

        is_rightmost_lane = MapUtils.get_lane_ordinal(map_state.lane_id) == 0
        is_leftmost_lane = (len(MapUtils.get_adjacent_lane_ids(map_state.lane_id, RelativeLane.LEFT_LANE)) == 0)

        # TODO: here we assume a constant lane width from the current state to the goal
        dist_from_right_lane_border, dist_from_left_lane_border = \
            MapUtils.get_dist_to_lane_borders(map_state.lane_id, map_state.lane_fstate[FS_SX])

        # the following two will dictate a cost for being too close to the road border, this is irrelevant when
        # when being on a lane which is not the rightmost or the leftmost, so we override it with a big value.
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
                                           lon_jerk_cost_weight=LON_JERK_COST_WEIGHT,
                                           lat_jerk_cost_weight=LAT_JERK_COST_WEIGHT,
                                           velocity_limits=VELOCITY_LIMITS,
                                           lon_acceleration_limits=LON_ACC_LIMITS,
                                           lat_acceleration_limits=LAT_ACC_LIMITS,
                                           desired_velocity=BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED +
                                                            TP_DESIRED_VELOCITY_DEVIATION)

        return cost_params
