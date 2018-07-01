from abc import abstractmethod, ABCMeta
import rte.python.profiler as prof
from logging import Logger
import numpy as np
import six

from typing import Optional, List
from decision_making.src.global_constants import PREDICTION_LOOKAHEAD_COMPENSATION_RATIO, TRAJECTORY_ARCLEN_RESOLUTION, \
    SHOULDER_SIGMOID_OFFSET, DEVIATION_FROM_LANE_COST, LANE_SIGMOID_K_PARAM, SHOULDER_SIGMOID_K_PARAM, \
    DEVIATION_TO_SHOULDER_COST, DEVIATION_FROM_ROAD_COST, ROAD_SIGMOID_K_PARAM, OBSTACLE_SIGMOID_COST, \
    OBSTACLE_SIGMOID_K_PARAM, DEVIATION_FROM_GOAL_COST, GOAL_SIGMOID_K_PARAM, GOAL_SIGMOID_OFFSET, \
    DEVIATION_FROM_GOAL_LAT_LON_RATIO, LON_JERK_COST_WEIGHT, LAT_JERK_COST_WEIGHT, VELOCITY_LIMITS, LON_ACC_LIMITS, \
    LAT_ACC_LIMITS
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
from decision_making.src.planning.trajectory.fixed_trajectory_planner import FixedSamplableTrajectory
from decision_making.src.planning.trajectory.trajectory_planner import SamplableTrajectory
from decision_making.src.planning.trajectory.trajectory_planning_strategy import TrajectoryPlanningStrategy
from decision_making.src.planning.trajectory.werling_planner import SamplableWerlingTrajectory
from decision_making.src.planning.types import FS_DA, FS_SA, FS_SX, FS_DX
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame
from decision_making.src.planning.utils.optimal_control.poly1d import QuinticPoly1D
from decision_making.src.prediction.ego_aware_prediction.road_following_predictor import RoadFollowingPredictor
from decision_making.src.state.map_state import MapState
from decision_making.src.state.state import State, ObjectSize, NewEgoState
from decision_making.src.utils.map_utils import MapUtils
from mapping.src.model.constants import ROAD_SHOULDERS_WIDTH
from mapping.src.service.map_service import MapService


@six.add_metaclass(ABCMeta)
class CostBasedBehavioralPlanner:
    def __init__(self, action_space: ActionSpace, recipe_evaluator: Optional[ActionRecipeEvaluator],
                 action_spec_evaluator: Optional[ActionSpecEvaluator],
                 action_spec_validator: Optional[ActionSpecFiltering],
                 value_approximator: ValueApproximator, predictor: RoadFollowingPredictor, logger: Logger):
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

    @prof.ProfileFunction()
    def _generate_terminal_states(self, state: State, action_specs: List[ActionSpec], mask: np.ndarray) -> \
            List[BehavioralGridState]:
        """
        Given current state and action specifications, generate a corresponding list of future states using the
        predictor. Uses mask over list of action specifications to avoid unnecessary computation
        :param state: the current world state
        :param action_specs: list of action specifications
        :param mask: 1D mask vector (boolean) for filtering valid action specifications
        :return: a list of terminal states
        """
        # create a new behavioral state at the action end
        ego = state.ego_state

        # TODO: assumes everyone on the same road!
        road_id = ego.map_state.road_id
        total_action_time = np.array([spec.t for i, spec in enumerate(action_specs) if mask[i]])
        terminal_timestamps = ego.timestamp_in_sec + total_action_time

        objects_curr_fstates = np.array([dynamic_object.map_state.road_fstate for dynamic_object in state.dynamic_objects])
        objects_terminal_fstates = self.predictor._vectorized_predict_objects(objects_curr_fstates, total_action_time)

        # Create ego states, dynamic objects, states and finally behavioral states
        terminal_ego_states = [ego.clone_from_map_state(MapState([spec.s, spec.v, 0, spec.d, 0, 0], road_id),
                                                        ego.timestamp_in_sec + spec.t)
                                                        for i, spec in enumerate(action_specs) if mask[i]]
        terminal_dynamic_objects = [[dynamic_object.clone_from_map_state(MapState(objects_terminal_fstates[i][j], road_id))
                                 for i, dynamic_object in enumerate(state.dynamic_objects)]
                                for j, terminal_timestamp in enumerate(terminal_timestamps)]
        terminal_states = [state.clone_with(dynamic_objects=terminal_dynamic_objects[i], ego_state=terminal_ego_states[i])
                           for i in range(len(terminal_ego_states))]

        terminal_behavioral_states = []
        cnt = 0
        for i, spec in enumerate(action_specs):
            if mask[i]:
                terminal_behavioral_states.append(
                    BehavioralGridState.create_from_state(terminal_states[cnt], self.logger))
                cnt += 1
            else:
                terminal_behavioral_states.append(None)
        return terminal_behavioral_states

    @prof.ProfileFunction()
    def _generate_old_terminal_states(self, state: State, action_specs: List[ActionSpec], mask: np.ndarray) -> \
            List[BehavioralGridState]:
        """
        Given current state and action specifications, generate a corresponding list of future states using the
        predictor. Uses mask over list of action specifications to avoid unnecessary computation
        :param state: the current world state
        :param action_specs: list of action specifications
        :param mask: 1D mask vector (boolean) for filtering valid action specifications
        :return: a list of terminal states
        """
        # create a new behavioral state at the action end
        ego = state.ego_state
        # TODO: assumes everyone on the same road!
        road_id = ego.map_state.road_id

        terminal_behavioral_states = []
        for i, spec in enumerate(action_specs):
            # For invalid actions (masked out), return None
            if not mask[i]:
                terminal_behavioral_states.append(None)
                continue

            # predict ego (s,d,v_s are according to action_spec)
            terminal_timestamp = ego.timestamp_in_sec + spec.t
            terminal_ego_fstate = np.array([spec.s, spec.v, 0, spec.d, 0, 0])
            terminal_ego_cstate = [1, 1, 1, 1, 1, 1]  # MapUtils.convert_map_to_cartesian_state(MapState(terminal_ego_fstate, road_id))
            terminal_ego_state_trajectory = FixedSamplableTrajectory(np.array([np.array(terminal_ego_cstate)]), terminal_timestamp)
            terminal_state = self.predictor.predict_state(state=state, prediction_timestamps=np.array([terminal_timestamp]),
                                                          action_trajectory=terminal_ego_state_trajectory)
            new_behavioral_state = BehavioralGridState.create_from_state(terminal_state[0], self.logger)
            terminal_behavioral_states.append(new_behavioral_state)

        return terminal_behavioral_states

    @staticmethod
    @prof.ProfileFunction()
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
        ego = behavioral_state.ego_state

        # Add a margin to the lookahead path of dynamic objects to avoid extrapolation
        # caused by the curve linearization approximation in the resampling process
        # The compensation here is multiplicative because of the different curve-fittings we use:
        # in BP we use piecewise-linear and in TP we use cubic-fit.
        # Due to that, a point's longitude-value will be different between the 2 curves.
        # This error is accumulated depending on the actual length of the curvature -
        # when it is long, the error will potentially be big.
        lookahead_distance = action_spec.s * PREDICTION_LOOKAHEAD_COMPENSATION_RATIO

        # TODO: figure out how to solve the issue of lagging ego-vehicle (relative to reference route)
        # TODO: better than sending the whole road. Fix when map service is redesigned!
        center_lane_reference_route = MapService.get_instance().get_uniform_path_lookahead(
            road_id=ego.map_state.road_id,
            lat_shift=action_spec.d,  # THIS ASSUMES THE GOAL ALWAYS FALLS ON THE REFERENCE ROUTE
            starting_lon=0,
            lon_step=TRAJECTORY_ARCLEN_RESOLUTION,
            steps_num=int(np.ceil(lookahead_distance / TRAJECTORY_ARCLEN_RESOLUTION)),
            navigation_plan=navigation_plan)

        # The frenet frame used in specify (RightHandSide of road)
        rhs_reference_route = MapService.get_instance().get_uniform_path_lookahead(
            road_id=ego.map_state.road_id,
            lat_shift=0,
            starting_lon=0,
            lon_step=TRAJECTORY_ARCLEN_RESOLUTION,
            steps_num=int(np.ceil(lookahead_distance / TRAJECTORY_ARCLEN_RESOLUTION)),
            navigation_plan=navigation_plan)
        rhs_frenet = FrenetSerret2DFrame(rhs_reference_route)

        # Get road details
        road_id = ego.map_state.road_id

        # Convert goal state from rhs-frenet-frame to center-lane-frenet-frame
        goal_cstate = rhs_frenet.fstate_to_cstate(np.array([action_spec.s, action_spec.v, 0, action_spec.d, 0, 0]))

        cost_params = CostBasedBehavioralPlanner._generate_cost_params(
            road_id=road_id,
            ego_size=ego.size,
            reference_route_latitude=action_spec.d  # this assumes the target falls on the reference route
        )

        trajectory_parameters = TrajectoryParams(reference_route=center_lane_reference_route,
                                                 time=action_spec.t + ego.timestamp_in_sec,
                                                 target_state=goal_cstate,
                                                 cost_params=cost_params,
                                                 strategy=TrajectoryPlanningStrategy.HIGHWAY)

        return trajectory_parameters

    @staticmethod
    @prof.ProfileFunction()
    def generate_baseline_trajectory(ego: NewEgoState, action_spec: ActionSpec) -> SamplableTrajectory:
        """
        Creates a SamplableTrajectory as a reference trajectory for a given ActionSpec, assuming T_d=T_s
        :param ego: ego object
        :param action_spec: action specification that contains all relevant info about the action's terminal state
        :return: a SamplableWerlingTrajectory object
        """
        # Note: We create the samplable trajectory as a reference trajectory of the current action.from
        # We assume correctness only of the longitudinal axis, and set T_d to be equal to T_s.

        # project ego vehicle onto the road
        ego_init_fstate = ego.map_state.road_fstate

        target_fstate = np.array([action_spec.s, action_spec.v, 0, action_spec.d, 0, 0])

        A_inv = np.linalg.inv(QuinticPoly1D.time_constraints_matrix(action_spec.t))

        constraints_s = np.concatenate((ego_init_fstate[FS_SX:(FS_SA + 1)], target_fstate[FS_SX:(FS_SA + 1)]))
        constraints_d = np.concatenate((ego_init_fstate[FS_DX:(FS_DA + 1)], target_fstate[FS_DX:(FS_DA + 1)]))

        poly_coefs_s = QuinticPoly1D.solve(A_inv, constraints_s[np.newaxis, :])[0]
        poly_coefs_d = QuinticPoly1D.solve(A_inv, constraints_d[np.newaxis, :])[0]

        road_frenet = MapUtils.get_road_rhs_frenet(ego)

        return SamplableWerlingTrajectory(timestamp_in_sec=ego.timestamp_in_sec,
                                          T_s=action_spec.t,
                                          T_d=action_spec.t,
                                          frenet_frame=road_frenet,
                                          poly_s_coefs=poly_coefs_s,
                                          poly_d_coefs=poly_coefs_d)

    @staticmethod
    def _generate_cost_params(road_id: int, ego_size: ObjectSize, reference_route_latitude: float) -> \
            TrajectoryCostParams:
        """
        Generate cost specification for trajectory planner
        :param road_id: the road's id - it currently assumes a single road for the whole action.
        :param ego_size: ego size used to extract margins (for dilation of other objects on road)
        :param reference_route_latitude: the latitude of the reference route. This is used to compute out-of-lane cost
        :return: a TrajectoryCostParams instance that encodes all parameters for TP cost computation.
        """
        road = MapService.get_instance().get_road(road_id)
        target_lane_num = int(reference_route_latitude / road.lane_width)

        # lateral distance in [m] from ref. path to rightmost edge of lane
        right_lane_offset = max(0.0, reference_route_latitude - ego_size.width / 2 - target_lane_num * road.lane_width)
        # lateral distance in [m] from ref. path to leftmost edge of lane
        left_lane_offset = (road.road_width - reference_route_latitude) - ego_size.width / 2 - \
                           (road.lanes_num - target_lane_num - 1) * road.lane_width
        # as stated above, for shoulders
        right_shoulder_offset = reference_route_latitude - ego_size.width / 2 + SHOULDER_SIGMOID_OFFSET
        # as stated above, for shoulders
        left_shoulder_offset = (road.road_width - reference_route_latitude) - ego_size.width / 2 + \
                               SHOULDER_SIGMOID_OFFSET
        # as stated above, for whole road including shoulders
        right_road_offset = reference_route_latitude - ego_size.width / 2 + ROAD_SHOULDERS_WIDTH
        # as stated above, for whole road including shoulders
        left_road_offset = (road.road_width - reference_route_latitude) - ego_size.width / 2 + ROAD_SHOULDERS_WIDTH

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
