from abc import abstractmethod, ABCMeta
from logging import Logger
from typing import Optional, List

import numpy as np
import six

import rte.python.profiler as prof
from decision_making.src.global_constants import PREDICTION_LOOKAHEAD_COMPENSATION_RATIO, TRAJECTORY_ARCLEN_RESOLUTION, \
    SHOULDER_SIGMOID_OFFSET, DEVIATION_FROM_LANE_COST, LANE_SIGMOID_K_PARAM, SHOULDER_SIGMOID_K_PARAM, \
    DEVIATION_TO_SHOULDER_COST, DEVIATION_FROM_ROAD_COST, ROAD_SIGMOID_K_PARAM, OBSTACLE_SIGMOID_COST, \
    OBSTACLE_SIGMOID_K_PARAM, DEVIATION_FROM_GOAL_COST, GOAL_SIGMOID_K_PARAM, GOAL_SIGMOID_OFFSET, \
    DEVIATION_FROM_GOAL_LAT_LON_RATIO, LON_JERK_COST_WEIGHT, LAT_JERK_COST_WEIGHT, VELOCITY_LIMITS, LON_ACC_LIMITS, \
    LAT_ACC_LIMITS, SAFETY_MARGIN_TIME_DELAY, TRAJECTORY_TIME_RESOLUTION, EPS
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
from decision_making.src.planning.trajectory.samplable_trajectory import SamplableTrajectory
from decision_making.src.planning.trajectory.samplable_werling_trajectory import SamplableWerlingTrajectory
from decision_making.src.planning.trajectory.trajectory_planning_strategy import TrajectoryPlanningStrategy
from decision_making.src.planning.types import FS_DA, FS_SA, FS_SX, FS_DX, LIMIT_MAX
from decision_making.src.planning.utils.optimal_control.poly1d import QuinticPoly1D
from decision_making.src.planning.utils.safety_utils import SafetyUtils
from decision_making.src.prediction.ego_aware_prediction.ego_aware_predictor import EgoAwarePredictor
from decision_making.src.state.map_state import MapState
from decision_making.src.state.state import State, ObjectSize, EgoState
from decision_making.src.utils.map_utils import MapUtils
from mapping.src.model.constants import ROAD_SHOULDERS_WIDTH
from mapping.src.service.map_service import MapService


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
        # create a new behavioral state at the action end
        ego = state.ego_state

        # TODO: assumes everyone on the same road!
        road_id = ego.map_state.road_id
        actions_horizons = np.array([spec.t for i, spec in enumerate(action_specs) if mask[i]])
        terminal_timestamps = ego.timestamp_in_sec + actions_horizons

        objects_curr_fstates = np.array(
            [dynamic_object.map_state.road_fstate for dynamic_object in state.dynamic_objects])
        objects_terminal_fstates = self.predictor.predict_frenet_states(objects_curr_fstates, actions_horizons)

        # Create ego states, dynamic objects, states and finally behavioral states
        terminal_ego_states = [ego.clone_from_map_state(MapState([spec.s, spec.v, 0, spec.d, 0, 0], road_id),
                                                        ego.timestamp_in_sec + spec.t)
                               for i, spec in enumerate(action_specs) if mask[i]]
        terminal_dynamic_objects = [
            [dynamic_object.clone_from_map_state(MapState(objects_terminal_fstates[i][j], road_id))
             for i, dynamic_object in enumerate(state.dynamic_objects)]
            for j, terminal_timestamp in enumerate(terminal_timestamps)]
        terminal_states = [
            state.clone_with(dynamic_objects=terminal_dynamic_objects[i], ego_state=terminal_ego_states[i])
            for i in range(len(terminal_ego_states))]

        valid_behavioral_grid_states = (BehavioralGridState.create_from_state(terminal_state, self.logger)
                                        for terminal_state in terminal_states)
        terminal_behavioral_states = [valid_behavioral_grid_states.__next__() if m else None for m in mask]
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

        # Get road details
        road_id = ego.map_state.road_id

        # Add a margin to the lookahead path of dynamic objects to avoid extrapolation
        # caused by the curve linearization approximation in the resampling process
        # The compensation here is multiplicative because of the different curve-fittings we use:
        # in BP we use piecewise-linear and in TP we use cubic-fit.
        # Due to that, a point's longitude-value will be different between the 2 curves.
        # This error is accumulated depending on the actual length of the curvature -
        # when it is long, the error will potentially be big.
        lookahead_distance = action_spec.s * PREDICTION_LOOKAHEAD_COMPENSATION_RATIO + \
            action_spec.v * SAFETY_MARGIN_TIME_DELAY + action_spec.v ** 2 / (2*LAT_ACC_LIMITS[LIMIT_MAX])

        # TODO: figure out how to solve the issue of lagging ego-vehicle (relative to reference route)
        # TODO: better than sending the whole road. Fix when map service is redesigned!
        center_lane_reference_route = MapService.get_instance().get_uniform_path_lookahead(
            road_id=road_id,
            lat_shift=action_spec.d,  # THIS ASSUMES THE GOAL ALWAYS FALLS ON THE REFERENCE ROUTE
            starting_lon=0,
            lon_step=TRAJECTORY_ARCLEN_RESOLUTION,
            steps_num=int(np.ceil(lookahead_distance / TRAJECTORY_ARCLEN_RESOLUTION)),
            navigation_plan=navigation_plan)

        # The frenet frame used in specify (RightHandSide of road)
        rhs_frenet = MapService.get_instance()._rhs_roads_frenet[ego.map_state.road_id]
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
    def generate_baseline_trajectory(ego: EgoState, action_spec: ActionSpec) -> SamplableTrajectory:
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

    def _check_actions_safety(self, state: State, action_specs: List[ActionSpec], action_specs_mask: np.array) \
            -> List[bool]:
        """
        Check RSS safety for all action specs, for which action_specs_mask is true.
        An action spec is considered safe if it's safe wrt all dynamic objects for all timestamps < spec.t.
        :param state: the current world state
        :param action_specs: list of action specifications
        :param action_specs_mask: 1D mask vector (boolean) for filtering valid action specifications
        :return: boolean list of safe specifications. The list's size is equal to the original action_specs size.
        Specifications filtered by action_specs_mask are considered "unsafe".
        """
        # TODO: in the current version T_d = T_s. Test safety for different values of T_d.
        ego = state.ego_state
        ego_init_fstate = ego.map_state.road_fstate

        spec_arr = np.array([np.array([spec.t, spec.s, spec.v, spec.d])
                             for i, spec in enumerate(action_specs) if action_specs_mask[i]])
        t_arr, s_arr, v_arr, d_arr = np.split(spec_arr, 4, axis=1)
        zeros = np.zeros(t_arr.shape[0])

        init_fstates = np.tile(ego_init_fstate, t_arr.shape[0]).reshape(t_arr.shape[0], 6)
        target_fstates = np.c_[s_arr, v_arr, zeros, d_arr, zeros, zeros]

        A_inv = np.linalg.inv(QuinticPoly1D.time_constraints_tensor(t_arr))

        constraints_s = np.concatenate((init_fstates[:, :FS_DX], target_fstates[:, :FS_DX]), axis=1)
        constraints_d = np.concatenate((init_fstates[:, FS_DX:], target_fstates[:, FS_DX:]), axis=1)

        poly_coefs_s = QuinticPoly1D.zip_solve(A_inv, constraints_s)
        poly_coefs_d = QuinticPoly1D.zip_solve(A_inv, constraints_d)

        time_points = np.arange(0, np.max(t_arr) + EPS, TRAJECTORY_TIME_RESOLUTION)
        fstates_s = QuinticPoly1D.polyval_with_derivatives(poly_coefs_s, time_points)
        fstates_d = QuinticPoly1D.polyval_with_derivatives(poly_coefs_d, time_points)  # T_d = T_s
        ftrajectories = np.concatenate((fstates_s, fstates_d), axis=-1)

        # set all points beyond spec.t at infinity, such that they will be safe and will not affect the result
        for i, ftrajectory in enumerate(ftrajectories):
            end_t_idx = int(t_arr[i] / TRAJECTORY_TIME_RESOLUTION)
            ftrajectory[end_t_idx:, FS_SX] = np.inf

        # predict objects' trajectories
        obj_fstates = np.array([obj.map_state.road_fstate for obj in state.dynamic_objects])
        obj_sizes = [obj.size for obj in state.dynamic_objects]
        obj_trajectories = np.array(self.predictor.predict_frenet_states(obj_fstates, time_points))

        # calculate safety for each trajectory, each object, each timestamp
        safe_times = SafetyUtils.get_safe_times(ftrajectories, ego.size, obj_trajectories, obj_sizes)
        # trajectory is considered safe if it's safe wrt all dynamic objects for all timestamps
        safe_trajectories = safe_times.all(axis=(1, 2))

        # assign safety to the specs, for which specs_mask is true
        safe_specs = np.copy(np.array(action_specs_mask))
        safe_specs[safe_specs] = safe_trajectories
        return list(safe_specs)  # list's size like the original action_specs size
