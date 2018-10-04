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
    LAT_ACC_LIMITS, LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT, LATERAL_SAFETY_MARGIN_FROM_OBJECT, SAFETY_MARGIN_TIME_DELAY, \
    TRAJECTORY_TIME_RESOLUTION, EPS, SPECIFICATION_MARGIN_TIME_DELAY, DX_OFFSET_MIN, DX_OFFSET_MAX, TD_STEPS
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
from decision_making.src.planning.trajectory.frenet_constraints import FrenetConstraints
from decision_making.src.planning.trajectory.samplable_trajectory import SamplableTrajectory
from decision_making.src.planning.trajectory.samplable_werling_trajectory import SamplableWerlingTrajectory
from decision_making.src.planning.trajectory.trajectory_planning_strategy import TrajectoryPlanningStrategy
from decision_making.src.planning.trajectory.werling_planner import WerlingPlanner
from decision_making.src.planning.types import FS_DA, FS_SA, FS_SX, FS_DX, LIMIT_MAX, FS_SV, FS_DV, FrenetState2D
from decision_making.src.planning.utils.optimal_control.poly1d import QuinticPoly1D
from decision_making.src.planning.utils.safety_utils import SafetyUtils
from decision_making.src.prediction.ego_aware_prediction.ego_aware_predictor import EgoAwarePredictor
from decision_making.src.state.map_state import MapState
from decision_making.src.state.state import State, ObjectSize, EgoState
from decision_making.src.utils.map_utils import MapUtils
from decision_making.src.utils.metric_logger import MetricLogger
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
        if len(state.dynamic_objects) == 0:
            return list(action_specs_mask)

        # convert the specifications list to 2D matrix, where rows represent different specifications
        spec_arr = np.array([[spec.t, spec.s, spec.v, spec.d] for i, spec in enumerate(action_specs)
                             if action_specs_mask[i]])
        T_s_arr, s_arr, v_arr, d_arr = np.split(spec_arr, 4, axis=1)
        T_s_arr, s_arr, v_arr, d_arr = T_s_arr.flatten(), s_arr.flatten(), v_arr.flatten(), d_arr.flatten()

        ego = state.ego_state
        ego_init_fstate = ego.map_state.road_fstate
        lane_width = MapService.get_instance().get_road(ego.map_state.road_id).lane_width
        road_frenet = MapService.get_instance()._rhs_roads_frenet[ego.map_state.road_id]

        # duplicate initial frenet states and create target frenet states based on the specifications
        zeros = np.zeros(T_s_arr.shape[0])
        init_fstates = np.tile(ego_init_fstate, T_s_arr.shape[0]).reshape(T_s_arr.shape[0], 6)
        target_fstates = np.c_[s_arr, v_arr, zeros, d_arr, zeros, zeros]

        # calculate A_inv_d as a concatenation of inverse matrices for maximal T_d (= T_s) and for minimal T_d
        A_inv_s = np.linalg.inv(QuinticPoly1D.time_constraints_tensor(T_s_arr))

        T_d_arr = np.concatenate([CostBasedBehavioralPlanner._calc_T_d_grid(ego_init_fstate[FS_DX:], d, T_s_arr[i])
                                    for i, d in enumerate(d_arr)])
        A_inv_d = np.linalg.inv(QuinticPoly1D.time_constraints_tensor(T_d_arr))

        # create ftrajectories_s and duplicated ftrajectories_d (for max_T_d and min_T_d)
        constraints_s = np.concatenate((init_fstates[:, :FS_DX], target_fstates[:, :FS_DX]), axis=1)
        constraints_d = np.concatenate((init_fstates[:, FS_DX:], target_fstates[:, FS_DX:]), axis=1)
        duplicated_constraints_d = np.repeat(constraints_d, TD_STEPS, axis=0)
        poly_coefs_s = QuinticPoly1D.zip_solve(A_inv_s, constraints_s)
        poly_coefs_d = QuinticPoly1D.zip_solve(A_inv_d, duplicated_constraints_d)
        time_points = np.arange(0, np.max(T_s_arr) + EPS, TRAJECTORY_TIME_RESOLUTION)
        ftrajectories_s = QuinticPoly1D.polyval_with_derivatives(poly_coefs_s, time_points)
        ftrajectories_d = QuinticPoly1D.polyval_with_derivatives(poly_coefs_d, time_points)
        # for any T_d < T_s, complement ftrajectories_d to the length of T_s by adding states with zero lateral velocity
        last_t_d_indices = np.floor(T_d_arr / TRAJECTORY_TIME_RESOLUTION).astype(int)
        for i, ftrajectory_d in enumerate(ftrajectories_d):
            ftrajectory_d[(last_t_d_indices[i] + 1):] = np.array([ftrajectory_d[last_t_d_indices[i], 0], 0, 0])

        # set all points beyond spec.t at infinity, such that they will be safe and will not affect the result
        end_traj_indices = np.floor(T_s_arr / TRAJECTORY_TIME_RESOLUTION).astype(int)
        for i, ftrajectory_s in enumerate(ftrajectories_s):
            ftrajectory_s[(end_traj_indices[i] + 1):] = np.array([road_frenet.s_limits[1] - road_frenet.ds, VELOCITY_LIMITS[1], 0])
        # duplicate ftrajectories_s TD_STEPS times to be aligned with ftrajectories_d
        duplicated_ftrajectories_s = np.repeat(ftrajectories_s, TD_STEPS, axis=0)

        # create full Frenet trajectories
        ftrajectories = np.concatenate((duplicated_ftrajectories_s, ftrajectories_d), axis=-1)

        # filter the trajectories by Frenet and Cartesian fitlers, like in TP
        cost_params = TrajectoryCostParams(None, None, None, None, None, None, None, None, None, None, None, None,
                                           VELOCITY_LIMITS, LON_ACC_LIMITS, LAT_ACC_LIMITS)
        frenet_filtered_indices = WerlingPlanner._filter_by_frenet_limits(
            ftrajectories, poly_coefs_d, T_d_arr, cost_params, road_frenet.s_limits)
        ctrajectories = road_frenet.ftrajectories_to_ctrajectories(ftrajectories[frenet_filtered_indices])
        cartesian_refiltered_indices = WerlingPlanner._filter_by_cartesian_limits(ctrajectories, cost_params)
        refiltered_indices = frenet_filtered_indices[cartesian_refiltered_indices]

        # for each action leave only min_T_d and max_T_d among all refiltered_indices
        min_T_d = np.repeat(np.inf, T_s_arr.shape[0])
        max_T_d = np.repeat(-np.inf, T_s_arr.shape[0])
        refiltered_action_indices = np.floor(refiltered_indices.astype(float) / TD_STEPS).astype(int)
        for i, ftraj_idx in enumerate(refiltered_indices):
            action_idx = refiltered_action_indices[i]
            min_T_d[action_idx] = min(min_T_d[action_idx], ftraj_idx)
            max_T_d[action_idx] = max(max_T_d[action_idx], ftraj_idx)

        # predict objects' trajectories
        obj_fstates = np.array([obj.map_state.road_fstate for obj in state.dynamic_objects])
        obj_sizes = [obj.size for obj in state.dynamic_objects]
        obj_trajectories = np.array(self.predictor.predict_frenet_states(obj_fstates, time_points))

        # verify that terminal state of any (static) action keeps distance of at least 2 sec from the front object
        keep_distance_trajectories = np.full(T_s_arr.shape[0], True)
        for action_idx in range(T_s_arr.shape[0]):
            end_traj_idx = int(T_s_arr[action_idx] / TRAJECTORY_TIME_RESOLUTION)
            for obj_idx, obj_trajectory in enumerate(obj_trajectories):
                end_dist_from_obj = target_fstates[action_idx] - obj_trajectory[end_traj_idx]
                if end_dist_from_obj[FS_SX] > 0 and abs(end_dist_from_obj[FS_DX]) < lane_width / 2:
                    min_dist = SPECIFICATION_MARGIN_TIME_DELAY * obj_trajectory[end_traj_idx, FS_SV] + \
                               (ego.size.length + obj_sizes[obj_idx].length) / 2.
                    keep_distance_trajectories[action_idx] &= (end_dist_from_obj[FS_SX] >= min_dist)

        # extract min_T_d and max_T_d indices for keeping distance trajectories
        filtered_indices = []
        for action_idx in range(T_s_arr.shape[0]):
            if not np.isinf(min_T_d[action_idx]) and keep_distance_trajectories[action_idx]:
                filtered_indices.append(min_T_d[action_idx])
                if min_T_d[action_idx] != max_T_d[action_idx]:
                    filtered_indices.append(max_T_d[action_idx])
        filtered_indices = np.array(filtered_indices).astype(int)

        # calculate safety for each trajectory, each object, each timestamp
        safety_costs = SafetyUtils.get_safety_costs(ftrajectories[filtered_indices], ego.size, obj_trajectories, obj_sizes)
        # trajectory is considered safe if it's safe wrt all dynamic objects for all timestamps
        safe_filtered_trajectories = (safety_costs < 1).all(axis=(1, 2))
        safe_indices = filtered_indices[safe_filtered_trajectories]
        # OR between safe trajectories for max_d and safe trajectories for min_d
        safe_trajectories = np.full(T_s_arr.shape[0], False)
        safe_trajectories[np.floor(safe_indices.astype(float) / TD_STEPS).astype(int)] = True
        if not safe_trajectories.any():
            self.logger.warning("CostBasedBehavioralPlanner._check_actions_safety: No safe action found")

        CostBasedBehavioralPlanner.log_safety(ego_init_fstate, obj_fstates, ego.size, obj_sizes, lane_width, ego.timestamp_in_sec)

        # assign safety to the specs, for which specs_mask is true
        safe_specs = np.copy(np.array(action_specs_mask))
        safe_specs[safe_specs] = safe_trajectories
        return list(safe_specs)  # list's size like the original action_specs size

    @staticmethod
    def _calc_T_d_grid(fstate_d: np.array, target_d: float, T_s: float) -> np.array:
        """
        Calculate the lower bound of the lateral time horizon T_d_low_bound and return a grid of possible lateral
        planning time values.
        :param fstate_d: 1D array containing: current latitude, lateral velocity and lateral acceleration
        :param target_d: [m] target latitude
        :param T_s: [m] longitudinal time horizon
        :return: numpy array (1D) of the possible lateral planning horizons
        """
        dt = TRAJECTORY_TIME_RESOLUTION
        dx = min(abs(fstate_d[0] - (target_d - DX_OFFSET_MIN)), abs(fstate_d[0] - (target_d + DX_OFFSET_MAX)))
        fconstraints_t0 = FrenetConstraints(0, 0, 0, dx, fstate_d[1], fstate_d[2])
        fconstraints_tT = FrenetConstraints(0, 0, 0, 0, 0, 0)
        lower_bound_T_d = WerlingPlanner.low_bound_lat_horizon(fconstraints_t0, fconstraints_tT, dt)
        T_d_grid = WerlingPlanner._create_lat_horizon_grid(T_s, lower_bound_T_d, dt)
        if len(T_d_grid) < TD_STEPS:  # make T_d_grid to be of size TD_STEPS
            T_d_grid = np.concatenate((T_d_grid, np.full(TD_STEPS - len(T_d_grid), T_d_grid[-1])))
        elif len(T_d_grid) > TD_STEPS:
            T_d_grid = T_d_grid[:TD_STEPS]
        return T_d_grid

    @staticmethod
    def log_safety(ego_fstate: FrenetState2D, obj_fstates: np.array, ego_size: ObjectSize, obj_sizes: np.array,
                   lane_width: float, time: float):
        """
        The logging used to debug safety
        :param ego_fstate:
        :param obj_fstates:
        :param ego_size:
        :param obj_sizes:
        :param lane_width:
        :param time: the current timestamp
        """
        MetricLogger.init('Safety')
        ml = MetricLogger.get_logger()

        actual_lon_distance = np.zeros(obj_fstates.shape[0])
        min_safe_lon_distance = np.zeros(obj_fstates.shape[0])
        obj_size_arr = np.zeros((obj_fstates.shape[0], 2))
        front_obj_dist = np.zeros(obj_fstates.shape[0])

        # for each object, calculate actual longitudinal distance and minimal safe longitudinal distance
        for i, obj_fstate in enumerate(obj_fstates):
            cars_size_lon_margin = (ego_size.length + obj_sizes[i].length) / 2
            if obj_fstates[i, FS_SX] > ego_fstate[FS_SX]:
                actual_lon_distance[i] = obj_fstates[i, FS_SX] - ego_fstate[FS_SX]
                min_safe_lon_distance[i] = max(0, ego_fstate[FS_SV] ** 2 - obj_fstates[i, FS_SV] ** 2) / \
                                           (-2 * LON_ACC_LIMITS[0]) + \
                                           ego_fstate[FS_SV] * SAFETY_MARGIN_TIME_DELAY + cars_size_lon_margin
            else:
                actual_lon_distance[i] = ego_fstate[FS_SX] - obj_fstates[i, FS_SX]
                min_safe_lon_distance[i] = max(0, obj_fstates[i, FS_SV] ** 2 - ego_fstate[FS_SV] ** 2) / \
                                           (-2 * LON_ACC_LIMITS[0]) + \
                                           obj_fstates[i, FS_SV] * SPECIFICATION_MARGIN_TIME_DELAY + cars_size_lon_margin
            obj_size_arr[i] = np.array([obj_sizes[i].length, obj_sizes[i].width])

            lat_dist = abs(ego_fstate[FS_DX] - obj_fstate[FS_DX])
            front_obj_dist[i] = actual_lon_distance[i] + lat_dist \
                if actual_lon_distance[i] > 0 and lat_dist < lane_width/2 else np.inf

        front_obj_idx = np.argmin(front_obj_dist)

        # calculate components of lateral RSS safety formula
        lat_relative_to_obj = obj_fstates[:, FS_DX] - ego_fstate[FS_DX]
        sign_of_lat_relative_to_obj = np.sign(lat_relative_to_obj)
        ego_vel_after_reaction_time = ego_fstate[FS_DV] - sign_of_lat_relative_to_obj * SAFETY_MARGIN_TIME_DELAY
        obj_vel_after_reaction_time = obj_fstates[:, FS_DV] + sign_of_lat_relative_to_obj * SPECIFICATION_MARGIN_TIME_DELAY
        # the distance objects move one towards another during their reaction time
        avg_ego_vel = 0.5 * (ego_fstate[FS_DV] + ego_vel_after_reaction_time)
        avg_obj_vel = 0.5 * (obj_fstates[:, FS_DV] + obj_vel_after_reaction_time)
        reaction_dist = sign_of_lat_relative_to_obj * (avg_obj_vel * SPECIFICATION_MARGIN_TIME_DELAY -
                                                       avg_ego_vel * SAFETY_MARGIN_TIME_DELAY)
        actual_lat_distance = np.abs(lat_relative_to_obj)
        min_safe_lat_dist = np.maximum(np.divide(sign_of_lat_relative_to_obj *
                                                 (obj_vel_after_reaction_time * np.abs(obj_vel_after_reaction_time) -
                                                  ego_vel_after_reaction_time * np.abs(ego_vel_after_reaction_time)),
                                                 2 * LAT_ACC_LIMITS[1]) + reaction_dist, 0) + \
                            (ego_size.width + obj_size_arr[:, 1]) / 2

        # write the data to the MetricLogger
        if not np.isinf(front_obj_dist[front_obj_idx]):
            ml.bind(time=time, ego_sx=ego_fstate[FS_SX], ego_sv=ego_fstate[FS_SV], ego_dx=ego_fstate[FS_DX])
            ml.bind(actual_lon_dist=actual_lon_distance[front_obj_idx], min_safe_lon_dist=min_safe_lon_distance[front_obj_idx])
            ml.bind(actual_lat_dist=actual_lat_distance, min_safe_lat_dist=min_safe_lat_dist)
        for i, obj_fstate in enumerate(obj_fstates):
            ml.bind(obj_fstate=obj_fstate[:4])
