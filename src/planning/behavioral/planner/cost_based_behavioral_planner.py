import numpy as np
import six
from abc import abstractmethod, ABCMeta
from logging import Logger
from typing import Optional, List, Dict

import rte.python.profiler as prof
from decision_making.src.global_constants import SHOULDER_SIGMOID_OFFSET, DEVIATION_FROM_LANE_COST, \
    LANE_SIGMOID_K_PARAM, SHOULDER_SIGMOID_K_PARAM, DEVIATION_TO_SHOULDER_COST, DEVIATION_FROM_ROAD_COST, \
    ROAD_SIGMOID_K_PARAM, OBSTACLE_SIGMOID_COST, OBSTACLE_SIGMOID_K_PARAM, DEVIATION_FROM_GOAL_COST, \
    GOAL_SIGMOID_K_PARAM, GOAL_SIGMOID_OFFSET, DEVIATION_FROM_GOAL_LAT_LON_RATIO, LON_JERK_COST_WEIGHT, \
    LAT_JERK_COST_WEIGHT, VELOCITY_LIMITS, LON_ACC_LIMITS, LAT_ACC_LIMITS, LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT, \
    LATERAL_SAFETY_MARGIN_FROM_OBJECT, PLANNING_LOOKAHEAD_DIST, MAX_HORIZON_DISTANCE
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
from decision_making.src.planning.types import FS_DA, FS_SA, FS_SX, FS_DX, FrenetState2D, FrenetStates2D
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame
from decision_making.src.planning.utils.generalized_frenet_serret_frame import GeneralizedFrenetSerretFrame
from decision_making.src.planning.utils.optimal_control.poly1d import QuinticPoly1D
from decision_making.src.prediction.ego_aware_prediction.ego_aware_predictor import EgoAwarePredictor
from decision_making.src.state.map_state import MapState
from decision_making.src.state.state import State, ObjectSize, DynamicObject
from decision_making.src.utils.map_utils import MapUtils
from mapping.src.model.constants import ROAD_SHOULDERS_WIDTH


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
                      recipes_mask: List[bool], nav_plan: NavigationPlanMsg):
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
    def _generate_terminal_states(self, state: State, behavioral_state: BehavioralGridState,
                                  action_specs: List[ActionSpec], mask: np.ndarray, nav_plan: NavigationPlanMsg) \
            -> List[BehavioralGridState]:
        """
        Given current state and action specifications, generate a corresponding list of future states using the
        predictor. Uses mask over list of action specifications to avoid unnecessary computation
        :param state: the current world state
        :param action_specs: list of action specifications
        :param mask: 1D mask vector (boolean) for filtering valid action specifications
        :return: a list of terminal states
        """
        ego = state.ego_state
        relative_lane_ids = MapUtils.get_closest_lane_ids(ego.map_state.lane_id)
        # collect terminal / specs' lane_ids and fstates wrt these lanes
        valid_specs = np.array([spec for i, spec in enumerate(action_specs) if mask[i]])
        # get lane_ids adjacent to ego according to specs' relative lanes
        lane_ids_per_spec = np.array([relative_lane_ids[spec.relative_lane] for spec in valid_specs])
        action_horizons = np.array([spec.t for spec in valid_specs])

        terminal_ego_states = np.full(len(valid_specs), None)
        terminal_dynamic_objects = np.full((len(valid_specs), len(state.dynamic_objects)), None)

        # calculate terminal ego fstates w.r.t. 3 extended_lane_frames
        for rel_lane, extended_lane_frame in behavioral_state.extended_lane_frames.items():
            # find all specs, whose target belongs to the current extended_lane_frame
            relevant_spec_idxs = extended_lane_frame.has_segment_ids(lane_ids_per_spec)
            # create terminal ego states for those specs, whose target is no the extended_lane_frame
            terminal_ego_states[relevant_spec_idxs] = self._create_terminal_ego_states_for_lane(
                state, extended_lane_frame, valid_specs[relevant_spec_idxs])

        # Since at terminal states objects from more than 3 closest lanes may be relevant, but also from their neighbor
        # lanes, calculate extended_lane_frames for all lanes, whose lateral distance from ego lane <= 2 lanes.
        # The number of such lanes is at most 5.
        extended_lane_frames = BehavioralGridState.create_extended_lane_frames(state, nav_plan, adjacent_lanes_num=2)

        # calculate terminal predicted objects fstates on (at most) 5 adjacent extended_lane_frames
        for _, extended_lane_frame in extended_lane_frames.items():
            # collect objects' lane_ids
            all_objects_lane_ids = np.array([dynamic_object.map_state.lane_id for dynamic_object in state.dynamic_objects])
            # find all objects that belong to the current extended_lane_frames
            relevant_object_idxs = extended_lane_frame.has_segment_ids(all_objects_lane_ids)
            if relevant_object_idxs.any():
                terminal_dynamic_objects[:, relevant_object_idxs] = \
                    self._create_terminal_objects_states_for_lane(state, extended_lane_frame, action_horizons, relevant_object_idxs)

        # create full terminal states
        terminal_states = [state.clone_with(dynamic_objects=[obj for obj in terminal_dynamic_objects[i] if obj is not None],
                                            ego_state=terminal_ego_states[i])
                           for i in range(len(valid_specs))]

        # for each spec calculate dictionary of its 3 closest extended frames
        extended_frames_per_spec = [{RelativeLane(rel_lane_val - spec.relative_lane.value):
                                     frame for rel_lane_val, frame in extended_lane_frames.items()
                                     if abs(rel_lane_val - spec.relative_lane.value) <= 1}
                                    for spec in valid_specs]

        # create behavioral states from states
        valid_behavioral_grid_states = (BehavioralGridState.create_from_state(terminal_state, nav_plan, self.logger,
                                                                              extended_frames_per_spec[i])
                                        for i, terminal_state in enumerate(terminal_states))
        terminal_behavioral_states = [valid_behavioral_grid_states.__next__() if m else None for m in mask]
        return terminal_behavioral_states

    def _create_terminal_ego_states_for_lane(self, state: State, extended_lane_frame: GeneralizedFrenetSerretFrame,
                                             actions_specs: List[ActionSpec]) -> np.array:
        """
        Create terminal ego states given: (1) frame of relative lane; (2) action_specs whose target is that lane.
        Used by _generate_terminal_states.
        :param state: contains ego state & dynamic objects
        :param extended_lane_frame: generalized Frenet frame (GFF) of the given single target lane
        :param actions_specs: subset of the action_specs whose target is on the given target lane
        :return: array of terminal ego states with predicted dynamic objects
        """
        ego = state.ego_state
        ego_terminal_fstates = np.array([[spec.s, spec.v, 0, spec.d, 0, 0] for spec in actions_specs])
        action_horizons = np.array([spec.t for spec in actions_specs])

        # convert the obtained ego terminal states to the segments (ids & fstates)
        ego_terminal_segment_ids, ego_terminal_segment_fstates = extended_lane_frame.convert_to_segment_states(ego_terminal_fstates)

        # create states from ego states and dynamic objects
        return np.array([ego.clone_from_map_state(MapState(ego_terminal_segment_fstates[i], ego_terminal_segment_ids[i]),
                                                  ego.timestamp_in_sec + action_horizons[i])
                         for i in range(len(actions_specs))])

    def _create_terminal_objects_states_for_lane(self, state: State, extended_lane_frame: GeneralizedFrenetSerretFrame,
                                                 action_horizons: np.array, relevant_object_idxs: np.array) -> np.array:
        """
        Create terminal dynamic objects, given: (1) frame of relative lane; (2) all action horizons; (3) indices of
        objects from the given lane. Used by _generate_terminal_states.
        :param state:
        :param extended_lane_frame: generalized Frenet frame (GFF) of the given single target lane
        :param action_horizons: all action horizons (from the valid action_specs)
        :param relevant_object_idxs: array of indices of state.dynamic_objects that are located on the given lane
        :return: 2D array of terminal dynamic objects of shape: len(action_horizons) x len(relevant_object_idxs).
        """
        # collect segment map states (from SP) of the objects located on the given target lane
        relevant_dynamic_objects = [obj for i, obj in enumerate(state.dynamic_objects) if relevant_object_idxs[i]]

        # collect objects' current fstates and lane_ids
        current_segment_ids = [object.map_state.lane_id for object in relevant_dynamic_objects]
        current_segment_fstates = np.array([object.map_state.lane_fstate for object in relevant_dynamic_objects])

        # convert segment map states to the extended frame fstates
        current_extended_fstates = extended_lane_frame.convert_from_segment_states(current_segment_fstates, current_segment_ids)

        # predict the relevant objects for all specs
        terminal_fstates = self.predictor.predict_frenet_states(current_extended_fstates, action_horizons)

        # allocate memory for the objects' terminal segment map states
        terminal_segment_ids = np.full((len(relevant_dynamic_objects), len(action_horizons)), None)
        terminal_segment_fstates = np.empty((len(relevant_dynamic_objects), len(action_horizons), 6), dtype=float)

        # convert the objects' terminal extended_frame fstates back to the segments map states (ids & fstates)
        # convert only those predictions that are located inside extended_lane_frame; the rest get segment_id = None
        legal_objects = (terminal_fstates[..., FS_SX] < extended_lane_frame.s_max)
        terminal_segment_ids[legal_objects], terminal_segment_fstates[legal_objects] = \
            extended_lane_frame.convert_to_segment_states(terminal_fstates[legal_objects])

        # create 2D array of dynamic_objects:
        # the external dimension is on all specs, the internal dimension is on terminal objects for the current spec
        ego_timestamp = state.ego_state.timestamp_in_sec
        return np.array([[obj.clone_from_map_state(MapState(terminal_segment_fstates[i, j], terminal_segment_ids[i, j]),
                                                   ego_timestamp + action_horizons[j])
                          if terminal_segment_ids[i, j] is not None else None
                          for i, obj in enumerate(relevant_dynamic_objects)]
                         for j in range(len(action_horizons))])

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

        # create TrajectoryParams for TP
        trajectory_parameters = TrajectoryParams(reference_route=action_frame,
                                                 time=action_spec.t + ego.timestamp_in_sec,
                                                 target_state=goal_cstate,
                                                 cost_params=cost_params,
                                                 strategy=TrajectoryPlanningStrategy.HIGHWAY,
                                                 bp_time=ego.timestamp)

        return trajectory_parameters

    @staticmethod
    @prof.ProfileFunction()
    def generate_baseline_trajectory(timestamp: float, action_spec: ActionSpec, reference_route: FrenetSerret2DFrame,
                                     ego_fstate: FrenetState2D) -> SamplableTrajectory:
        """
        Creates a SamplableTrajectory as a reference trajectory for a given ActionSpec, assuming T_d=T_s
        :param timestamp: [s] ego timestamp in seconds
        :param action_spec: action specification that contains all relevant info about the action's terminal state
        :param reference_route: the reference Frenet frame sent to TP
        :param ego_fstate: ego Frenet state w.r.t. reference_route
        :return: a SamplableWerlingTrajectory object
        """
        # Note: We create the samplable trajectory as a reference trajectory of the current action.from
        # We assume correctness only of the longitudinal axis, and set T_d to be equal to T_s.
        A_inv = np.linalg.inv(QuinticPoly1D.time_constraints_matrix(action_spec.t))
        goal_fstate = action_spec.as_fstate()

        constraints_s = np.concatenate((ego_fstate[FS_SX:(FS_SA + 1)], goal_fstate[FS_SX:(FS_SA + 1)]))
        constraints_d = np.concatenate((ego_fstate[FS_DX:(FS_DA + 1)], goal_fstate[FS_DX:(FS_DA + 1)]))

        poly_coefs_s = QuinticPoly1D.solve(A_inv, constraints_s[np.newaxis, :])[0]
        poly_coefs_d = QuinticPoly1D.solve(A_inv, constraints_d[np.newaxis, :])[0]

        return SamplableWerlingTrajectory(timestamp_in_sec=timestamp,
                                          T_s=action_spec.t,
                                          T_d=action_spec.t,
                                          frenet_frame=reference_route,
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
        # TODO: here we assume a constant lane width from the current state to the goal
        dist_from_right_lane_border, dist_from_left_lane_border = \
            MapUtils.get_dist_to_lane_borders(map_state.lane_id, map_state.lane_fstate[FS_SX])
        dist_from_right_road_border, dist_from_left_road_border = \
            MapUtils.get_dist_to_road_borders(map_state.lane_id, map_state.lane_fstate[FS_SX])

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
